import os
import glob

import cv2
import yaml
import torch

from preprocessing import ManuscriptPreprocessor
from data import ArabicCharTokenizer
from model import ScriptFormer
from postprocessing import ArabicPostProcessor


class OCRPipeline:

    def __init__(
        self,
        model: ScriptFormer,
        tokenizer: ArabicCharTokenizer,
        preprocessor: ManuscriptPreprocessor,
        postprocessor: ArabicPostProcessor = None,
        device: str = "cpu",
        image_height: int = 64,
        image_width: int = 384,
        max_length: int = 128,
    ):
        self.model = model.to(device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor or ArabicPostProcessor()
        self.device = device
        self.image_height = image_height
        self.image_width = image_width
        self.max_length = max_length

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        config_path: str = "configs/default.yaml",
        device: str = None,
        postprocessor: ArabicPostProcessor = None,
    ) -> "OCRPipeline":
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        if "config" in checkpoint and checkpoint["config"] and "model" in checkpoint["config"]:
            config = checkpoint["config"]
        else:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        checkpoint_dir = os.path.dirname(checkpoint_path)
        tokenizer_path = os.path.join(checkpoint_dir, "tokenizer.json")

        if os.path.exists(tokenizer_path):
            tokenizer = ArabicCharTokenizer()
            tokenizer.load(tokenizer_path)
        else:
            raise FileNotFoundError(
                f"Tokenizer not found at {tokenizer_path}. "
                "Make sure tokenizer.json was saved during training."
            )

        dec_cfg = config["model"]["decoder"]
        state = checkpoint["model_state_dict"]

        layer_keys = [k for k in state if k.startswith("decoder.decoder.layers.")]
        layer_indices = set(int(k.split(".")[3]) for k in layer_keys)
        num_layers = len(layer_indices) if layer_indices else dec_cfg["num_layers"]

        hidden_size = state["decoder.token_embedding.weight"].shape[1]

        ff_key = "decoder.decoder.layers.0.linear1.weight"
        ff_size = state[ff_key].shape[0] if ff_key in state else dec_cfg["feedforward_size"]

        attn_key = "decoder.decoder.layers.0.self_attn.in_proj_weight"
        if attn_key in state:
            num_heads = dec_cfg.get("num_heads", hidden_size // 32)
        else:
            num_heads = dec_cfg["num_heads"]

        model = ScriptFormer(
            vocab_size=tokenizer.vocab_size,
            encoder_hidden=hidden_size,
            decoder_hidden=hidden_size,
            decoder_layers=num_layers,
            decoder_heads=num_heads,
            decoder_ff=ff_size,
            max_length=dec_cfg["max_length"],
            dropout=dec_cfg["dropout"],
            pad_id=tokenizer.pad_id,
            sos_id=tokenizer.sos_id,
            eos_id=tokenizer.eos_id,
        )

        model.load_state_dict(checkpoint["model_state_dict"])

        preprocessor = ManuscriptPreprocessor(config["preprocessing"])

        return cls(
            model=model,
            tokenizer=tokenizer,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            device=device,
            image_height=config["data"]["image"]["height"],
            image_width=config["data"]["image"]["width"],
            max_length=dec_cfg["max_length"],
        )

    def _load_and_preprocess(self, image_path: str) -> torch.Tensor:
        raw = cv2.imread(image_path)
        if raw is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")

        processed = self.preprocessor(
            raw,
            target_height=self.image_height,
            target_width=self.image_width,
        )

        tensor = torch.from_numpy(processed).unsqueeze(0).unsqueeze(0).float()
        return tensor.to(self.device)

    def predict(self, image_path: str, max_length: int = None) -> str:
        if max_length is None:
            max_length = self.max_length

        image_tensor = self._load_and_preprocess(image_path)

        with torch.no_grad():
            generated_ids = self.model.generate(image_tensor, max_length=max_length)

        text = self.tokenizer.decode(generated_ids[0].tolist())
        text = self.postprocessor(text)

        return text

    def predict_batch(self, image_paths: list[str], max_length: int = None) -> list[dict]:
        results = []
        for path in image_paths:
            try:
                text = self.predict(path, max_length=max_length)
                results.append({"path": path, "text": text})
            except Exception as e:
                results.append({"path": path, "text": "", "error": str(e)})
        return results

    def predict_directory(
        self,
        directory: str,
        extensions: tuple = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"),
        max_length: int = None,
    ) -> list[dict]:
        image_paths = []
        for ext in extensions:
            image_paths.extend(glob.glob(os.path.join(directory, f"*{ext}")))
            image_paths.extend(glob.glob(os.path.join(directory, f"*{ext.upper()}")))

        image_paths = sorted(set(image_paths))

        if not image_paths:
            return []

        return self.predict_batch(image_paths, max_length=max_length)