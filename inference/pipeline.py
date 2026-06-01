import os
import glob

import cv2
import yaml
import torch

from preprocessing import ManuscriptPreprocessor
from data import ArabicCharTokenizer
from model import ScriptFormer
from postprocessing import ArabicPostProcessor


def _resize_vocab_tensor(tensor: torch.Tensor, target_vocab_size: int) -> torch.Tensor:
    """Resize a vocab-shaped tensor by copying overlapping rows and zero-filling new rows."""
    current_vocab_size = tensor.shape[0]
    if current_vocab_size == target_vocab_size:
        return tensor

    if tensor.ndim == 2:
        resized = tensor.new_zeros((target_vocab_size, tensor.shape[1]))
        copy_rows = min(current_vocab_size, target_vocab_size)
        resized[:copy_rows] = tensor[:copy_rows]
        return resized

    if tensor.ndim == 1:
        resized = tensor.new_zeros((target_vocab_size,))
        copy_rows = min(current_vocab_size, target_vocab_size)
        resized[:copy_rows] = tensor[:copy_rows]
        return resized

    raise ValueError(f"Expected 1D or 2D vocab tensor, got shape {tuple(tensor.shape)}")


def _resize_positional_encoding_tensor(tensor: torch.Tensor, target_seq_len: int) -> torch.Tensor:
    """Resize a positional-encoding buffer to the requested sequence length."""
    if tensor.ndim != 3:
        raise ValueError(f"Expected 3D positional-encoding tensor, got shape {tuple(tensor.shape)}")

    current_seq_len = tensor.shape[1]
    if current_seq_len == target_seq_len:
        return tensor

    resized = tensor.new_zeros((tensor.shape[0], target_seq_len, tensor.shape[2]))
    copy_len = min(current_seq_len, target_seq_len)
    resized[:, :copy_len, :] = tensor[:, :copy_len, :]
    return resized


class OCRPipeline:

    def __init__(
        self,
        model: ScriptFormer,
        tokenizer: ArabicCharTokenizer,
        preprocessor: ManuscriptPreprocessor,
        postprocessor: ArabicPostProcessor = None,
        device: str = "cpu",
        image_height: int = 64,
        image_width: int = 1536,
        max_length: int = 128,
        beam_size: int = 1,
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
        self.beam_size = beam_size

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        config_path: str = "configs/default.yml",
        device: str = None,
        postprocessor: ArabicPostProcessor = None,
        tokenizer_path: str = None,
        beam_size: int = None,
        pad_alignment: str = None,
    ) -> "OCRPipeline":
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        if "config" in checkpoint and checkpoint["config"] and "model" in checkpoint["config"]:
            config = checkpoint["config"]
        else:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)

        if pad_alignment is not None:
            config = dict(config)
            config["preprocessing"] = dict(config.get("preprocessing", {}))
            config["preprocessing"]["pad_alignment"] = pad_alignment

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        checkpoint_dir = os.path.dirname(checkpoint_path)
        resolved_tokenizer_path = tokenizer_path or os.path.join(checkpoint_dir, "tokenizer.json")

        if os.path.exists(resolved_tokenizer_path):
            tokenizer = ArabicCharTokenizer()
            tokenizer.load(resolved_tokenizer_path)
        else:
            raise FileNotFoundError(
                f"Tokenizer not found at {resolved_tokenizer_path}. "
                "If you trained in Kaggle, copy tokenizer.json from the training artifacts "
                "or pass --tokenizer to point to the saved tokenizer file."
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
            encoder_type=config.get("model", {}).get("encoder", {}).get("type", "cnn"),
            encoder_model_name=config.get("model", {}).get("encoder", {}).get("model_name", "microsoft/beit-base-patch16-224-pt22k"),
            encoder_pretrained=config.get("model", {}).get("encoder", {}).get("pretrained", False),
            encoder_freeze_backbone=config.get("model", {}).get("encoder", {}).get("freeze_backbone", False),
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
            encoder_transformer_layers=config.get("model", {}).get("encoder_context", {}).get("num_layers", 0),
            encoder_transformer_heads=config.get("model", {}).get("encoder_context", {}).get("num_heads", 8),
            encoder_transformer_ff=config.get("model", {}).get("encoder_context", {}).get("feedforward_size", 512),
            encoder_transformer_dropout=config.get("model", {}).get("encoder_context", {}).get("dropout", 0.1),
        )

        checkpoint_vocab_size = state["decoder.token_embedding.weight"].shape[0]
        tokenizer_vocab_size = tokenizer.vocab_size
        state_to_load = dict(checkpoint["model_state_dict"])
        if checkpoint_vocab_size != tokenizer_vocab_size:
            print(
                f"Vocab mismatch detected (checkpoint={checkpoint_vocab_size}, "
                f"tokenizer={tokenizer_vocab_size})."
            )
            print("Resizing decoder vocab tensors to match tokenizer vocab size.")
            state_to_load["decoder.token_embedding.weight"] = _resize_vocab_tensor(
                state_to_load["decoder.token_embedding.weight"],
                tokenizer_vocab_size,
            )
            state_to_load["decoder.output_projection.weight"] = _resize_vocab_tensor(
                state_to_load["decoder.output_projection.weight"],
                tokenizer_vocab_size,
            )
            state_to_load["decoder.output_projection.bias"] = _resize_vocab_tensor(
                state_to_load["decoder.output_projection.bias"],
                tokenizer_vocab_size,
            )

        for key in list(state_to_load.keys()):
            if key.endswith("positional_encoding.pe") or key.endswith("positional_encoding.pe.weight"):
                try:
                    src = state_to_load[key]
                    parts = key.split(".")[:-1]
                    target = model
                    for part in parts:
                        target = getattr(target, part, None)
                        if target is None:
                            break
                    if target is None or not hasattr(target, "pe"):
                        continue

                    target_pe = target.pe
                    if not hasattr(src, "shape") or src.ndim != target_pe.ndim:
                        continue

                    if src.shape[1] != target_pe.shape[1]:
                        print(
                            f"Adjusting checkpoint positional-encoding '{key}': "
                            f"{src.shape[1]} -> {target_pe.shape[1]}"
                        )
                        state_to_load[key] = _resize_positional_encoding_tensor(src, target_pe.shape[1])
                except Exception:
                    # Leave the original tensor in place if anything unexpected happens.
                    continue

        loaded = model.load_state_dict(state_to_load, strict=False)
        if loaded.missing_keys:
            print(f"Vocab load: missing keys: {loaded.missing_keys}")
        if loaded.unexpected_keys:
            print(f"Vocab load: unexpected keys: {loaded.unexpected_keys}")

        preprocessor = ManuscriptPreprocessor(config["preprocessing"])

        resolved_beam_size = beam_size
        if resolved_beam_size is None:
            resolved_beam_size = config.get("evaluation", {}).get("beam_size", 1)

        return cls(
            model=model,
            tokenizer=tokenizer,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            device=device,
            image_height=config["data"]["image"]["height"],
            image_width=config["data"]["image"]["width"],
            max_length=dec_cfg["max_length"],
            beam_size=resolved_beam_size,
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

    def predict(self, image_path: str, max_length: int = None, beam_size: int = None) -> str:
        if max_length is None:
            max_length = self.max_length
        if beam_size is None:
            beam_size = self.beam_size
        image_tensor = self._load_and_preprocess(image_path)

        # Generate tokens
        with torch.no_grad():
            # compute encoder output ourselves so we can reuse it to compute logits
            encoder_output = self.model.encoder(image_tensor)
            if getattr(self.model, "encoder_transformer", None) is not None:
                encoder_output = self.model.encoder_transformer(encoder_output)

            generated_ids = self.model.generate(
                image_tensor,
                max_length=max_length,
                beam_size=beam_size,
            )

            # compute confidence: run decoder on generated sequence and compute
            # average token probability for produced tokens (excluding SOS/PAD)
            try:
                logits = self.model.decoder(encoder_output, generated_ids)
                probs = torch.softmax(logits, dim=-1)
                # gather probability of each generated token
                gen_tokens = generated_ids
                # shift to align logits (logits correspond to positions of gen_tokens)
                token_probs = probs.gather(dim=-1, index=gen_tokens.unsqueeze(-1)).squeeze(-1)
                # mask out PAD and SOS tokens for confidence averaging
                mask = (gen_tokens != self.model.pad_id) & (gen_tokens != self.model.sos_id)
                if mask.any():
                    avg_confidence = token_probs[mask].mean().item()
                else:
                    avg_confidence = float(token_probs.mean().item())
            except Exception:
                avg_confidence = 0.0

        text = self.tokenizer.decode(generated_ids[0].tolist())
        text = self.postprocessor(text)

        # Attach confidence as an attribute on return for callers that expect it
        return {"text": text, "confidence": avg_confidence, "ids": generated_ids}

    def predict_batch(self, image_paths: list[str], max_length: int = None, beam_size: int = None) -> list[dict]:
        results = []
        for path in image_paths:
            try:
                res = self.predict(path, max_length=max_length, beam_size=beam_size)
                if isinstance(res, dict):
                    results.append({"path": path, "text": res.get("text", ""), "confidence": res.get("confidence")})
                else:
                    results.append({"path": path, "text": res})
            except Exception as e:
                results.append({"path": path, "text": "", "error": str(e)})
        return results

    def predict_directory(
        self,
        directory: str,
        extensions: tuple = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"),
        max_length: int = None,
        beam_size: int = None,
    ) -> list[dict]:
        image_paths = []
        for ext in extensions:
            image_paths.extend(glob.glob(os.path.join(directory, f"*{ext}")))
            image_paths.extend(glob.glob(os.path.join(directory, f"*{ext.upper()}")))

        image_paths = sorted(set(image_paths))

        if not image_paths:
            return []

        return self.predict_batch(image_paths, max_length=max_length, beam_size=beam_size)