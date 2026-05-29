import argparse
import torch
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import ArabicCharTokenizer,parse_khatt_dataset,ManuscriptPreprocessor,ArabicOCRDataset,DataLoader,collate_fn
import yaml
from model import ScriptFormer
from evaluation import compute_metrics, print_evaluation_report
from postprocessing import ArabicPostProcessor
from tqdm import tqdm


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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",type=str,default="configs/default.yml")
    parser.add_argument("--checkpoint",type=str,default="checkpoint/best_model.pt")
    parser.add_argument("--tokenizer",type=str,default=None)
    parser.add_argument("--show_samples",type=int,default=5)
    parser.add_argument("--max-length",type=int,default=None)
    parser.add_argument("--beam-size",type=int,default=None)
    parser.add_argument("--no_postprocess",action="store_true")
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")

    if "config" in checkpoint and checkpoint["config"] and "model" in checkpoint["config"]:
        config = checkpoint["config"]
        print(f"Using config embedded in checkpoint")
    else:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
        print(f"Using config from {args.config}")

    device = config["project"]["device"]
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        print("CUDA not available, falling back to CPU")

    tokenizer_path = args.tokenizer or os.path.join(
        os.path.dirname(args.checkpoint), "tokenizer.json"
    )
    tokenizer_cfg = dict(config.get("tokenizer", {}).get("normalization", {}))
    tokenizer_cfg["normalize_alef"] = False
    if os.path.exists(tokenizer_path):
        tokenizer = ArabicCharTokenizer(**tokenizer_cfg)
        tokenizer.load(tokenizer_path)
        print(f"Loaded tokenizer from {tokenizer_path}")
    else:
        raise FileNotFoundError(
            f"Tokenizer not found at {tokenizer_path}. "
            "Copy tokenizer.json from the Kaggle training artifacts or pass --tokenizer."
        )
    print(f"Vocab size: {tokenizer.vocab_size}")

    print(f"Loading model from {args.checkpoint}...")
    dec_cfg = config["model"]["decoder"]
    state = checkpoint["model_state_dict"]

    layer_keys = [k for k in state if k.startswith("decoder.decoder.layers.")]
    layer_indices = set(int(k.split(".")[3]) for k in layer_keys)
    num_layers = len(layer_indices) if layer_indices else dec_cfg["num_layers"]
    hidden_size = state["decoder.token_embedding.weight"].shape[1]
    ff_key = "decoder.decoder.layers.0.linear1.weight"
    ff_size = state[ff_key].shape[0] if ff_key in state else dec_cfg["feedforward_size"]
    num_heads = dec_cfg.get("num_heads", hidden_size // 32)

    print(f"  Detected: {num_layers} layers, hidden={hidden_size}, ff={ff_size}, heads={num_heads}")

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
            f"  Vocab mismatch detected (checkpoint={checkpoint_vocab_size}, "
            f"tokenizer={tokenizer_vocab_size})."
        )
        print("  Resizing decoder vocab tensors to match tokenizer vocab size.")
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

    loaded = model.load_state_dict(state_to_load, strict=False)
    if loaded.missing_keys:
        print(f"  Warning: missing model keys in checkpoint: {loaded.missing_keys}")
    if loaded.unexpected_keys:
        print(f"  Warning: unexpected keys in checkpoint: {loaded.unexpected_keys}")
    model = model.to(device)
    model.eval()
    print(f"  Loaded epoch {checkpoint['epoch']}, val_loss={checkpoint['val_loss']:.4f}")

    data = parse_khatt_dataset(config["data"]["raw_dir"] + "/KHATT")
    val_samples = data["val"]
    if not val_samples:
        raise FileNotFoundError(
            "No validation samples were found. "
            "Make sure the KHATT validation CSV and images are available under data/raw/KHATT, "
            "or run prediction with scripts/predict.py instead of evaluation."
        )

    preprocessor = ManuscriptPreprocessor(config["preprocessing"])
    val_dataset = ArabicOCRDataset(
        val_samples, tokenizer, preprocessor,
        config["data"]["image"]["height"],
        config["data"]["image"]["width"],
        config["model"]["decoder"]["max_length"],
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, pad_id=tokenizer.pad_id),
        num_workers=0,
    )
    print(f"  Val samples: {len(val_dataset)}")

    max_len = args.max_length or config["model"]["decoder"]["max_length"]
    all_raw_predictions = []  
    all_predictions = []     
    all_references = []

    postprocessor = None
    if not args.no_postprocess:
        postprocessor_cfg = dict(tokenizer_cfg)
        postprocessor_cfg.update({
            "fix_repetitions": False,
            "clean_punctuation": False,
            "normalize_alef": False,
        })
        postprocessor = ArabicPostProcessor(**postprocessor_cfg)
    if postprocessor:
        print(f"  Postprocessing: {postprocessor.describe()}")

    beam_size = args.beam_size or config.get("evaluation", {}).get("beam_size", 1)

    print(f"\nRunning inference (max_length={max_len}, beam_size={beam_size})...")
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            images = batch["images"].to(device)
            texts = batch["texts"] 

            generated_ids = model.generate(images, max_length=max_len, beam_size=beam_size)

            for token_ids in generated_ids:
                raw_text = tokenizer.decode(token_ids.tolist())
                all_raw_predictions.append(raw_text)

                if postprocessor:
                    all_predictions.append(postprocessor(raw_text))
                else:
                    all_predictions.append(raw_text)

            all_references.extend(texts)

    normalized_references = [postprocessor(text) for text in all_references] if postprocessor else all_references

    print(f"\nComputing metrics on {len(all_predictions)} samples...\n")

    if postprocessor:
        raw_metrics = compute_metrics(all_raw_predictions, normalized_references)
        pp_metrics = compute_metrics(all_predictions, normalized_references)

        print("=" * 60)
        print("  WITHOUT postprocessing:")
        print(f"    CER: {raw_metrics['cer']:.4f}  ({raw_metrics['cer']*100:.1f}%)")
        print(f"    WER: {raw_metrics['wer']:.4f}  ({raw_metrics['wer']*100:.1f}%)")
        print()
        print("  WITH postprocessing:")
        print(f"    CER: {pp_metrics['cer']:.4f}  ({pp_metrics['cer']*100:.1f}%)")
        print(f"    WER: {pp_metrics['wer']:.4f}  ({pp_metrics['wer']*100:.1f}%)")

        cer_improvement = (raw_metrics['cer'] - pp_metrics['cer']) * 100
        wer_improvement = (raw_metrics['wer'] - pp_metrics['wer']) * 100
        print()
        print(f"  Improvement: CER {cer_improvement:+.2f}pp, WER {wer_improvement:+.2f}pp")
        print("=" * 60)
        print()

        metrics = pp_metrics
    else:
        metrics = compute_metrics(all_predictions, all_references)

    print_evaluation_report(
        metrics,
        show_samples=args.show_samples,
        predictions=all_predictions,
        references=all_references,
    )


if __name__ == "__main__":
    main()


