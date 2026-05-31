import argparse
import os
import sys
import yaml
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import (
    ArabicCharTokenizer,
    load_dataset_from_config,
    ManuscriptPreprocessor,
    ArabicOCRDataset,
    DataLoader,
    collate_fn,
)
from model import ScriptFormer
from evaluation import compute_metrics, print_evaluation_report
from postprocessing import ArabicPostProcessor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yml")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--show-samples", type=int, default=5)
    parser.add_argument("--no-postprocess", action="store_true")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if "config" in checkpoint and checkpoint["config"] and "model" in checkpoint["config"]:
        config = checkpoint["config"]
    else:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)

    device = config["project"].get("device", "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        print("CUDA not available, falling back to CPU")

    tokenizer_path = args.tokenizer or os.path.join(os.path.dirname(args.checkpoint), "tokenizer.json")
    if os.path.exists(tokenizer_path):
        tokenizer = ArabicCharTokenizer(**dict(config.get("tokenizer", {}).get("normalization", {})))
        tokenizer.load(tokenizer_path)
        print(f"Loaded tokenizer from {tokenizer_path}")
    else:
        raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}")

    # Build model (mirror logic in evaluate.py / OCRPipeline)
    dec_cfg = config["model"]["decoder"]
    state = checkpoint["model_state_dict"]

    layer_keys = [k for k in state if k.startswith("decoder.decoder.layers.")]
    layer_indices = set(int(k.split(".")[3]) for k in layer_keys) if layer_keys else set()
    num_layers = len(layer_indices) if layer_indices else dec_cfg["num_layers"]
    hidden_size = state["decoder.token_embedding.weight"].shape[1]
    ff_key = "decoder.decoder.layers.0.linear1.weight"
    ff_size = state[ff_key].shape[0] if ff_key in state else dec_cfg.get("feedforward_size", 1024)
    num_heads = dec_cfg.get("num_heads", hidden_size // 32)

    print(f"  Detected: {num_layers} layers, hidden={hidden_size}, ff={ff_size}, heads={num_heads}")

    model = ScriptFormer(
        vocab_size=tokenizer.vocab_size,
        encoder_type=config.get("model", {}).get("encoder", {}).get("type", "cnn"),
        encoder_pretrained=config.get("model", {}).get("encoder", {}).get("pretrained", False),
        encoder_freeze_backbone=config.get("model", {}).get("encoder", {}).get("freeze_backbone", False),
        encoder_hidden=hidden_size,
        decoder_hidden=hidden_size,
        decoder_layers=num_layers,
        decoder_heads=num_heads,
        decoder_ff=ff_size,
        max_length=dec_cfg["max_length"],
        dropout=dec_cfg.get("dropout", 0.1),
        pad_id=tokenizer.pad_id,
        sos_id=tokenizer.sos_id,
        eos_id=tokenizer.eos_id,
        encoder_transformer_layers=config.get("model", {}).get("encoder_context", {}).get("num_layers", 0),
        encoder_transformer_heads=config.get("model", {}).get("encoder_context", {}).get("num_heads", 8),
        encoder_transformer_ff=config.get("model", {}).get("encoder_context", {}).get("feedforward_size", 512),
        encoder_transformer_dropout=config.get("model", {}).get("encoder_context", {}).get("dropout", 0.1),
    )

    # Load state with safe resizing for vocab and positional encodings
    checkpoint_vocab_size = state["decoder.token_embedding.weight"].shape[0]
    tokenizer_vocab_size = tokenizer.vocab_size
    state_to_load = dict(checkpoint["model_state_dict"])
    if checkpoint_vocab_size != tokenizer_vocab_size:
        print(f"  Vocab mismatch detected (checkpoint={checkpoint_vocab_size}, tokenizer={tokenizer_vocab_size}). Resizing.")
        def _resize(tensor, target):
            cur = tensor.shape[0]
            if tensor.ndim == 2:
                out = tensor.new_zeros((target, tensor.shape[1]))
                out[: min(cur, target)] = tensor[: min(cur, target)]
                return out
            if tensor.ndim == 1:
                out = tensor.new_zeros((target,))
                out[: min(cur, target)] = tensor[: min(cur, target)]
                return out
            return tensor
        state_to_load["decoder.token_embedding.weight"] = _resize(state_to_load["decoder.token_embedding.weight"], tokenizer_vocab_size)
        state_to_load["decoder.output_projection.weight"] = _resize(state_to_load["decoder.output_projection.weight"], tokenizer_vocab_size)
        state_to_load["decoder.output_projection.bias"] = _resize(state_to_load["decoder.output_projection.bias"], tokenizer_vocab_size)

    loaded = model.load_state_dict(state_to_load, strict=False)
    if loaded.missing_keys:
        print(f"  Warning: missing keys: {loaded.missing_keys}")
    if loaded.unexpected_keys:
        print(f"  Warning: unexpected keys: {loaded.unexpected_keys}")

    model = model.to(device)
    model.eval()

    data = load_dataset_from_config(config)
    val_samples = data.get("val", [])
    if not val_samples:
        raise FileNotFoundError("No validation samples found")

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

    postprocessor = None
    if not args.no_postprocess:
        pp_cfg = dict(config.get("tokenizer", {}).get("normalization", {}))
        pp_cfg.update({"fix_repetitions": False, "clean_punctuation": False})
        postprocessor = ArabicPostProcessor(**pp_cfg)
    if postprocessor:
        print(f"  Postprocessing: {postprocessor.describe()}")

    all_preds = []
    all_raw = []
    all_refs = []

    with torch.no_grad():
        for batch in val_loader:
            images = batch["images"].to(device)
            token_ids = batch["token_ids"].to(device)
            texts = batch["texts"]

            # encoder outputs
            encoder_output = model.encoder(images)
            if model.encoder_transformer is not None:
                encoder_output = model.encoder_transformer(encoder_output)

            # Teacher forcing: feed ground-truth inputs (all except last token)
            input_ids = token_ids[:, :-1]
            logits = model.decoder(encoder_output, input_ids)
            pred_ids = logits.argmax(dim=-1)

            for i in range(pred_ids.size(0)):
                raw = tokenizer.decode(pred_ids[i].tolist())
                all_raw.append(raw)
                if postprocessor:
                    all_preds.append(postprocessor(raw))
                else:
                    all_preds.append(raw)

            all_refs.extend(texts)

    # Compute metrics: raw predictions vs raw refs, and pp predictions vs normalized refs
    if postprocessor:
        normalized_refs = [postprocessor(r) for r in all_refs]
        raw_metrics = compute_metrics(all_raw, all_refs)
        pp_metrics = compute_metrics(all_preds, normalized_refs)
        print("\nTeacher-forcing evaluation results:\n")
        print("  RAW (predictions vs raw refs):")
        print(f"    CER: {raw_metrics['cer']:.4f}  ({raw_metrics['cer']*100:.1f}%)")
        print(f"    WER: {raw_metrics['wer']:.4f}  ({raw_metrics['wer']*100:.1f}%)")
        print()
        print("  POSTPROCESSED (predictions vs normalized refs):")
        print(f"    CER: {pp_metrics['cer']:.4f}  ({pp_metrics['cer']*100:.1f}%)")
        print(f"    WER: {pp_metrics['wer']:.4f}  ({pp_metrics['wer']*100:.1f}%)")
        metrics = pp_metrics
    else:
        metrics = compute_metrics(all_raw, all_refs)

    print_evaluation_report(metrics, show_samples=args.show_samples, predictions=all_preds, references=all_refs)


if __name__ == "__main__":
    main()
