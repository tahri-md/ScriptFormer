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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",type=str,default="configs/default.yaml")
    parser.add_argument("--checkpoint",type=str,default="checkpoint/best_model.pt")
    parser.add_argument("--show_samples",type=int,default=5)
    parser.add_argument("--max-length",type=int,default=None)
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

    tokenizer_path = os.path.join(
        os.path.dirname(args.checkpoint), "tokenizer.json"
    )
    if os.path.exists(tokenizer_path):
        tokenizer = ArabicCharTokenizer()
        tokenizer.load(tokenizer_path)
        print(f"Loaded tokenizer from {tokenizer_path}")
    else:
        print("Tokenizer not found in checkpoint dir, rebuilding from data...")
        data = parse_khatt_dataset(config["data"]["raw_dir"] + "/KHATT")
        tokenizer = ArabicCharTokenizer()
        tokenizer.build_vocab(
            [s["text"] for s in data["train"] + data["val"]]
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
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    print(f"  Loaded epoch {checkpoint['epoch']}, val_loss={checkpoint['val_loss']:.4f}")

    data = parse_khatt_dataset(config["data"]["raw_dir"] + "/KHATT")
    val_samples = data["val"]

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

    postprocessor = None if args.no_postprocess else ArabicPostProcessor()
    if postprocessor:
        print(f"  Postprocessing: {postprocessor.describe()}")

    print(f"\nRunning inference (max_length={max_len})...")
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            images = batch["images"].to(device)
            texts = batch["texts"] 

            generated_ids = model.generate(images, max_length=max_len)

            for token_ids in generated_ids:
                raw_text = tokenizer.decode(token_ids.tolist())
                all_raw_predictions.append(raw_text)

                if postprocessor:
                    all_predictions.append(postprocessor(raw_text))
                else:
                    all_predictions.append(raw_text)

            all_references.extend(texts)

    print(f"\nComputing metrics on {len(all_predictions)} samples...\n")

    if postprocessor:
        raw_metrics = compute_metrics(all_raw_predictions, all_references)
        pp_metrics = compute_metrics(all_predictions, all_references)

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


