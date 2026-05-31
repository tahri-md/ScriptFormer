import sys
import os
import argparse
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference import OCRPipeline
from inference.attention_export import export_attention_overlays
from postprocessing import ArabicPostProcessor


def main():
    parser = argparse.ArgumentParser(description="ScriptFormer OCR Inference")
    parser.add_argument("--image", type=str, nargs="+", default=None)
    parser.add_argument("--dir", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--config", type=str, default="configs/default.yml")
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--beam-size", type=int, default=None)
    parser.add_argument("--no-postprocess", action="store_true")
    parser.add_argument("--normalize-alef", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--remove-diacritics", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--pad-alignment", type=str, choices=["left", "center", "right"], default=None)
    parser.add_argument("--export-attention-dir", type=str, default=None)
    args = parser.parse_args()

    if not args.image and not args.dir:
        parser.error("Provide either --image or --dir")

    postprocessor = None
    if not args.no_postprocess:
        tokenizer_norm = {}
        with open(args.config, "r") as f:
            tokenizer_norm = yaml.safe_load(f).get("tokenizer", {}).get("normalization", {})
        postprocessor_cfg = dict(tokenizer_norm)
        postprocessor_cfg.update({
            "fix_repetitions": False,
            "clean_punctuation": False,
        })
        if args.normalize_alef is not None:
            postprocessor_cfg["normalize_alef"] = args.normalize_alef
        if args.remove_diacritics is not None:
            postprocessor_cfg["remove_diacritics"] = args.remove_diacritics
        postprocessor = ArabicPostProcessor(**postprocessor_cfg)

    pipeline = OCRPipeline.from_checkpoint(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        postprocessor=postprocessor,
        tokenizer_path=args.tokenizer,
        beam_size=args.beam_size,
        pad_alignment=args.pad_alignment,
    )

    if args.image:
        for path in args.image:
            if args.export_attention_dir:
                if args.beam_size not in (None, 1):
                    parser.error("--export-attention-dir currently supports greedy decoding only; use --beam-size 1 or omit --beam-size.")
                manifest = export_attention_overlays(
                    pipeline,
                    path,
                    args.export_attention_dir,
                    max_length=args.max_length,
                )
                print(f"{os.path.basename(path)}: {manifest['prediction']}")
                print(f"  attention overlays saved to {manifest['export_dir']}")
            else:
                text = pipeline.predict(path, max_length=args.max_length, beam_size=args.beam_size)
                print(f"{os.path.basename(path)}: {text}")

    elif args.dir:
        results = pipeline.predict_directory(args.dir, max_length=args.max_length, beam_size=args.beam_size)
        out_path = "predictions.csv"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("filename,predicted_text\n")
            for r in results:
                filename = os.path.basename(r["path"])
                pred = r["text"].replace('"', '""').replace(',', ' ')
                print(f"{filename}: {r['text']}")
                f.write(f'"{filename}","{pred}"\n')
        print(f"\nSaved to {out_path}")

        if args.export_attention_dir:
            if args.beam_size not in (None, 1):
                parser.error("--export-attention-dir currently supports greedy decoding only; use --beam-size 1 or omit --beam-size.")
            for item in results:
                if item.get("error"):
                    print(f"Skipping {item['path']}: {item['error']}")
                    continue
                manifest = export_attention_overlays(
                    pipeline,
                    item["path"],
                    args.export_attention_dir,
                    max_length=args.max_length,
                )
                print(f"attention overlays saved to {manifest['export_dir']}")


if __name__ == "__main__":
    main()