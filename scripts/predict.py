import sys
import os
import argparse
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference import OCRPipeline
from postprocessing import ArabicPostProcessor


def main():
    parser = argparse.ArgumentParser(description="ScriptFormer OCR Inference")
    parser.add_argument("--image", type=str, nargs="+", default=None)
    parser.add_argument("--dir", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--beam-size", type=int, default=None)
    parser.add_argument("--no-postprocess", action="store_true")
    parser.add_argument("--normalize-alef", action="store_true", default=True)
    parser.add_argument("--remove-diacritics", action="store_true", default=False)
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
            "normalize_alef": args.normalize_alef,
            "remove_diacritics": args.remove_diacritics,
        })
        postprocessor = ArabicPostProcessor(**postprocessor_cfg)

    pipeline = OCRPipeline.from_checkpoint(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        postprocessor=postprocessor,
        tokenizer_path=args.tokenizer,
        beam_size=args.beam_size,
    )

    if args.image:
        for path in args.image:
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


if __name__ == "__main__":
    main()