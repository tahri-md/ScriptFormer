
import sys
import os
import argparse
import random

import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import parse_khatt_dataset, ArabicCharTokenizer, ArabicOCRDataset, collate_fn
from preprocessing import ManuscriptPreprocessor
from model import ScriptFormer
from training import Trainer


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.lr is not None:
        config["training"]["learning_rate"] = args.lr

    set_seed(config["project"]["seed"])

    data_root = config["data"]["raw_dir"] + "/KHATT"
    data = parse_khatt_dataset(data_root)
    train_samples = data["train"]
    val_samples = data["val"]

    tokenizer = ArabicCharTokenizer()
    all_texts = [s["text"] for s in train_samples + val_samples]
    tokenizer.build_vocab(all_texts)

    tokenizer_path = os.path.join(config["training"]["checkpoint_dir"], "tokenizer.json")
    os.makedirs(config["training"]["checkpoint_dir"], exist_ok=True)
    tokenizer.save(tokenizer_path)

    preprocessor = ManuscriptPreprocessor(config["preprocessing"])

    img_h = config["data"]["image"]["height"]
    img_w = config["data"]["image"]["width"]
    max_len = config["model"]["decoder"]["max_length"]
    batch_size = config["training"]["batch_size"]

    train_dataset = ArabicOCRDataset(
        train_samples, tokenizer, preprocessor, img_h, img_w, max_len,
    )
    val_dataset = ArabicOCRDataset(
        val_samples, tokenizer, preprocessor, img_h, img_w, max_len,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, pad_id=tokenizer.pad_id),
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, pad_id=tokenizer.pad_id),
        num_workers=0,
    )

    model = ScriptFormer(
        vocab_size=tokenizer.vocab_size,
        encoder_hidden=config["model"]["decoder"]["hidden_size"],
        decoder_hidden=config["model"]["decoder"]["hidden_size"],
        decoder_layers=config["model"]["decoder"]["num_layers"],
        decoder_heads=config["model"]["decoder"]["num_heads"],
        decoder_ff=config["model"]["decoder"]["feedforward_size"],
        max_length=max_len,
        dropout=config["model"]["decoder"]["dropout"],
        pad_id=tokenizer.pad_id,
        sos_id=tokenizer.sos_id,
        eos_id=tokenizer.eos_id,
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        tokenizer=tokenizer,
        config=config,
    )

    if args.resume:
        trainer.load_checkpoint(args.resume)

    trainer.train()


if __name__ == "__main__":
    main()