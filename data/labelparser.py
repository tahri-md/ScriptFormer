import csv
import os
import hashlib
from pathlib import Path

BUCKWALTER_TO_ARABIC = {
    # === Basic Arabic Letters (28 letters) ===
    "aa":  "ا",   # Alif
    "ba":  "ب",   # Baa
    "ta":  "ت",   # Taa
    "th":  "ث",   # Thaa
    "ja":  "ج",   # Jeem
    "ha":  "ح",   # Haa
    "kh":  "خ",   # Khaa
    "da":  "د",   # Dal
    "dh":  "ذ",   # Dhal
    "ra":  "ر",   # Raa
    "za":  "ز",   # Zayn
    "se":  "س",   # Seen
    "sh":  "ش",   # Sheen
    "sa":  "ص",   # Sad
    "de":  "ض",   # Dad
    "to":  "ط",   # Taa (emphatic)
    "zha": "ظ",   # Dhaa (emphatic)
    "ay":  "ع",   # Ain
    "gh":  "غ",   # Ghain
    "fa":  "ف",   # Faa
    "ka":  "ق",   # Qaf
    "ke":  "ك",   # Kaf
    "la":  "ل",   # Lam
    "ma":  "م",   # Meem
    "na":  "ن",   # Noon
    "he":  "ه",   # Haa (end)
    "wa":  "و",   # Waw
    "ya":  "ي",   # Yaa

    # === Special Arabic Forms ===
    "tee":  "ة",  # Taa Marbuta (ة) — the round taa at end of words
    "teE":  "ة",  # Taa Marbuta (variant annotation)
    "ah":   "أ",  # Alif with Hamza above
    "ae":   "إ",  # Alif with Hamza below
    "ee":   "ئ",  # Yaa with Hamza
    "hh":   "ء",  # Hamza alone
    "al":   "ى",  # Alif Maqsura (looks like yaa without dots)
    "laaa": "لا", # Lam-Alif ligature
    "laae": "لإ", # Lam-Alif with Hamza below ligature
    "laah": "لأ", # Lam-Alif with Hamza above ligature
    "laam": "لآ", # Lam-Alif Madda ligature
    "wl":   "ؤ",  # Waw with Hamza

    # === Punctuation & Symbols ===
    "sp":   " ",  # Space
    "dot":  ".",  # Period / full stop
    "com":  ",",  # Comma (Arabic comma ، could also be used)
    "col":  ":",  # Colon
    "scr":  "؛",  # Arabic semicolon
    "am":   "—",  # Dash / em-dash
    "dbq":  '"',  # Double quote
    "bro":  "(",  # Open bracket
    "brc":  ")",  # Close bracket
    "hyp":  "-",  # Hyphen
    "per":  "%",  # Percent

    # === Numbers ===
    "n0": "٠",  # Arabic-Indic digit 0
    "n1": "١",  # Arabic-Indic digit 1
    "n2": "٢",  # Arabic-Indic digit 2
    "n3": "٣",  # Arabic-Indic digit 3
    "n4": "٤",  # Arabic-Indic digit 4
    "n5": "٥",  # Arabic-Indic digit 5
    "n6": "٦",  # Arabic-Indic digit 6
    "n7": "٧",  # Arabic-Indic digit 7
    "n8": "٨",  # Arabic-Indic digit 8
    "n9": "٩",  # Arabic-Indic digit 9

    # === Additional punctuation (discovered from data) ===
    "fsl": "/",  # Forward slash
    "bsl": "\\", # Backslash
    "exc": "!",  # Exclamation mark
    "qts": "'",  # Single quote / apostrophe
    "equ": "=",  # Equals sign
    "usc": "_",  # Underscore
}
def codes_to_arabic(codes:list[str])->str :
    result = []
    for code in codes:
        if not code or  code == ';':
            continue
        if code in BUCKWALTER_TO_ARABIC:
            result.append(BUCKWALTER_TO_ARABIC[code])
        else :
            print("character not found")
            result.append('?')

    return "".join(result)

def parse_khatt_csv(csv_path:str,image_dir:str,image_ext:str=".jpg"):
    samples = []
    missing_count = 0
    with open(csv_path,"r",encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if not row or not row[0].strip():
                continue
            raw_filename = row[0].strip()
            base_name = os.path.splitext(raw_filename)[0]
            actual_filename = base_name+image_ext
            image_path = os.path.join(image_dir,actual_filename)
            codes = []
            for token in row[1:]:
                token = token.strip()
                if token == ";" or token == "":
                    if token ==";":
                        break
                    continue
                codes.append(token)
            
            arabic_text = codes_to_arabic(codes)

            if not os.path.isfile(image_path):
                missing_count+=1
                continue
            samples.append({
                "image_path":image_path,
                "text":arabic_text,
            })
    if missing_count >0:
        print(f"{missing_count} images that are in CSV were not found ")
    return samples    

def parse_khatt_dataset(data_root:str)->dict:
    result = {}
    root = Path(data_root)
    train_csv = root / "Train.csv"
    train_images = root / "Train_deskewed" / "Train_deskewed"
    if train_csv.exists() and train_images.exists():
        result["train"] = parse_khatt_csv(str(train_csv), str(train_images))
    else:
        print("train data not found")
        result["train"] = []

    val_csv = root / "Validation.csv"
    val_images = root / "Validate_deskewed" / "Validate_deskewed"
    if val_csv.exists() and val_images.exists():
        result["val"] = parse_khatt_csv(str(val_csv), str(val_images))
    else:
        print("validation data not found")
        result["val"] = []

    return result


def _ifnenit_token_to_base(token: str) -> str:
    token = token.strip()
    if not token:
        return ""
    if token in BUCKWALTER_TO_ARABIC:
        return token

    # IFN/ENIT AW2 tokens often include a position suffix (A/B/M/E).
    # Try removing one suffix char if that yields a known mapping.
    if len(token) > 1 and token[-1] in {"A", "B", "M", "E"}:
        candidate = token[:-1]
        if candidate in BUCKWALTER_TO_ARABIC:
            return candidate

    # Handle occasional markers like "llL" that can appear in AW2 streams.
    candidate = token.replace("llL", "")
    if candidate in BUCKWALTER_TO_ARABIC:
        return candidate

    if len(candidate) > 1 and candidate[-1] in {"A", "B", "M", "E"}:
        candidate2 = candidate[:-1]
        if candidate2 in BUCKWALTER_TO_ARABIC:
            return candidate2

    return token


def _ifnenit_aw2_to_arabic(aw2_value: str) -> str:
    tokens = [t for t in aw2_value.split("|") if t.strip()]
    out = []
    for token in tokens:
        base = _ifnenit_token_to_base(token)
        out.append(BUCKWALTER_TO_ARABIC.get(base, "?"))
    return "".join(out)


def _extract_ifnenit_aw2_label(tru_path: Path) -> str | None:
    try:
        lines = tru_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return None

    for line in lines:
        if line.startswith("LBL:") and "AW2:" in line:
            aw2_part = line.split("AW2:", 1)[1]
            aw2_value = aw2_part.split(";", 1)[0]
            if aw2_value.strip():
                return _ifnenit_aw2_to_arabic(aw2_value)
    return None


def _find_ifnenit_image(set_dir: Path, stem: str) -> str | None:
    candidates = [
        set_dir / "bmp" / f"{stem}.bmp",
        set_dir / "tif" / f"{stem}.tif",
        set_dir / "tif" / f"{stem}.tif.gz",
    ]
    for path in candidates:
        if path.is_file():
            return str(path)
    return None


def parse_ifnenit_dataset(data_root: str, val_ratio: float = 0.1) -> dict:
    """Parse IFN/ENIT dataset from data/set_*/tru + image folders.

    Splits samples deterministically by writer/page prefix to reduce near-duplicate
    leakage across train/val.
    """
    root = Path(data_root)
    data_root_dir = root / "data"
    if not data_root_dir.exists():
        print(f"IFN/ENIT data folder not found: {data_root_dir}")
        return {"train": [], "val": []}

    all_samples = []
    set_dirs = sorted([p for p in data_root_dir.glob("set_*") if p.is_dir()])
    for set_dir in set_dirs:
        tru_dir = set_dir / "tru"
        if not tru_dir.exists():
            continue
        for tru_file in sorted(tru_dir.glob("*.tru")):
            stem = tru_file.stem
            image_path = _find_ifnenit_image(set_dir, stem)
            if image_path is None:
                continue
            text = _extract_ifnenit_aw2_label(tru_file)
            if not text:
                continue
            all_samples.append({"image_path": image_path, "text": text})

    if not all_samples:
        print("No IFN/ENIT samples parsed")
        return {"train": [], "val": []}

    def in_val(sample: dict) -> bool:
        # Use writer/page prefix (e.g., ae07) for deterministic group split.
        stem = Path(sample["image_path"]).stem
        group = stem.split("_")[0]
        score = int(hashlib.md5(group.encode("utf-8")).hexdigest(), 16) % 100
        return score < int(val_ratio * 100)

    train = [s for s in all_samples if not in_val(s)]
    val = [s for s in all_samples if in_val(s)]

    # Guard against pathological tiny val set.
    if not val and train:
        cut = max(1, int(len(train) * val_ratio))
        val = train[:cut]
        train = train[cut:]

    print(f"IFN/ENIT parsed samples: train={len(train)}, val={len(val)}")
    return {"train": train, "val": val}


def parse_generated_dataset(data_root: str, val_ratio: float = 0.1) -> dict:
    """Parse the generated line-image dataset stored under raw/generated_data.

    Expected layout:
    - raw/generated_data/data.csv with columns: img_path,text
    - image paths can be relative to raw/generated_data or already absolute

    The dataset is split deterministically so the same config produces the same
    train/val partitions across runs.
    """
    root = Path(data_root)
    csv_path = root / "data.csv"
    if not csv_path.exists():
        print(f"Generated dataset CSV not found: {csv_path}")
        return {"train": [], "val": []}

    def _resolve_generated_image_path(root_dir: Path, raw_path: str) -> Path | None:
        path = Path(raw_path)
        candidates = []

        if path.is_absolute():
            candidates.append(path)
        else:
            candidates.append(root_dir / path)

        parts = path.parts
        if "line_images" in parts:
            suffix = Path(*parts[parts.index("line_images") + 1 :])
            candidates.append(root_dir / suffix)
        if len(parts) >= 2 and parts[0] == "dataset" and parts[1] == "line_images":
            candidates.append(root_dir / Path(*parts[2:]))

        candidates.append(root_dir / path.name)

        for candidate in candidates:
            if candidate.is_file():
                return candidate
        return None

    samples = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row:
                continue

            rel_image_path = (row.get("img_path") or row.get("image_path") or "").strip()
            text = (row.get("text") or "").strip()
            if not rel_image_path or not text:
                continue

            image_path = _resolve_generated_image_path(root, rel_image_path)
            if image_path is None:
                continue

            samples.append({"image_path": str(image_path), "text": text})

    if not samples:
        print(f"No generated samples parsed from {csv_path}")
        return {"train": [], "val": []}

    def in_val(sample: dict) -> bool:
        stem = Path(sample["image_path"]).stem
        score = int(hashlib.md5(stem.encode("utf-8")).hexdigest(), 16) % 100
        return score < int(val_ratio * 100)

    train = [s for s in samples if not in_val(s)]
    val = [s for s in samples if in_val(s)]

    if not val and train:
        cut = max(1, int(len(train) * val_ratio))
        val = train[:cut]
        train = train[cut:]

    print(f"Generated dataset parsed samples: train={len(train)}, val={len(val)}")
    return {"train": train, "val": val}


def parse_muharaf_dataset(data_root: str, val_ratio: float = 0.1) -> dict:
    """Parse Muharaf / public line images datasets.

    Expected layout (common variants):
    - raw/public_line_images/public/*.png and corresponding .txt files with the same stem
    - or raw/public_line_images/*.png + .txt

    The parser pairs each image with a text file of the same stem and performs a
    deterministic train/val split based on the image stem hash.
    """
    root = Path(data_root)
    # Try a few common locations
    candidates = [root, root / "public", root / "public_line_images", root / "public_line_images" / "public"]
    chosen = None
    for c in candidates:
        if c.exists() and any(c.glob("*.*")):
            chosen = c
            break

    if chosen is None:
        print(f"Muharaf/public line images folder not found under: {data_root}")
        return {"train": [], "val": []}

    image_exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    samples = []
    for img_path in sorted(chosen.glob("**/*")):
        if not img_path.is_file():
            continue
        if img_path.suffix.lower() not in image_exts:
            continue
        stem = img_path.stem
        # possible label files
        txt_candidates = [img_path.with_suffix('.txt'), img_path.with_suffix('.gt.txt'), img_path.with_suffix('.trans.txt')]
        label = None
        for t in txt_candidates:
            if t.exists():
                try:
                    label = t.read_text(encoding='utf-8', errors='replace').strip().splitlines()[0].strip()
                except Exception:
                    label = None
                break

        if label is None or label == "":
            # try sibling .txt with same name in parent
            sibling = img_path.parent / (stem + '.txt')
            if sibling.exists():
                try:
                    label = sibling.read_text(encoding='utf-8', errors='replace').strip().splitlines()[0].strip()
                except Exception:
                    label = None

        if label is None or label == "":
            continue

        samples.append({"image_path": str(img_path), "text": label})

    if not samples:
        print(f"No Muharaf/public_line_images samples parsed from {chosen}")
        return {"train": [], "val": []}

    def in_val(sample: dict) -> bool:
        stem = Path(sample["image_path"]).stem
        score = int(hashlib.md5(stem.encode("utf-8")).hexdigest(), 16) % 100
        return score < int(val_ratio * 100)

    train = [s for s in samples if not in_val(s)]
    val = [s for s in samples if in_val(s)]

    if not val and train:
        cut = max(1, int(len(train) * val_ratio))
        val = train[:cut]
        train = train[cut:]

    print(f"Muharaf/public parsed samples: train={len(train)}, val={len(val)}")
    return {"train": train, "val": val}


def load_dataset_from_config(config: dict) -> dict:
    """Load dataset(s) based on config.data.dataset.

    Supported values:
    - khatt
    - ifnenit
    - generated
    - mixed (KHATT + IFN/ENIT)
    """
    data_cfg = config.get("data", {})
    raw_dir = data_cfg.get("raw_dir", "data/raw")
    dataset_mode = str(data_cfg.get("dataset", "khatt")).lower()
    ifnenit_val_ratio = float(data_cfg.get("ifnenit_val_ratio", data_cfg.get("val_ratio", 0.1)))

    khatt_root = os.path.join(raw_dir, "KHATT")
    ifnenit_root = os.path.join(raw_dir, "ifnenit")
    generated_root = os.path.join(raw_dir, "generated_data")
    muharaf_root = os.path.join(raw_dir, "public_line_images")

    if dataset_mode == "khatt":
        return parse_khatt_dataset(khatt_root)

    if dataset_mode == "ifnenit":
        return parse_ifnenit_dataset(ifnenit_root, val_ratio=ifnenit_val_ratio)

    if dataset_mode == "generated":
        return parse_generated_dataset(generated_root, val_ratio=data_cfg.get("val_ratio", 0.1))

    if dataset_mode in {"muharaf", "public", "public_line_images"}:
        return parse_muharaf_dataset(muharaf_root, val_ratio=data_cfg.get("val_ratio", 0.1))

    if dataset_mode == "mixed":
        khatt = parse_khatt_dataset(khatt_root)
        ifnenit = parse_ifnenit_dataset(ifnenit_root, val_ratio=ifnenit_val_ratio)
        return {
            "train": khatt["train"] + ifnenit["train"],
            "val": khatt["val"] + ifnenit["val"],
        }

    raise ValueError(f"Unknown data.dataset='{dataset_mode}'. Use khatt, ifnenit, generated, or mixed.")

