import csv
import os

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
    from pathlib import Path
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

