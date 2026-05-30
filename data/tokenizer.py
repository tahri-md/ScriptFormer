from typing import Optional
import json

from postprocessing import normalize_arabic_text

class ArabicCharTokenizer:
    def __init__(
        self,
        pad_token: str = "<PAD>",
        sos_token: str = "<SOS>",
        eos_token: str = "<EOS>",
        unk_token: str = "<UNK>",
        normalize_whitespace: bool = True,
        normalize_alef: bool = False,
        normalize_taa_marbuta: bool = False,
        normalize_alef_maqsura: bool = False,
        remove_diacritics: bool = False,
        clean_punctuation: bool = False,
        strip_non_arabic: bool = False,
    ):
        self.pad_token = pad_token
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.unk_token = unk_token
        self.special_tokens = [pad_token,sos_token,eos_token,unk_token]

        self.normalize_whitespace = normalize_whitespace
        self.normalize_alef = normalize_alef
        self.normalize_taa_marbuta = normalize_taa_marbuta
        self.normalize_alef_maqsura = normalize_alef_maqsura
        self.remove_diacritics = remove_diacritics
        self.clean_punctuation = clean_punctuation
        self.strip_non_arabic = strip_non_arabic
        
        self.char_to_id: dict[str,int] = {}
        self.id_to_char: dict[int,str] = {}
        self.vocab_size: int = 0

    def _normalize_text(self, text: str) -> str:
        return normalize_arabic_text(
            text,
            normalize_whitespace=self.normalize_whitespace,
            normalize_alef=self.normalize_alef,
            normalize_taa_marbuta=self.normalize_taa_marbuta,
            normalize_alef_maqsura=self.normalize_alef_maqsura,
            remove_diacritics=self.remove_diacritics,
            clean_punctuation=self.clean_punctuation,
            strip_non_arabic=self.strip_non_arabic,
        )

    def build_vocab(self,texts:list[str]) :
       all_charts = set()
       for text in texts:
           all_charts.update(self._normalize_text(text))
        
       sorted_charts = sorted(all_charts)
       self.id_to_char = {}
       self.char_to_id = {}
       for i,token in enumerate(self.special_tokens):
           self.char_to_id[token] = i
           self.id_to_char[i] = token
       for i,char in enumerate(sorted_charts,start = len(self.special_tokens)):
           self.char_to_id[char] = i
           self.id_to_char[i] = char
       self.vocab_size = len(self.char_to_id)
    
    def encode(self,text:str,add_special_tokens:bool=True,max_length:Optional[int] = None)->list[int]:
        text = self._normalize_text(text)
        unk_id = self.char_to_id[self.unk_token]
        ids = [self.char_to_id.get(char,unk_id) for char in text]

        if add_special_tokens:
            sos_id = self.char_to_id[self.sos_token]
            eos_id = self.char_to_id[self.eos_token]
            ids = [sos_id] + ids + [eos_id]

        if max_length is not None and len(ids)>max_length:
            ids = ids[:max_length]
            if add_special_tokens:
                ids[-1] = self.char_to_id[self.eos_token]

        return ids

    def decode(self, ids: list[int], remove_special_tokens: bool = True) -> str:
        chars = []
        special_ids = {self.char_to_id[token] for token in self.special_tokens}

        for id in ids:
            if remove_special_tokens and id in special_ids:
                if id == self.eos_id:
                    break
                continue
            chars.append(self.id_to_char.get(id, self.unk_token))

        return "".join(chars)

    @property
    def pad_id(self) -> int:
        return self.char_to_id[self.pad_token]

    @property
    def sos_id(self) -> int:
        return self.char_to_id[self.sos_token]

    @property
    def eos_id(self) -> int:
        return self.char_to_id[self.eos_token]

    @property
    def unk_id(self) -> int:
        return self.char_to_id[self.unk_token]
    
    def save(self,path:str)->None:
        data = {
            "char_to_id":self.char_to_id,
            "special_tokens":self.special_tokens,
            "normalization": {
                "normalize_whitespace": self.normalize_whitespace,
                "normalize_alef": self.normalize_alef,
                "normalize_taa_marbuta": self.normalize_taa_marbuta,
                "normalize_alef_maqsura": self.normalize_alef_maqsura,
                "remove_diacritics": self.remove_diacritics,
                "clean_punctuation": self.clean_punctuation,
                "strip_non_arabic": self.strip_non_arabic,
            },
        }
        with open(path,"w",encoding="utf-8") as f:
            json.dump(data,f,ensure_ascii=False,indent=2)

    def load(self,path:str)->None:
        with open(path,"r",encoding="utf-8") as f:
            data = json.load(f)
        self.char_to_id = data["char_to_id"]
        self.id_to_char = {int(v):k for k,v in self.char_to_id.items()}
        self.special_tokens = data["special_tokens"]
        normalization = data.get("normalization", {})
        self.normalize_whitespace = normalization.get("normalize_whitespace", self.normalize_whitespace)
        self.normalize_alef = normalization.get("normalize_alef", self.normalize_alef)
        self.normalize_taa_marbuta = normalization.get("normalize_taa_marbuta", self.normalize_taa_marbuta)
        self.normalize_alef_maqsura = normalization.get("normalize_alef_maqsura", self.normalize_alef_maqsura)
        self.remove_diacritics = normalization.get("remove_diacritics", self.remove_diacritics)
        self.clean_punctuation = normalization.get("clean_punctuation", self.clean_punctuation)
        self.strip_non_arabic = normalization.get("strip_non_arabic", self.strip_non_arabic)
        self.vocab_size = len(self.char_to_id)


