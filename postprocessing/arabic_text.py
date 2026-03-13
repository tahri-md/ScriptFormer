import re

ARABIC_DIACRITICS = (
    "\u0610\u0611\u0612\u0613\u0614\u0615\u0616\u0617\u0618\u0619\u061A"
    "\u064B\u064C\u064D\u064E\u064F\u0650\u0651\u0652\u0653\u0654\u0655"
    "\u0656\u0657\u0658\u065F\u0670"
)

DIACRITICS_PATTERN = re.compile(f"[{''.join(ARABIC_DIACRITICS)}]")

ALEF_VARIANTS = {
    "\u0622": "\u0627",
    "\u0623": "\u0627",
    "\u0625": "\u0627",
    "\u0671": "\u0627",
}

OPTIONAL_NORMALIZATIONS = {
    "\u0629": "\u0647",
    "\u0649": "\u064A",
}


class ArabicPostProcessor:

    def __init__(
        self,
        remove_special_tokens: bool = True,
        fix_repetitions: bool = True,
        max_char_repeat: int = 2,
        normalize_whitespace: bool = True,
        normalize_alef: bool = True,
        normalize_taa_marbuta: bool = False,
        normalize_alef_maqsura: bool = False,
        remove_diacritics: bool = False,
        clean_punctuation: bool = True,
        strip_non_arabic: bool = False,
    ):
        self.remove_special_tokens = remove_special_tokens
        self.fix_repetitions = fix_repetitions
        self.max_char_repeat = max_char_repeat
        self.normalize_whitespace = normalize_whitespace
        self.normalize_alef = normalize_alef
        self.normalize_taa_marbuta = normalize_taa_marbuta
        self.normalize_alef_maqsura = normalize_alef_maqsura
        self.remove_diacritics = remove_diacritics
        self.clean_punctuation = clean_punctuation
        self.strip_non_arabic = strip_non_arabic

        self._special_tokens = {"<PAD>", "<SOS>", "<EOS>", "<UNK>"}

    def __call__(self, text: str) -> str:
        if not text:
            return ""

        if self.remove_special_tokens:
            text = self._remove_special_tokens(text)

        if self.remove_diacritics:
            text = self._remove_diacritics(text)

        if self.fix_repetitions:
            text = self._fix_repetitions(text, self.max_char_repeat)

        if self.normalize_alef:
            text = self._normalize_alef(text)
        if self.normalize_taa_marbuta:
            text = self._normalize_taa_marbuta(text)
        if self.normalize_alef_maqsura:
            text = self._normalize_alef_maqsura(text)

        if self.clean_punctuation:
            text = self._clean_punctuation(text)

        if self.normalize_whitespace:
            text = self._normalize_whitespace(text)

        if self.strip_non_arabic:
            text = self._strip_non_arabic(text)

        return text

    def _remove_special_tokens(self, text: str) -> str:
        for token in self._special_tokens:
            text = text.replace(token, "")
        return text

    def _remove_diacritics(self, text: str) -> str:
        return DIACRITICS_PATTERN.sub("", text)

    def _fix_repetitions(self, text: str, max_repeat: int = 2) -> str:
        if max_repeat < 1:
            max_repeat = 1
        pattern = re.compile(r"(.)\1{" + str(max_repeat) + r",}")
        return pattern.sub(lambda m: m.group(1) * max_repeat, text)

    def _normalize_alef(self, text: str) -> str:
        for variant, normalized in ALEF_VARIANTS.items():
            text = text.replace(variant, normalized)
        return text

    def _normalize_taa_marbuta(self, text: str) -> str:
        return text.replace("\u0629", "\u0647")

    def _normalize_alef_maqsura(self, text: str) -> str:
        return text.replace("\u0649", "\u064A")

    def _clean_punctuation(self, text: str) -> str:
        punct = r"[\.،؛؟!:,;?]"
        text = re.sub(r"\s+(" + punct + r")", r"\1", text)
        text = re.sub(r"(" + punct + r")(\S)", r"\1 \2", text)
        return text

    def _normalize_whitespace(self, text: str) -> str:
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _strip_non_arabic(self, text: str) -> str:
        return re.sub(r"[^\u0600-\u06FF\s]", "", text)

    def describe(self) -> str:
        steps = []
        if self.remove_special_tokens:
            steps.append("remove special tokens")
        if self.remove_diacritics:
            steps.append("remove diacritics")
        if self.fix_repetitions:
            steps.append(f"fix repetitions (max={self.max_char_repeat})")
        if self.normalize_alef:
            steps.append("normalize alef variants")
        if self.normalize_taa_marbuta:
            steps.append("normalize taa marbuta")
        if self.normalize_alef_maqsura:
            steps.append("normalize alef maqsura")
        if self.clean_punctuation:
            steps.append("clean punctuation spacing")
        if self.normalize_whitespace:
            steps.append("normalize whitespace")
        if self.strip_non_arabic:
            steps.append("strip non-Arabic characters")
        return " -> ".join(steps) if steps else "none"