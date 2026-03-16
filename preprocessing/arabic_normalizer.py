"""
Gulf Arabic text normalization for UAE and broader Gulf social media.

Handles unicode variants, tatweel, diacritics, and dialect spelling.
"""

import re
import unicodedata
from typing import Dict


class GulfArabicNormalizer:
    """
    Gulf Arabic-specific text normalization.
    Handles script variants, dialect forms, and informal writing patterns
    common in UAE and broader Gulf social media.
    """

    # Unicode ranges for Arabic (including extended)
    ARABIC_DIACRITICS = re.compile(r"[\u064B-\u0652\u0670]")
    TATWEEL = "\u0640"

    # Hamza alef variants -> plain alef
    HAMZA_ALEF_MAP = {
        "\u0623": "\u0627",  # أ
        "\u0625": "\u0627",  # إ
        "\u0622": "\u0627",  # آ
        "\u0671": "\u0627",  # ٱ
    }

    def normalize_unicode(self, text: str) -> str:
        """
        Normalize Arabic unicode variants to canonical forms.
        - All hamza alef variants (أ إ آ ٱ) -> plain alef (ا)
        - Teh marbuta (ة) -> heh (ه)
        - Alef maqsura (ى) -> yeh (ي)
        - Remove tatweel elongation (ـ)
        - Remove diacritics/tashkeel (U+064B to U+0652)
        Does NOT modify English characters or punctuation.
        """
        if not text:
            return text
        result = []
        for char in text:
            if char in self.HAMZA_ALEF_MAP:
                result.append(self.HAMZA_ALEF_MAP[char])
            elif char == "\u0629":  # ة teh marbuta
                result.append("\u0647")  # ه heh
            elif char == "\u0649":  # ى alef maqsura
                result.append("\u064A")  # ي yeh
            elif char == self.TATWEEL:  # ـ tatweel
                continue
            elif self.ARABIC_DIACRITICS.match(char):
                continue
            else:
                result.append(char)
        return "".join(result)

    def normalize_gulf_dialect(self, text: str) -> str:
        """
        Normalize Gulf Arabic dialect spelling variations.
        Applies a dictionary of 30+ Gulf-specific normalizations:
        - Elongated laughter: هههههه/هههه/ههه -> هه
        - وايد/وايت -> وايد
        - الحين variants -> الحين
        - يبغي/يبي -> يبغي
        - زين/زيين -> زين
        - Common Gulf abbreviations
        """
        if not text:
            return text
        # Order matters: longer patterns first where applicable
        gulf_map: Dict[str, str] = {
            # Elongated laughter
            "هههههههه": "هه",
            "ههههههه": "هه",
            "هههههه": "هه",
            "ههههه": "هه",
            "هههه": "هه",
            "ههه": "هه",
            "هاهاها": "هه",
            "هاها": "هه",
            # وايد
            "وايت": "وايد",
            "وَايد": "وايد",
            "وَايت": "وايد",
            # الحين
            "الحين": "الحين",
            "الحِين": "الحين",
            "الحينْ": "الحين",
            # يبغي/يبي
            "يبي": "يبغي",
            "يِبِي": "يبغي",
            "يبغى": "يبغي",
            # زين
            "زيين": "زين",
            "زِين": "زين",
            "زينْ": "زين",
            # Common Gulf
            "ماعندي": "ما عندي",
            "ماعندك": "ما عندك",
            "ماعندهم": "ما عندهم",
            "وش": "ايش",
            "ايش": "ايش",
            "ليش": "ليش",
            "كيفك": "كيفك",
            "شلونك": "شلونك",
            "انشالله": "ان شاء الله",
            "ماشاءالله": "ما شاء الله",
            "الله يعطيك العافيه": "الله يعطيك العافية",
            "العافيه": "العافية",
            "ترا": "ترى",
            "لأن": "لان",
            "لان": "لان",
            "عشان": "عشان",
            "عسب": "عشان",
            "عسبان": "عشان",
            "تبي": "تبغي",
            "نبي": "نبغي",
            "شكراً": "شكرا",
            "الله يسلمك": "الله يسلمك",
            "يسلم": "يسلم",
        }
        t = text
        for key, value in gulf_map.items():
            t = t.replace(key, value)
        # Collapse repeated هه to at most two
        t = re.sub(r"هه(ه)+", "هه", t)
        return t

    def full_pipeline(self, text: str) -> str:
        """
        Apply full normalization pipeline in order:
        normalize_unicode -> normalize_gulf_dialect
        Returns cleaned string.
        """
        if not text:
            return text
        t = self.normalize_unicode(text)
        t = self.normalize_gulf_dialect(t)
        return t
