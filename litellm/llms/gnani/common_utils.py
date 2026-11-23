import re
from typing import Final

_REGION_CODE: Final = re.compile(r"([A-Za-z]{2})-([A-Za-z]{2})")


def normalize_language(hint: str) -> str:
    if hint.lower() == "indic-auto":
        return "indic-auto"
    if len(hint) == 2:
        return f"{hint.lower()}-IN"
    region: Final = _REGION_CODE.fullmatch(hint)
    return f"{region[1].lower()}-{region[2].upper()}" if region else hint
