import re
import string
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Final

_SEPARATORS: Final = re.compile(r"[\s\-]")
_IBAN_CHARACTERS: Final = re.compile(r"[0-9A-Z]{5,34}")
_IGNORECASE_ASCII_LETTER_ALIASES: Final = str.maketrans({"\u0130": "I", "\u212a": "K"})
_ISO_13616_IBAN_LENGTHS: Final[Mapping[str, int]] = MappingProxyType(
    {
        "AD": 24,
        "AE": 23,
        "AL": 28,
        "AT": 20,
        "AX": 18,
        "AZ": 28,
        "BA": 20,
        "BE": 16,
        "BG": 22,
        "BH": 22,
        "BI": 27,
        "BL": 27,
        "BR": 29,
        "BY": 28,
        "CH": 21,
        "CR": 22,
        "CY": 28,
        "CZ": 24,
        "DE": 22,
        "DJ": 27,
        "DK": 18,
        "DO": 28,
        "EE": 20,
        "EG": 29,
        "ES": 24,
        "FI": 18,
        "FK": 18,
        "FO": 18,
        "FR": 27,
        "GB": 22,
        "GE": 22,
        "GF": 27,
        "GG": 22,
        "GI": 23,
        "GL": 18,
        "GP": 27,
        "GR": 27,
        "GT": 28,
        "HN": 28,
        "HR": 21,
        "HU": 28,
        "IE": 22,
        "IL": 23,
        "IM": 22,
        "IQ": 23,
        "IS": 26,
        "IT": 27,
        "JE": 22,
        "JO": 30,
        "KW": 30,
        "KZ": 20,
        "LB": 28,
        "LC": 32,
        "LI": 21,
        "LT": 20,
        "LU": 20,
        "LV": 21,
        "LY": 25,
        "MC": 27,
        "MD": 24,
        "ME": 22,
        "MF": 27,
        "MK": 19,
        "MN": 20,
        "MQ": 27,
        "MR": 27,
        "MT": 31,
        "MU": 30,
        "NC": 27,
        "NI": 28,
        "NL": 18,
        "NO": 15,
        "OM": 23,
        "PF": 27,
        "PK": 24,
        "PL": 28,
        "PM": 27,
        "PS": 29,
        "PT": 25,
        "QA": 29,
        "RE": 27,
        "RO": 24,
        "RS": 22,
        "RU": 33,
        "SA": 24,
        "SC": 31,
        "SD": 18,
        "SE": 24,
        "SI": 19,
        "SK": 24,
        "SM": 27,
        "SO": 23,
        "ST": 25,
        "SV": 28,
        "TF": 27,
        "TL": 23,
        "TN": 24,
        "TR": 26,
        "UA": 29,
        "VA": 22,
        "VG": 24,
        "WF": 27,
        "XK": 20,
        "YE": 30,
        "YT": 27,
    }
)
_IBAN_LETTER_VALUES: Final = str.maketrans(
    {letter: str(value) for value, letter in enumerate(string.ascii_uppercase, 10)}
)
_DIGIT_VALUES: Final = bytes.maketrans(b"0123456789", bytes(range(10)))
_DOUBLED_DIGIT_SUMS: Final = bytes.maketrans(b"0123456789", bytes((0, 2, 4, 6, 8, 1, 3, 5, 7, 9)))


def _to_ascii(value: str) -> str:
    if value.isascii():
        return value
    folded: Final = value.translate(_IGNORECASE_ASCII_LETTER_ALIASES)
    return "".join(str(int(character)) if character.isdecimal() else character for character in folded)


def passes_luhn(value: str) -> bool:
    digits: Final = _SEPARATORS.sub("", value)
    if not digits.isdecimal():
        return False
    reversed_digits: Final = _to_ascii(digits)[::-1].encode()
    checksum: Final = sum(reversed_digits[::2].translate(_DIGIT_VALUES)) + sum(
        reversed_digits[1::2].translate(_DOUBLED_DIGIT_SUMS)
    )
    return checksum % 10 == 0


def passes_iban_mod97(value: str) -> bool:
    compact: Final = _to_ascii(_SEPARATORS.sub("", value)).upper()
    iban: Final = compact[: _ISO_13616_IBAN_LENGTHS.get(compact[:2], len(compact))]
    if _IBAN_CHARACTERS.fullmatch(iban) is None:
        return False
    rearranged: Final = iban[4:] + iban[:4]
    return int(rearranged.translate(_IBAN_LETTER_VALUES)) % 97 == 1


PATTERN_VALIDATORS: Final[Mapping[str, Callable[[str], bool]]] = MappingProxyType(
    {"luhn": passes_luhn, "iban_mod97": passes_iban_mod97}
)
