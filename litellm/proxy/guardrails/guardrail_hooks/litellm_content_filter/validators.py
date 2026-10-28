import re
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Final

_SEPARATORS: Final = re.compile(r"[\s\-]")
_IBAN_CHARACTERS: Final = re.compile(r"[0-9A-Z]{5,34}")
_IGNORECASE_ASCII_LETTER_ALIASES: Final = str.maketrans({"\u0130": "I", "\u212a": "K"})
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
    iban: Final = _to_ascii(_SEPARATORS.sub("", value)).upper()
    if _IBAN_CHARACTERS.fullmatch(iban) is None:
        return False
    rearranged: Final = iban[4:] + iban[:4]
    return int("".join(str(int(character, 36)) for character in rearranged)) % 97 == 1


PATTERN_VALIDATORS: Final[Mapping[str, Callable[[str], bool]]] = MappingProxyType(
    {"luhn": passes_luhn, "iban_mod97": passes_iban_mod97}
)
