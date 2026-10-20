import re
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Final

_SEPARATORS: Final = re.compile(r"[\s\-]")
_IBAN_CHARACTERS: Final = re.compile(r"[0-9A-Z]{5,34}")


def _luhn_term(position: int, digit: int) -> int:
    if position % 2 == 0:
        return digit
    doubled: Final = digit * 2
    return doubled - 9 if doubled > 9 else doubled


def passes_luhn(value: str) -> bool:
    digits: Final = _SEPARATORS.sub("", value)
    if not digits.isdecimal():
        return False
    return sum(_luhn_term(position, int(digit)) for position, digit in enumerate(reversed(digits))) % 10 == 0


def passes_iban_mod97(value: str) -> bool:
    iban: Final = _SEPARATORS.sub("", value).upper()
    if _IBAN_CHARACTERS.fullmatch(iban) is None:
        return False
    rearranged: Final = iban[4:] + iban[:4]
    return int("".join(str(int(character, 36)) for character in rearranged)) % 97 == 1


PATTERN_VALIDATORS: Final[Mapping[str, Callable[[str], bool]]] = MappingProxyType(
    {"luhn": passes_luhn, "iban_mod97": passes_iban_mod97}
)
