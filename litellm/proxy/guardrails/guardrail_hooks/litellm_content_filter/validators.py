from typing import Final


def validate_luhn(candidate: str) -> bool:
    clean_digits: Final = tuple(int(char) for char in candidate if char.isdigit())
    if len(clean_digits) < 12 or len(clean_digits) > 19:
        return False
    total: Final = sum(
        digit if index % 2 == 0 else (digit * 2 - 9 if digit * 2 > 9 else digit * 2)
        for index, digit in enumerate(reversed(clean_digits))
    )
    return total % 10 == 0


def validate_iban(candidate: str) -> bool:
    cleaned: Final = "".join(char for char in candidate if char.isalnum()).upper()
    if len(cleaned) < 15 or len(cleaned) > 34:
        return False
    if not (cleaned[:2].isalpha() and cleaned[2:4].isdigit()):
        return False
    rotated: Final = cleaned[4:] + cleaned[:4]
    numeric_string: Final = "".join(str(int(char, 36)) for char in rotated)
    return int(numeric_string) % 97 == 1
