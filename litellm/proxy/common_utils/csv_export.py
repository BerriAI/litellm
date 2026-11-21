import csv
import io
from collections.abc import Sequence
from typing import Final


def csv_cell(value: object) -> object:
    if isinstance(value, str) and value.startswith(("=", "+", "-", "@", "\t", "\r")):
        return f"'{value}"
    return value


def csv_row(values: Sequence[object]) -> bytes:
    output: Final = io.StringIO(newline="")
    csv.writer(output, lineterminator="\r\n").writerow(tuple(csv_cell(value) for value in values))
    return output.getvalue().encode("utf-8")
