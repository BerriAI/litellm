"""Run stdlib regex in isolation so a caller can terminate pathological matches."""

import json
import re
import sys
from contextlib import redirect_stdout
from typing import Final, TextIO


def _execute(operation: str, text: str, pattern: str, flags: int, replacement: str) -> object:
    match operation:
        case "search":
            return bool(re.search(pattern, text, flags))
        case "fullmatch":
            return bool(re.fullmatch(pattern, text, flags))
        case "sub":
            return re.sub(pattern, replacement, text, flags=flags)
        case "findall":
            return re.findall(pattern, text, flags)
        case _:
            raise ValueError("unknown regex operation")


def _main(request: str, output: TextIO, errors: TextIO) -> int:
    payload: Final[object] = json.loads(request)  # pyright: ignore[reportAny]  # stdlib JSON is validated below
    match payload:
        case [str(operation), str(text), str(pattern), int(flags), str(replacement)]:
            pass
        case _:
            raise ValueError("invalid regex request")
    try:
        with redirect_stdout(errors):
            result: Final = _execute(operation, text, pattern, flags, replacement)
    except re.error as error:
        errors.write(str(error))
        return 65
    output.write(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.stdin.read(), sys.stdout, sys.stderr))
