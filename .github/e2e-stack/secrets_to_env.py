import os
import re
import sys
from pathlib import Path
from typing import Final

from pydantic import TypeAdapter, ValidationError

secrets_adapter: Final[TypeAdapter[dict[str, str]]] = TypeAdapter(dict[str, str])
ENV_NAME: Final = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
MIN_MASKED_LENGTH: Final = 8


def main() -> int:
    env_path: Final = Path(sys.argv[1])
    try:
        secrets: Final = {
            key: value.rstrip("\r\n") for key, value in secrets_adapter.validate_json(sys.stdin.read()).items()
        }
    except (ValidationError, UnicodeError):
        _ = sys.stderr.write("expected a JSON object containing string environment values\n")
        return 1
    unusable: Final = tuple(
        key
        for key, value in secrets.items()
        if ENV_NAME.fullmatch(key) is None or any(char in value for char in "'\n\r\0")
    )
    if unusable:
        _ = sys.stderr.write(
            f"these names or values cannot be represented in both bash and dotenv: {' '.join(sorted(unusable))}\n"
        )
        return 1
    for value in secrets.values():
        if len(value) >= MIN_MASKED_LENGTH:
            _ = sys.stdout.write(f"::add-mask::{value.replace('%', '%25')}\n")
    sys.stdout.flush()
    lines: Final = tuple(f"{key}='{value}'" for key, value in secrets.items() if value)
    try:
        with os.fdopen(os.open(env_path, os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW, 0o600), "w") as handle:
            os.fchmod(handle.fileno(), 0o600)
            _ = handle.write("\n".join(lines) + "\n")
    except OSError:
        _ = sys.stderr.write("could not write the environment file\n")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
