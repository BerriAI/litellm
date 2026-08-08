import sys
from pathlib import Path

from pydantic import TypeAdapter

secrets_adapter: TypeAdapter[dict[str, str]] = TypeAdapter(dict[str, str])


def main() -> int:
    env_path = Path(sys.argv[1])
    secrets = secrets_adapter.validate_json(sys.stdin.read())
    unwritable = tuple(
        key for key, value in secrets.items() if "'" in value or "\n" in value or "\r" in value
    )
    if unwritable:
        _ = sys.stderr.write(f"values contain characters unsafe for both bash and dotenv: {', '.join(unwritable)}\n")
        return 1
    lines = tuple(f"{key}='{value}'" for key, value in secrets.items() if value)
    with env_path.open("a") as handle:
        _ = handle.write("\n".join(lines) + "\n")
    for value in secrets.values():
        if value:
            _ = sys.stdout.write(f"::add-mask::{value}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
