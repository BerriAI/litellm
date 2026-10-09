from __future__ import annotations

import argparse
import os
import re
import secrets
import shlex
import sys
from pathlib import Path
from typing import Final

SECRET_NAMES: Final = (
    "LITELLM_MASTER_KEY",
    "LITELLM_SALT_KEY",
    "LITELLM_LENS_SERVICE_TOKEN",
    "POSTGRES_PASSWORD",
    "CLICKHOUSE_PASSWORD",
)


def environment_content(path: Path) -> str:
    if not path.exists():
        return "".join(
            f"{name}={'sk-' if name.endswith('KEY') else ''}{secrets.token_hex(32)}\n" for name in SECRET_NAMES
        )
    saved: Final = path.read_text().splitlines()
    values: Final = dict(line.split("=", 1) for line in saved if "=" in line)
    if any(not values.get(name) for name in SECRET_NAMES):
        raise ValueError(f"{path} is incomplete. Restore your saved credentials before continuing")
    return "\n".join(line for line in saved if not line.startswith("LITELLM_VERSION=")) + "\n"


def configure(path: Path, version: str) -> None:
    release: Final = version.removeprefix("v")
    if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+(?:[-.][a-zA-Z0-9.-]+)?", release):
        raise ValueError("Use a published release version, such as 1.82.0 or 1.82.0-nightly")
    if path.is_symlink():
        raise ValueError(f"Refusing to replace a symlink: {path}")
    existing: Final = path.exists()
    content: Final = environment_content(path)
    temporary: Final = path.with_name(f".{path.name}.{secrets.token_hex(8)}")
    descriptor: Final = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w") as output:
            output.write(content + f"LITELLM_VERSION={release}\n")
        if existing:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser: Final = argparse.ArgumentParser(description="Create or update the configuration for the Lens Compose stack")
    parser.add_argument("--version", required=True, help="Published LiteLLM release; Lens uses the matching version")
    parser.add_argument("--env-file", type=Path, default=Path(__file__).with_name(".env"))
    arguments: Final = parser.parse_args()
    try:
        configure(arguments.env_file, arguments.version)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Could not configure Lens: {error}\n")
    sys.stdout.write(
        f"Saved {arguments.env_file}. Existing keys and database passwords are preserved\n"
        f"Start with: docker compose --env-file {shlex.quote(str(arguments.env_file))} "
        "-f deploy/lens/stack.yaml up -d --wait\n"
        "Open http://localhost:4000/ui/ and sign in as admin with LITELLM_MASTER_KEY from the saved file\n"
        "Back up this file with your database volumes. Do not commit it\n"
    )


if __name__ == "__main__":
    main()
