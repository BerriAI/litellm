"""Check that LiteLLM's Python decrypt reads every value in fixtures/rust_ciphertexts.json back to its plaintext.

PYTHONPATH=<repo root> uv run --no-sync python litellm/rust/crates/at-rest/scripts/verify_rust_ciphertexts.py
"""

import json
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Final

FIXTURE: Final = json.loads(
    (Path(__file__).resolve().parent.parent / "fixtures" / "rust_ciphertexts.json").read_text(encoding="utf-8")
)

os.environ["LITELLM_SALT_KEY"] = FIXTURE["salt_key"]
PROXY_SERVER: Final = ModuleType("litellm.proxy.proxy_server")
PROXY_SERVER.general_settings = {}
PROXY_SERVER.master_key = None
sys.modules["litellm.proxy.proxy_server"] = PROXY_SERVER

from litellm.proxy.common_utils.encrypt_decrypt_utils import (  # noqa: E402  # the real proxy_server needs the Rust extension
    decrypt_value_helper,
)


def main() -> int:
    mismatches: Final = tuple(
        case["name"]
        for case in FIXTURE["cases"]
        if decrypt_value_helper(value=case["stored"], key=case["name"]) != case["plaintext"]
    )
    sys.stdout.write(
        f"{len(FIXTURE['cases']) - len(mismatches)} of {len(FIXTURE['cases'])} Rust ciphertexts decrypt in Python\n"
    )
    sys.stdout.write("".join(f"mismatch: {name}\n" for name in mismatches))
    return 1 if mismatches else 0


if __name__ == "__main__":
    sys.exit(main())
