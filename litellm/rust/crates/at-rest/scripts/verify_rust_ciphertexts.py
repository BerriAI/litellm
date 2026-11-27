"""Check that LiteLLM's Python decrypt reads every value in fixtures/rust_ciphertexts.json back to its plaintext, and
every value in fixtures/rust_fips_ciphertexts.json with general_settings.fips_mode on, run on OpenSSL's default
provider as python_fips_mode.py describes.

PYTHONPATH=<repo root> uv run --no-sync python litellm/rust/crates/at-rest/scripts/verify_rust_ciphertexts.py
"""

import json
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Final

FIXTURES: Final = Path(__file__).resolve().parent.parent / "fixtures"
STANDARD: Final = json.loads((FIXTURES / "rust_ciphertexts.json").read_text(encoding="utf-8"))
FIPS: Final = json.loads((FIXTURES / "rust_fips_ciphertexts.json").read_text(encoding="utf-8"))

PROXY_SERVER: Final = ModuleType("litellm.proxy.proxy_server")
PROXY_SERVER.general_settings = {}
PROXY_SERVER.master_key = None
sys.modules["litellm.proxy.proxy_server"] = PROXY_SERVER

from python_fips_mode import enter_fips_mode  # noqa: E402  # imports litellm after the proxy_server stub

from litellm.proxy.common_utils.encrypt_decrypt_utils import (  # noqa: E402  # the real proxy_server needs the Rust extension
    decrypt_value_helper,
)


def mismatches(fixture: dict[str, object], label: str) -> tuple[str, ...]:
    os.environ["LITELLM_SALT_KEY"] = fixture["salt_key"]
    cases: Final = fixture["cases"]
    failed: Final = tuple(
        case["name"]
        for case in cases
        if decrypt_value_helper(value=case["stored"], key=case["name"]) != case["plaintext"]
    )
    sys.stdout.write(f"{len(cases) - len(failed)} of {len(cases)} Rust {label}ciphertexts decrypt in Python\n")
    sys.stdout.write("".join(f"mismatch: {name}\n" for name in failed))
    return failed


def main() -> int:
    standard: Final = mismatches(STANDARD, "")
    enter_fips_mode()
    fips: Final = mismatches(FIPS, "fips_mode ")
    return 1 if standard or fips else 0


if __name__ == "__main__":
    sys.exit(main())
