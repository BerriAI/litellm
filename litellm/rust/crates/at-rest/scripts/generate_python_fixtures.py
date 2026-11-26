"""Write fixtures/python_ciphertexts.json from LiteLLM's own encrypt_decrypt_utils, outside FIPS mode.

    PYTHONPATH=<repo root> uv run --no-sync python litellm/rust/crates/at-rest/scripts/generate_python_fixtures.py

Ciphertexts carry random nonces, so every run rewrites every value.
"""

import base64
import json
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Final

SALT_KEY: Final = "sk-at-rest-fixture-salt-key"
OTHER_SALT_KEY: Final = "sk-a-different-salt-key"
OUTPUT: Final = Path(__file__).resolve().parent.parent / "fixtures" / "python_ciphertexts.json"
PLAINTEXTS: Final = {
    "api_key": "sk-proj-credential-value",
    "empty": "",
    "unicode": "clé secrète ✓ 鍵",
    "long": "https://example-resource.openai.azure.com/openai/deployments/" + "x" * 300,
}

os.environ["LITELLM_SALT_KEY"] = SALT_KEY
GENERAL_SETTINGS: Final[dict[str, str]] = {}
PROXY_SERVER: Final = ModuleType("litellm.proxy.proxy_server")
PROXY_SERVER.general_settings = GENERAL_SETTINGS
PROXY_SERVER.master_key = None
sys.modules["litellm.proxy.proxy_server"] = PROXY_SERVER

from litellm.proxy.common_utils.encrypt_decrypt_utils import (  # noqa: E402  # the real proxy_server needs the Rust extension
    decrypt_value_helper,
    encrypt_value,
    encrypt_value_helper,
)


def encrypt(value: str, algorithm: str, salt_key: str = SALT_KEY) -> str:
    GENERAL_SETTINGS["encryption_algorithm"] = algorithm
    return encrypt_value_helper(value, new_encryption_key=salt_key)


def legacy_standard_alphabet(value: str) -> str:
    encoded: Final = base64.b64encode(encrypt_value(value=value, signing_key=SALT_KEY)).decode()
    return encoded if ("+" in encoded or "/" in encoded) else legacy_standard_alphabet(value)


def tampered(stored: str) -> str:
    prefix: Final = "v2:gcm:" if stored.startswith("v2:gcm:") else ""
    sealed: Final = bytearray(base64.urlsafe_b64decode(stored.removeprefix(prefix)))
    sealed[-1] ^= 1
    return prefix + base64.urlsafe_b64encode(bytes(sealed)).decode()


def case(name: str, stored: str) -> dict[str, str | None]:
    return {
        "name": name,
        "stored": stored,
        "decrypted": decrypt_value_helper(value=stored, key=name, exception_type="debug"),
        "or_original": decrypt_value_helper(value=stored, key=name, exception_type="debug", return_original_value=True),
    }


def main() -> None:
    ciphertexts: Final = (
        *(case(f"legacy_{name}", encrypt(value, "xsalsa20-poly1305")) for name, value in PLAINTEXTS.items()),
        *(case(f"gcm_{name}", encrypt(value, "aes-256-gcm")) for name, value in PLAINTEXTS.items()),
        case("legacy_standard_alphabet", legacy_standard_alphabet(PLAINTEXTS["api_key"])),
        case("legacy_wrong_key", encrypt(PLAINTEXTS["api_key"], "xsalsa20-poly1305", OTHER_SALT_KEY)),
        case("gcm_wrong_key", encrypt(PLAINTEXTS["api_key"], "aes-256-gcm", OTHER_SALT_KEY)),
        case("legacy_tampered", tampered(encrypt(PLAINTEXTS["api_key"], "xsalsa20-poly1305"))),
        case("gcm_tampered", tampered(encrypt(PLAINTEXTS["api_key"], "aes-256-gcm"))),
        case("gcm_truncated", "v2:gcm:" + base64.urlsafe_b64encode(b"\x00" * 8).decode()),
        case("gcm_not_base64", "v2:gcm:not base64!"),
    )
    plain: Final = (
        case("plain_model_name", "azure/gpt-4o"),
        case("plain_url", "https://api.openai.com/v1"),
        case("plain_env_reference", "os.environ/OPENAI_API_KEY"),
        case("plain_base64_alphabet", "abcd"),
        case("plain_unicode", "héllo"),
        case("plain_empty", ""),
    )
    OUTPUT.write_text(
        json.dumps({"salt_key": SALT_KEY, "cases": [*ciphertexts, *plain]}, indent=2, ensure_ascii=False) + "\n"
    )


if __name__ == "__main__":
    main()
