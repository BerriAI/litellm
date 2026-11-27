"""Write fixtures/python_fips_ciphertexts.json from LiteLLM's own encrypt_decrypt_utils with general_settings.fips_mode
on, run on OpenSSL's default provider as python_fips_mode.py describes.

    PYTHONPATH=<repo root> uv run --no-sync python litellm/rust/crates/at-rest/scripts/generate_python_fips_fixtures.py

Ciphertexts carry random nonces, so every run rewrites every value.
"""

import base64
import json
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Final

SALT_KEY: Final = "sk-at-rest-fips-fixture-salt-key"
OTHER_SALT_KEY: Final = "sk-a-different-fips-salt-key"
OUTPUT: Final = Path(__file__).resolve().parent.parent / "fixtures" / "python_fips_ciphertexts.json"
PLAINTEXTS: Final = {
    "api_key": "sk-proj-credential-value",
    "empty": "",
    "unicode": "clé secrète ✓ 鍵",
    "long": "https://example-resource.openai.azure.com/openai/deployments/" + "x" * 300,
}
KBKDF_INPUTS: Final = (
    ("at_rest_key", SALT_KEY.encode(), b"litellm-at-rest:aes-256-gcm:", b""),
    ("with_context", b"sk-fourteen-by", b"a-label", b"a-context"),
    ("key_longer_than_a_hmac_block", b"k" * 100, b"litellm-at-rest:aes-256-gcm:", b"\x00\x01\x02"),
)

os.environ["LITELLM_SALT_KEY"] = SALT_KEY
GENERAL_SETTINGS: Final[dict[str, str]] = {}
PROXY_SERVER: Final = ModuleType("litellm.proxy.proxy_server")
PROXY_SERVER.general_settings = GENERAL_SETTINGS
PROXY_SERVER.master_key = None
sys.modules["litellm.proxy.proxy_server"] = PROXY_SERVER

from python_fips_mode import enter_fips_mode, kbkdf  # noqa: E402  # imports litellm after the proxy_server stub

from litellm.proxy.common_utils.encrypt_decrypt_utils import (  # noqa: E402  # the real proxy_server needs the Rust extension
    decrypt_value_helper,
    encrypt_value_helper,
)


def encrypt(value: str, salt_key: str = SALT_KEY) -> str:
    return encrypt_value_helper(value, new_encryption_key=salt_key)


def tampered(stored: str) -> str:
    sealed: Final = bytearray(base64.urlsafe_b64decode(stored.removeprefix("v2:gcm:")))
    sealed[-1] ^= 1
    return "v2:gcm:" + base64.urlsafe_b64encode(bytes(sealed)).decode()


def case(name: str, stored: str) -> dict[str, str | None]:
    return {
        "name": name,
        "stored": stored,
        "decrypted": decrypt_value_helper(value=stored, key=name, exception_type="debug"),
        "or_original": decrypt_value_helper(value=stored, key=name, exception_type="debug", return_original_value=True),
    }


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


def main() -> None:
    GENERAL_SETTINGS["encryption_algorithm"] = "aes-256-gcm"
    standard_gcm: Final = encrypt(PLAINTEXTS["api_key"])
    GENERAL_SETTINGS["encryption_algorithm"] = "xsalsa20-poly1305"
    legacy: Final = encrypt(PLAINTEXTS["api_key"])
    enter_fips_mode()
    vectors: Final = tuple(
        {
            "name": name,
            "key": b64(key),
            "label": b64(label),
            "context": b64(context),
            "derived": b64(kbkdf(key, label, context)),
        }
        for name, key, label, context in KBKDF_INPUTS
    )
    cases: Final = (
        *(case(f"fips_{name}", encrypt(value)) for name, value in PLAINTEXTS.items()),
        case("fips_wrong_key", encrypt(PLAINTEXTS["api_key"], OTHER_SALT_KEY)),
        case("fips_tampered", tampered(encrypt(PLAINTEXTS["api_key"]))),
        case("standard_gcm", standard_gcm),
        case("legacy_api_key", legacy),
        case("plain_model_name", "azure/gpt-4o"),
        case("plain_base64_alphabet", "abcd"),
        case("plain_empty", ""),
    )
    OUTPUT.write_text(
        json.dumps(
            {
                "produced_by": (
                    "scripts/generate_python_fips_fixtures.py, LiteLLM's Python with fips_mode on, its KBKDF and "
                    "AES-256-GCM run by OpenSSL's default provider instead of the FIPS provider, and every KBKDF "
                    "output checked against cryptography's KBKDFHMAC"
                ),
                "salt_key": SALT_KEY,
                "kbkdf": vectors,
                "cases": cases,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
