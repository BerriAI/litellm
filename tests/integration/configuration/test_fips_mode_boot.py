import _hashlib
import json
from pathlib import Path
from typing import Final

import pytest

from tests.integration._support.client import Gateway
from tests.integration._support.process import owned_proxy, refused_boot_log


def _config(directory: Path, upstream_url: str, *, ssl_verify: bool = True) -> Path:
    config: Final = directory / "fips_boot.yaml"
    config.write_text(
        json.dumps(
            {
                "model_list": [
                    {
                        "model_name": "fips-model",
                        "litellm_params": {"model": "openai/fips-model", "api_base": upstream_url, "api_key": "sk-x"},
                    }
                ],
                "litellm_settings": {"ssl_verify": ssl_verify},
            }
        )
    )
    return config


def test_fips_mode_refuses_to_start_when_openssl_does_not_enforce_fips(gateway: Gateway, tmp_path: Path) -> None:
    if _hashlib.get_fips_mode() == 1:
        pytest.skip("this runner's OpenSSL is in FIPS mode")
    log: Final = refused_boot_log(
        gateway, tmp_path, {"LITELLM_FIPS_MODE": "true"}, config=_config(tmp_path, gateway.upstream_url)
    )
    assert "LITELLM_FIPS_MODE is on but OpenSSL does not enforce FIPS" in log, log


@pytest.mark.parametrize(
    "overrides, ssl_verify",
    [({"SSL_VERIFY": "false"}, True), ({}, False)],
    ids=["ssl-verify-env", "litellm-settings"],
)
def test_fips_mode_refuses_to_start_with_tls_verification_disabled(
    gateway: Gateway, tmp_path: Path, overrides: dict[str, str], ssl_verify: bool
) -> None:
    config: Final = _config(tmp_path, gateway.upstream_url, ssl_verify=ssl_verify)
    log: Final = refused_boot_log(gateway, tmp_path, {"LITELLM_FIPS_MODE": "true", **overrides}, config=config)
    assert "LITELLM_FIPS_MODE is on but TLS certificate verification is disabled" in log, log


def test_fips_mode_refuses_to_start_on_a_value_that_is_not_a_boolean(gateway: Gateway, tmp_path: Path) -> None:
    log: Final = refused_boot_log(
        gateway, tmp_path, {"LITELLM_FIPS_MODE": "enforced"}, config=_config(tmp_path, gateway.upstream_url)
    )
    assert "LITELLM_FIPS_MODE='enforced' is not a boolean" in log, log


def test_fips_mode_off_starts_with_tls_verification_disabled(gateway: Gateway, tmp_path: Path) -> None:
    config: Final = _config(tmp_path, gateway.upstream_url, ssl_verify=False)
    with owned_proxy(
        gateway, tmp_path, {"SSL_VERIFY": "false"}, config=config, remove_environment=("LITELLM_FIPS_MODE",)
    ) as proxy:
        assert proxy.client.get("/health/readiness").status_code == 200
