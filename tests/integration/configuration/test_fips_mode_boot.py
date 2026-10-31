import _hashlib
import json
from pathlib import Path
from typing import Final

import pytest

from tests.integration._support.client import Gateway
from tests.integration._support.process import refused_boot_log


def _config(directory: Path, upstream_url: str) -> Path:
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


def test_fips_mode_refuses_to_start_on_a_value_that_is_not_a_boolean(gateway: Gateway, tmp_path: Path) -> None:
    log: Final = refused_boot_log(
        gateway, tmp_path, {"LITELLM_FIPS_MODE": "enforced"}, config=_config(tmp_path, gateway.upstream_url)
    )
    assert "LITELLM_FIPS_MODE='enforced' is not a boolean" in log, log
