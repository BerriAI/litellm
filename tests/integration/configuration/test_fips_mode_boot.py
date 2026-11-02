import _hashlib
import json
from pathlib import Path
from typing import Final

import pytest

from tests.integration._support.client import Gateway
from tests.integration._support.database import scratch_database, write_rows
from tests.integration._support.process import refused_boot_log


def _config(directory: Path, upstream_url: str, fips_mode: object) -> Path:
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
                "general_settings": {"fips_mode": fips_mode},
            }
        )
    )
    return config


def test_fips_mode_refuses_to_start_when_openssl_does_not_enforce_fips(gateway: Gateway, tmp_path: Path) -> None:
    if _hashlib.get_fips_mode() == 1:
        pytest.skip("this runner's OpenSSL is in FIPS mode")
    log: Final = refused_boot_log(gateway, tmp_path, {}, config=_config(tmp_path, gateway.upstream_url, True))
    assert "general_settings.fips_mode is on but OpenSSL does not enforce FIPS" in log, log


def test_fips_mode_refuses_to_start_on_a_value_that_is_not_a_boolean(gateway: Gateway, tmp_path: Path) -> None:
    log: Final = refused_boot_log(gateway, tmp_path, {}, config=_config(tmp_path, gateway.upstream_url, "enforced"))
    assert "general_settings.fips_mode='enforced' is not a boolean" in log, log


def test_proxy_refuses_to_start_without_fips_mode_on_a_database_set_up_in_fips_mode(
    gateway: Gateway, tmp_path: Path
) -> None:
    with scratch_database() as database_url:
        write_rows(
            'CREATE TABLE "LiteLLM_Config" (param_name TEXT NOT NULL PRIMARY KEY, param_value JSONB, '
            "last_run_at TIMESTAMP(3), reload_revision BIGINT NOT NULL DEFAULT 0)",
            (),
            database_url=database_url,
        )
        write_rows(
            "INSERT INTO \"LiteLLM_Config\" (param_name, param_value) VALUES ('fips_mode', 'true')",
            (),
            database_url=database_url,
        )
        log: Final = refused_boot_log(
            gateway,
            tmp_path,
            {"DATABASE_URL": database_url},
            config=_config(tmp_path, gateway.upstream_url, False),
        )
    assert "This database was set up with general_settings.fips_mode: true" in log, log
