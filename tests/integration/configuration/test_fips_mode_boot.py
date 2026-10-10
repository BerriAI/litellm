"""LITELLM_FIPS_MODE is a boot gate: the proxy refuses to serve unless the process really enforces FIPS.

Every leg launches the real proxy binary against the suite's Postgres and asserts on what an operator sees:
exit status and the refusal text in the log. Nothing is patched inside the proxy.
"""

import hashlib
from pathlib import Path
from typing import Final

import pytest
import yaml

from tests.integration._support.client import Gateway
from tests.integration._support.process import owned_proxy, refused_boot_log

REFUSAL: Final = "LiteLLM proxy refused to start"


def _this_python_enforces_fips() -> bool:
    try:
        hashlib.md5(b"probe", usedforsecurity=True)
    except ValueError:
        return True
    return False


def test_fips_mode_refuses_to_serve_when_this_python_does_not_enforce_fips(gateway: Gateway, tmp_path: Path) -> None:
    if _this_python_enforces_fips():
        pytest.skip("Runner OpenSSL enforces FIPS, so this leg cannot observe the non-enforcing refusal")
    log: Final = refused_boot_log(gateway, tmp_path, {"LITELLM_FIPS_MODE": "true"})
    assert REFUSAL in log, log
    assert "LITELLM_FIPS_MODE" in log and "does not enforce FIPS" in log, log


@pytest.mark.parametrize("source", ("environment", "config"))
def test_fips_mode_refuses_to_serve_with_tls_verification_disabled(
    gateway: Gateway, tmp_path: Path, source: str
) -> None:
    config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    path: Final = tmp_path / "ssl_verify_off.yaml"
    settings: Final = {**config.get("litellm_settings", {}), "ssl_verify": False}
    path.write_text(yaml.safe_dump({**config, "litellm_settings": settings}))
    log: Final = (
        refused_boot_log(gateway, tmp_path, {"LITELLM_FIPS_MODE": "true", "SSL_VERIFY": "false"})
        if source == "environment"
        else refused_boot_log(gateway, tmp_path, {"LITELLM_FIPS_MODE": "true"}, config=path)
    )
    assert REFUSAL in log, log
    assert "TLS certificate verification is disabled" in log, log
    assert ("SSL_VERIFY" if source == "environment" else "litellm_settings.ssl_verify") in log, log


def test_fips_mode_refuses_to_serve_on_a_value_that_is_not_a_boolean(gateway: Gateway, tmp_path: Path) -> None:
    log: Final = refused_boot_log(gateway, tmp_path, {"LITELLM_FIPS_MODE": "enforced"})
    assert REFUSAL in log, log
    assert "LITELLM_FIPS_MODE=enforced" in log and "true or false" in log, log


def test_fips_mode_off_serves_even_with_tls_verification_disabled(gateway: Gateway, tmp_path: Path) -> None:
    with owned_proxy(gateway, tmp_path, {"LITELLM_FIPS_MODE": "false", "SSL_VERIFY": "false"}) as candidate:
        assert candidate.client.get("/health/readiness").status_code == 200
