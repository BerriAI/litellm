import pytest
from litellm.proxy.guardrails.guardrail_hooks.aim.aim import AimGuardrail


@pytest.mark.parametrize("api_base", [
    "https://api.aim.security/",
    "https://api.aim.security",
])
def test_aim_base_url_trailing_slash(monkeypatch, api_base):
    monkeypatch.setenv("AIM_API_KEY", "test-key")
    guardrail = AimGuardrail(api_base=api_base)
    assert guardrail.api_base == "https://api.aim.security"
    assert guardrail.ws_api_base == "wss://api.aim.security"


def test_aim_base_url_from_env(monkeypatch):
    monkeypatch.setenv("AIM_API_KEY", "test-key")
    monkeypatch.setenv("AIM_API_BASE", "https://api.aim.security/")
    guardrail = AimGuardrail(api_base=None)
    assert guardrail.api_base == "https://api.aim.security"
    assert guardrail.ws_api_base == "wss://api.aim.security"
