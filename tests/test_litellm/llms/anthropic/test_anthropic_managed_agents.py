import httpx
import pytest

from litellm.llms.anthropic.common_utils import AnthropicError
from litellm.llms.anthropic.managed_agents import (
    managed_agents_api_base,
    managed_agents_headers,
    raise_for_status,
    with_managed_agents_beta,
)

BETA = "managed-agents-2026-04-01"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch):
    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_API_BASE", "ANTHROPIC_BASE_URL"):
        monkeypatch.delenv(var, raising=False)


def test_headers_carry_key_version_and_beta():
    headers = managed_agents_headers({}, api_key="sk-ant-test", api_base=None)
    assert headers["x-api-key"] == "sk-ant-test"
    assert headers["anthropic-version"] == "2023-06-01"
    assert headers["anthropic-beta"] == BETA
    assert headers["content-type"] == "application/json"


def test_headers_merge_caller_betas_without_dropping_them():
    headers = managed_agents_headers(
        {"anthropic-beta": "files-api-2025-04-14, skills-2025-10-02"}, api_key="sk-ant-test", api_base=None
    )
    assert headers["anthropic-beta"] == "files-api-2025-04-14,managed-agents-2026-04-01,skills-2025-10-02"


def test_headers_accept_a_list_of_betas_and_dedupe_the_managed_agents_one():
    assert with_managed_agents_beta(["files-api-2025-04-14", BETA]) == f"files-api-2025-04-14,{BETA}"
    assert with_managed_agents_beta(BETA) == BETA
    assert with_managed_agents_beta(None) == BETA


def test_headers_fall_back_to_the_env_key_without_a_custom_base(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-env")
    assert managed_agents_headers({}, api_key=None, api_base=None)["x-api-key"] == "sk-ant-env"


def test_headers_refuse_the_env_key_with_a_custom_base(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-env")
    with pytest.raises(ValueError, match="api_base"):
        managed_agents_headers({}, api_key=None, api_base="https://evil.example")


def test_headers_require_a_key():
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        managed_agents_headers({}, api_key=None, api_base=None)


def test_api_base_defaults_to_the_public_api_and_honours_an_override():
    assert managed_agents_api_base(None) == "https://api.anthropic.com"
    assert managed_agents_api_base("https://gw.example/anthropic") == "https://gw.example/anthropic"


def test_raise_for_status_maps_upstream_errors_to_anthropic_error():
    response = httpx.Response(429, text='{"type":"error"}', request=httpx.Request("GET", "https://api.anthropic.com"))
    with pytest.raises(AnthropicError) as excinfo:
        raise_for_status(response)
    assert excinfo.value.status_code == 429
    assert raise_for_status(httpx.Response(200, request=httpx.Request("GET", "https://api.anthropic.com"))) is None
