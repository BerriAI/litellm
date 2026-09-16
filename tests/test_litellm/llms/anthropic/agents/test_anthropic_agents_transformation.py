import json

import httpx
import pytest

import litellm
from litellm.llms.anthropic.agents.transformation import AnthropicAgentsConfig
from litellm.llms.anthropic.common_utils import AnthropicError

BETA = "managed-agents-2026-04-01"
AGENT_ID = "agent_011CZkYpogX7uDKUyvBTophP"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch):
    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_API_BASE", "ANTHROPIC_BASE_URL"):
        monkeypatch.delenv(var, raising=False)


def _config() -> AnthropicAgentsConfig:
    return AnthropicAgentsConfig()


def _response(payload, status_code: int = 200) -> httpx.Response:
    return httpx.Response(
        status_code,
        content=json.dumps(payload).encode(),
        headers={"request-id": "req_123"},
        request=httpx.Request("GET", "https://api.anthropic.com/v1/agents"),
    )


def _agent_json(**overrides) -> dict:
    return {
        "type": "agent",
        "id": AGENT_ID,
        "name": "support-bot",
        "version": 3,
        "model": {"id": "claude-haiku-4-5", "effort": {"type": "low"}, "inference_geo": None, "speed": None},
        "system": "Reply tersely.",
        "tools": [{"type": "agent_toolset_20260401"}],
        "archived_at": None,
        **overrides,
    }


def test_validate_environment_sets_managed_agents_beta_and_version_with_api_key():
    headers = _config().validate_environment(headers={}, litellm_params={"api_key": "sk-ant-test"})
    assert headers["x-api-key"] == "sk-ant-test"
    assert headers["anthropic-version"] == "2023-06-01"
    assert headers["anthropic-beta"] == BETA
    assert headers["content-type"] == "application/json"


def test_validate_environment_merges_caller_betas_without_dropping_them():
    headers = _config().validate_environment(
        headers={"anthropic-beta": "files-api-2025-04-14, skills-2025-10-02"},
        litellm_params={"api_key": "sk-ant-test"},
    )
    assert headers["anthropic-beta"] == "files-api-2025-04-14,managed-agents-2026-04-01,skills-2025-10-02"


def test_validate_environment_accepts_a_list_of_betas():
    headers = _config().validate_environment(
        headers={"anthropic-beta": ["files-api-2025-04-14"]},
        litellm_params={"api_key": "sk-ant-test"},
    )
    assert headers["anthropic-beta"] == "files-api-2025-04-14,managed-agents-2026-04-01"


def test_validate_environment_keeps_a_beta_the_caller_already_sent_once():
    headers = _config().validate_environment(
        headers={"anthropic-beta": BETA},
        litellm_params={"api_key": "sk-ant-test"},
    )
    assert headers["anthropic-beta"] == BETA


def test_validate_environment_falls_back_to_env_key_without_custom_base(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-env")
    headers = _config().validate_environment(headers={}, litellm_params={})
    assert headers["x-api-key"] == "sk-ant-env"


def test_validate_environment_refuses_env_key_fallback_with_custom_api_base(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-env")
    with pytest.raises(ValueError, match="api_base"):
        _config().validate_environment(headers={}, litellm_params={"api_base": "https://evil.example"})


def test_validate_environment_requires_a_key():
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        _config().validate_environment(headers={}, litellm_params={})


def test_get_complete_url_defaults_to_the_public_api_and_honours_api_base():
    assert _config().get_complete_url(api_base=None, litellm_params={}) == "https://api.anthropic.com/v1/agents"
    assert (
        _config().get_complete_url(api_base="https://gw.example/anthropic", litellm_params={})
        == "https://gw.example/anthropic/v1/agents"
    )


def test_create_request_maps_sdk_fields_onto_the_anthropic_agent_body():
    body = _config().transform_create_request(
        name="support-bot",
        litellm_params={
            "base_agent": "claude-haiku-4-5",
            "instructions": "Reply tersely.",
            "tools": [{"type": "custom", "name": "lookup", "description": "d", "input_schema": {"type": "object"}}],
            "mcp_servers": [{"type": "url", "name": "docs", "url": "https://mcp.example/sse"}],
            "api_key": "sk-ant-test",
            "custom_llm_provider": "anthropic",
        },
    )
    assert body == {
        "name": "support-bot",
        "model": "claude-haiku-4-5",
        "system": "Reply tersely.",
        "tools": [{"type": "custom", "name": "lookup", "description": "d", "input_schema": {"type": "object"}}],
        "mcp_servers": [{"type": "url", "name": "docs", "url": "https://mcp.example/sse"}],
    }


def test_create_request_does_not_forward_the_proxy_request_metadata():
    body = _config().transform_create_request(
        name="support-bot",
        litellm_params={
            "base_agent": "claude-haiku-4-5",
            "metadata": {"user_api_key": "sk-1234", "headers": {"host": "localhost:4000"}, "user_api_key_spend": 0.0},
            "proxy_server_request": {"url": "http://localhost:4000/v1beta/agents"},
        },
    )
    assert body == {"name": "support-bot", "model": "claude-haiku-4-5"}


def test_create_request_accepts_a_model_object_with_effort():
    body = _config().transform_create_request(
        name="support-bot",
        litellm_params={"base_agent": {"id": "claude-opus-5", "effort": "high"}},
    )
    assert body["model"] == {"id": "claude-opus-5", "effort": "high"}


def test_create_request_rejects_base_environment():
    with pytest.raises(litellm.BadRequestError, match="environment"):
        _config().transform_create_request(
            name="support-bot",
            litellm_params={"base_agent": "claude-haiku-4-5", "base_environment": {"type": "cloud"}},
        )


def test_create_request_rejects_a_malformed_definition_before_calling_anthropic():
    with pytest.raises(litellm.BadRequestError, match="tools"):
        _config().transform_create_request(
            name="support-bot",
            litellm_params={"base_agent": "claude-haiku-4-5", "tools": "bash"},
        )


def test_create_response_keeps_anthropic_id_name_and_version():
    agent = _config().transform_create_response(raw_response=_response(_agent_json()), name="support-bot")
    assert agent.id == AGENT_ID
    assert agent.name == "support-bot"
    assert agent.model_dump()["version"] == 3
    assert agent.model_dump()["model"]["id"] == "claude-haiku-4-5"


def test_create_response_raises_anthropic_error_with_the_upstream_status():
    with pytest.raises(AnthropicError) as excinfo:
        _config().transform_create_response(
            raw_response=_response({"type": "error", "error": {"type": "authentication_error"}}, status_code=401),
            name="support-bot",
        )
    assert excinfo.value.status_code == 401
    assert "authentication_error" in str(excinfo.value)


def test_list_request_maps_page_size_and_page_token_onto_limit_and_page():
    url, params = _config().transform_list_request(
        api_base=None,
        litellm_params={"page_size": 50, "page_token": "page_abc", "include_archived": True},
    )
    assert url == "https://api.anthropic.com/v1/agents"
    assert params == {"limit": 50, "page": "page_abc", "include_archived": True}


def test_list_request_sends_no_query_when_nothing_is_requested():
    _, params = _config().transform_list_request(api_base=None, litellm_params={})
    assert params == {}


def test_list_response_maps_data_and_next_page():
    page = _config().transform_list_response(
        raw_response=_response({"data": [_agent_json(), _agent_json(id="agent_2")], "next_page": "page_next"})
    )
    assert [a["id"] for a in page.agents] == [AGENT_ID, "agent_2"]
    assert page.next_page_token == "page_next"


def test_list_response_reports_the_last_page_as_no_token():
    page = _config().transform_list_response(raw_response=_response({"data": [], "next_page": None}))
    assert page.agents == []
    assert page.next_page_token is None


def test_get_request_addresses_the_agent_id_and_optional_version():
    url, params = _config().transform_get_request(name=AGENT_ID, api_base=None, litellm_params={"version": 2})
    assert url == f"https://api.anthropic.com/v1/agents/{AGENT_ID}"
    assert params == {"version": 2}


def test_get_request_percent_encodes_the_agent_id():
    url, _ = _config().transform_get_request(name="agent id/with?chars", api_base=None, litellm_params={})
    assert url == "https://api.anthropic.com/v1/agents/agent%20id%2Fwith%3Fchars"


def test_get_response_is_the_agent_object():
    agent = _config().transform_get_response(raw_response=_response(_agent_json(version=7)), name=AGENT_ID)
    assert agent.id == AGENT_ID
    assert agent.model_dump()["version"] == 7


def test_delete_is_refused_and_points_at_archive():
    with pytest.raises(litellm.BadRequestError) as excinfo:
        _config().transform_delete_request(name=AGENT_ID, api_base=None, litellm_params={})
    assert f"POST https://api.anthropic.com/v1/agents/{AGENT_ID}/archive" in str(excinfo.value)


def test_list_versions_request_addresses_the_versions_collection_with_paging():
    url, params = _config().transform_list_versions_request(
        name=AGENT_ID, api_base=None, litellm_params={"page_size": 10}
    )
    assert url == f"https://api.anthropic.com/v1/agents/{AGENT_ID}/versions"
    assert params == {"limit": 10}


def test_list_versions_response_maps_data_and_next_page():
    versions = _config().transform_list_versions_response(
        raw_response=_response({"data": [_agent_json(version=1), _agent_json(version=2)], "next_page": None}),
        name=AGENT_ID,
    )
    assert [v["version"] for v in versions.agent_versions] == [1, 2]
    assert versions.next_page_token is None


def test_page_response_that_is_not_a_page_raises_anthropic_error():
    with pytest.raises(AnthropicError, match="page schema"):
        _config().transform_list_response(raw_response=_response({"data": "not-a-list"}))


def test_get_error_class_returns_an_anthropic_error():
    error = _config().get_error_class(error_message="boom", status_code=429, headers={"retry-after": "1"})
    assert isinstance(error, AnthropicError)
    assert error.status_code == 429
