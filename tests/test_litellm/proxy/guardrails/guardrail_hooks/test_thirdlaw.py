import json
from collections.abc import AsyncIterator, Sequence
from typing import TypeAlias
from unittest.mock import AsyncMock

import httpx
import pytest

from litellm.caching import DualCache
from litellm.exceptions import GuardrailRaisedException
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.guardrails.guardrail_hooks.thirdlaw import (
    ThirdlawGuardrail,
    guardrail_class_registry,
    guardrail_initializer_registry,
    initialize_guardrail,
)
from litellm.proxy.guardrails.guardrail_hooks.thirdlaw.thirdlaw import (
    _BODY_STRIP_KEYS,
    ThirdlawGuardrailMissingConfig,
)
from litellm.types.guardrails import (
    GuardrailEventHooks,
    LitellmParams,
    SupportedGuardrailIntegrations,
)
from litellm.types.llms.openai import (
    OutputTextDeltaEvent,
    ReasoningSummaryTextDeltaEvent,
    ResponseCompletedEvent,
    ResponseCreatedEvent,
    ResponsesAPIResponse,
    ResponsesAPIStreamEvents,
)
from litellm.types.proxy.guardrails.guardrail_hooks.thirdlaw import (
    ThirdlawGuardrailConfigModel,
    ThirdlawGuardrailConfigModelOptionalParams,
)
from litellm.types.utils import (
    Choices,
    Delta,
    Message,
    ModelResponse,
    ModelResponseStream,
    StreamingChoices,
)

JsonDict: TypeAlias = dict[str, object]

_API_BASE = "https://thirdlaw.test"
_ENDPOINT = "https://thirdlaw.test/guardrails/litellm/v2"


def _decision_response(body: JsonDict, status_code: int = 200) -> httpx.Response:
    return httpx.Response(
        status_code=status_code,
        json=body,
        request=httpx.Request("POST", _ENDPOINT),
    )


def _make_guardrail(*, decisions: list[httpx.Response] | None = None, **overrides: object) -> ThirdlawGuardrail:
    handler = AsyncMock(spec=AsyncHTTPHandler)
    if decisions is not None:
        handler.post.side_effect = decisions
    kwargs: dict[str, object] = {
        "api_base": _API_BASE,
        "api_key": "thirdlaw_secret",
        "guardrail_name": "thirdlaw-guard",
        "event_hook": "pre_call",
        "default_on": True,
        "async_handler": handler,
        **overrides,
    }
    return ThirdlawGuardrail(**kwargs)


def _request_data() -> JsonDict:
    body = {
        "model": "gpt-5.6",
        "messages": [{"role": "user", "content": "my api key is sk-user-secret"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the weather",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
        "temperature": 0.2,
    }
    data: JsonDict = {
        **body,
        "litellm_call_id": "call-123",
        "litellm_session_id": "session-123",
        "metadata": {
            "user_api_key_hash": "hash-1",
            "user_api_key_alias": "alias-1",
            "user_api_key_user_id": "user-1",
            "user_api_key_project_id": "project-1",
            "user_api_key_project_alias": "project-alias-1",
            "user_api_key_org_alias": "org-alias-1",
            "agent_id": "agent-1",
            "tags": ["team:eng", "env:prod"],
            # Mirrors litellm core's actual merge behavior: user_api_key_auth_metadata already
            # carries the team's own custom metadata (here, "team_cost_center"), plus a key-level
            # reserved control key ("guardrails") that litellm core's merge does not strip today.
            "user_api_key_auth_metadata": {
                "cost_center": "eng-42",
                "team_cost_center": "team-eng-42",
                "guardrails": ["leaked-key-level-guardrail"],
                "opted_out_global_guardrails": ["leaked-opt-out"],
            },
            "user_api_key_auth": UserAPIKeyAuth(
                team_metadata={"cost_center": "team-eng-42", "guardrails": ["team-guard"], "tags": ["team-tag"]},
                project_metadata={
                    "cost_center": "project-eng-42",
                    "policies": ["project-policy"],
                    # A project can route through a named provider credential; the name itself
                    # is proxy operational config, not admin-authored data, and must not leak.
                    "model_config": [{"azure": {"litellm_credentials": "prod-azure-cred"}}],
                },
                organization_metadata={"cost_center": "org-eng-42", "disable_global_guardrails": True},
            ),
            "guardrails": ["thirdlaw-guard"],
        },
        "guardrails": ["thirdlaw-guard"],
        "api_key": "sk-forwarded-provider-key",
        "secret_fields": {
            "raw_headers": {
                "authorization": "Bearer sk-live-raw",
                "x-request-id": "req-9",
            }
        },
    }
    # The proxy strips the header used for LiteLLM auth (authorization here) from its
    # sanitized header copy; the raw value survives only in secret_fields.raw_headers.
    data["proxy_server_request"] = {
        "url": "http://localhost:4000/v1/chat/completions",
        "method": "POST",
        "headers": {
            "content-type": "application/json",
            "x-request-id": "req-9",
        },
        "body": {**body, "messages": [{"role": "user", "content": "snapshot message"}]},
    }
    return data


def _model_response() -> ModelResponse:
    return ModelResponse(
        id="chatcmpl-1",
        model="gpt-5.6",
        choices=[
            Choices(
                index=0,
                finish_reason="tool_calls",
                message=Message(
                    role="assistant",
                    content="calling tool",
                    tool_calls=[
                        {
                            "id": "tool-1",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": '{"city": "sf", "token": "sk-leak"}'},
                        }
                    ],
                ),
            )
        ],
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    )


def _responses_api_response() -> ResponsesAPIResponse:
    return ResponsesAPIResponse(
        id="resp-1",
        created_at=1700000000,
        model="gpt-5.6",
        object="response",
        output=[
            {
                "id": "msg-1",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "the secret is sk-leak", "annotations": []}],
            }
        ],
        parallel_tool_calls=True,
        status="completed",
    )


def _sent_payload(guardrail: ThirdlawGuardrail) -> JsonDict:
    return guardrail.async_handler.post.call_args.kwargs["json"]


async def _run_pre_call(guardrail: ThirdlawGuardrail, data: JsonDict) -> JsonDict:
    return await guardrail.async_pre_call_hook(
        user_api_key_dict=UserAPIKeyAuth(),
        cache=DualCache(),
        data=data,
        call_type="completion",
    )


def test_requires_api_base(monkeypatch):
    monkeypatch.delenv("THIRDLAW_API_BASE", raising=False)
    with pytest.raises(ThirdlawGuardrailMissingConfig):
        ThirdlawGuardrail(api_key="k")


def test_env_fallback(monkeypatch):
    monkeypatch.setenv("THIRDLAW_API_BASE", "https://env.thirdlaw.test")
    monkeypatch.setenv("THIRDLAW_API_KEY", "env_key")
    g = ThirdlawGuardrail(guardrail_name="thirdlaw", event_hook="pre_call", default_on=True)
    assert g.api_base == "https://env.thirdlaw.test/guardrails/litellm/v2"
    assert g.http_headers["Authorization"] == "Bearer env_key"


def test_endpoint_path_not_doubled():
    g = _make_guardrail(api_base=f"{_API_BASE}/guardrails/litellm/v2")
    assert g.api_base == _ENDPOINT


def test_default_supported_event_hooks():
    g = _make_guardrail()
    assert g.supported_event_hooks == [
        GuardrailEventHooks.pre_call,
        GuardrailEventHooks.post_call,
        GuardrailEventHooks.during_call,
    ]


def test_invalid_sampling_rate_rejected():
    with pytest.raises(ValueError, match="streaming_sampling_rate"):
        _make_guardrail(streaming_sampling_rate=0)


def test_enum_value():
    assert SupportedGuardrailIntegrations.THIRDLAW.value == "thirdlaw"


def test_config_model_ui_name():
    assert ThirdlawGuardrailConfigModel.ui_friendly_name() == "ThirdLaw"


def test_registry_lookup_builds_a_working_guardrail():
    """The loader finds thirdlaw by name and the object it builds enforces decisions."""
    lp = LitellmParams(guardrail="thirdlaw", mode="pre_call", api_base=_API_BASE, api_key="k")
    built = guardrail_initializer_registry["thirdlaw"](lp, {"guardrail_name": "thirdlaw-guard"})
    assert isinstance(built, guardrail_class_registry["thirdlaw"])
    assert built.api_base == _ENDPOINT


def test_config_driven_initialization_creates_callback():
    lp = LitellmParams(guardrail="thirdlaw", mode="pre_call", api_base=_API_BASE, api_key="k")
    cb = initialize_guardrail(lp, {"guardrail_name": "thirdlaw-guard"})
    assert isinstance(cb, ThirdlawGuardrail)
    assert cb.api_base == _ENDPOINT
    assert cb.unreachable_fallback == "fail_closed"
    assert cb.guardrail_timeout == httpx.Timeout(timeout=60, connect=5.0)


def test_config_driven_initialization_propagates_streaming_overrides():
    lp = LitellmParams(
        guardrail="thirdlaw",
        mode="pre_call",
        api_base=_API_BASE,
        api_key="k",
        streaming_end_of_stream_only=False,
        streaming_sampling_rate=10,
        streaming_buffer_until_moderated=False,
    )
    cb = initialize_guardrail(lp, {"guardrail_name": "thirdlaw-guard"})
    assert cb.streaming_end_of_stream_only is False
    assert cb.streaming_sampling_rate == 10
    assert cb.streaming_buffer_until_moderated is False


def test_config_model_streaming_defaults():
    params = ThirdlawGuardrailConfigModelOptionalParams()
    assert params.streaming_end_of_stream_only is True
    assert params.streaming_buffer_until_moderated is True
    assert params.streaming_sampling_rate == 5


async def test_pre_call_payload_shape():
    g = _make_guardrail(decisions=[_decision_response({"action": "allow"})])
    data = _request_data()
    out = await _run_pre_call(g, data)

    assert out["messages"] == data["messages"]
    call_kwargs = g.async_handler.post.call_args.kwargs
    assert call_kwargs["url"] == _ENDPOINT
    assert call_kwargs["headers"]["Authorization"] == "Bearer thirdlaw_secret"

    payload = call_kwargs["json"]
    assert payload["event_type"] == "pre_call"
    assert payload["request_url"] == "http://localhost:4000/v1/chat/completions"
    assert payload["metadata"]["user_api_key_hash"] == "hash-1"
    assert payload["metadata"]["litellm_call_id"] == "call-123"
    assert payload["metadata"]["model"] == "gpt-5.6"
    assert payload["request_body"]["model"] == "gpt-5.6"
    assert payload["request_body"]["tools"][0]["function"]["name"] == "get_weather"
    assert payload["request_body"]["temperature"] == 0.2
    for stripped_key in ("secret_fields", "api_key", "metadata", "guardrails", "litellm_call_id", "litellm_session_id"):
        assert stripped_key not in payload["request_body"]
    assert "response_body" not in payload
    assert "sk-forwarded-provider-key" not in json.dumps(payload)


_PROVIDER_CREDENTIALS: JsonDict = {
    "api_key": "sk-forwarded-provider-key",
    "aws_access_key_id": "AKIAIOSFODNN7EXAMPLE",
    "aws_secret_access_key": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
    "aws_session_token": "FwoGZXIvYXdzEJr//////////wEaDEXAMPLE",
    "azure_ad_token": "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.azure-ad",
    "azure_password": "hunter2-azure",
    "client_secret": "oauth-client-secret-value",
    "vertex_credentials": {"type": "service_account", "private_key": "-----BEGIN PRIVATE KEY-----"},
    "watsonx_region_name": "eu-de",
    "extra_headers": {"authorization": "Bearer sk-provider-side"},
}


async def test_no_provider_credential_reaches_thirdlaw():
    g = _make_guardrail(decisions=[_decision_response({"action": "allow"})])
    data = {**_request_data(), **_PROVIDER_CREDENTIALS}
    await _run_pre_call(g, data)
    payload = _sent_payload(g)
    for key in _PROVIDER_CREDENTIALS:
        assert key not in payload["request_body"], key
    serialized = json.dumps(payload)
    for secret in (
        "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
        "FwoGZXIvYXdzEJr//////////wEaDEXAMPLE",
        "azure-ad",
        "hunter2-azure",
        "oauth-client-secret-value",
        "BEGIN PRIVATE KEY",
        "sk-provider-side",
        "sk-forwarded-provider-key",
    ):
        assert secret not in serialized, secret
    assert payload["request_body"]["messages"] == data["messages"]


def test_every_credential_param_litellm_declares_is_withheld():
    """The strip set is derived from CredentialLiteLLMParams, so a provider credential
    added upstream is withheld without an edit here. Guards against it being re-frozen
    into a hand-maintained literal that silently falls behind."""
    from litellm.types.router import CredentialLiteLLMParams

    assert set(CredentialLiteLLMParams.model_fields) <= set(_BODY_STRIP_KEYS)


async def test_modify_request_cannot_restore_protected_root_controls():
    """add_litellm_data_to_request strips these from caller input; the guardrail decision lands
    after that strip, so a compromised or hostile service must not be able to put them back."""
    g = _make_guardrail(
        decisions=[
            _decision_response(
                {
                    "action": "modify_request",
                    "request_body": {
                        "messages": [{"role": "user", "content": "scrubbed"}],
                        "max_agentic_loops": 500,
                        "_code_interpreter_interception_active": True,
                        "_code_interpreter_interception_sandbox_key": "sk-sandbox",
                        "mock_response": "free lunch",
                        "callbacks": ["attacker_callback"],
                    },
                }
            )
        ]
    )
    out = await _run_pre_call(g, _request_data())
    assert out["messages"] == [{"role": "user", "content": "scrubbed"}]
    for key in (
        "max_agentic_loops",
        "_code_interpreter_interception_active",
        "_code_interpreter_interception_sandbox_key",
        "mock_response",
        "callbacks",
    ):
        assert key not in out, key


async def test_modify_request_cannot_redirect_routing_or_credentials():
    """custom_llm_provider and litellm_credential_name pick which provider and which stored
    credential the call runs against, both authorized before this hook ever runs."""
    g = _make_guardrail(
        decisions=[
            _decision_response(
                {
                    "action": "modify_request",
                    "request_body": {
                        "custom_llm_provider": "attacker_provider",
                        "litellm_credential_name": "someone-elses-credential",
                    },
                }
            )
        ]
    )
    data = _request_data()
    out = await _run_pre_call(g, data)
    assert "custom_llm_provider" not in out
    assert "litellm_credential_name" not in out
    assert out["model"] == data["model"]


async def test_modify_response_cannot_rename_a_non_streaming_response_id():
    """litellm hands out an encrypted response id carrying the deployment the turn routed to, and
    the client chains the next turn off it, so no response shape may have it rewritten."""
    g = _make_guardrail(
        decisions=[
            _decision_response(
                {
                    "action": "modify_response",
                    "response_body": {"id": "resp_attacker_supplied", "choices": []},
                }
            )
        ]
    )
    response = _model_response()
    original_id = response.id
    out = await g.async_post_call_success_hook(
        data=_request_data(), user_api_key_dict=UserAPIKeyAuth(), response=response
    )
    assert out.id == original_id


async def test_modify_request_cannot_inject_a_provider_credential():
    g = _make_guardrail(
        decisions=[
            _decision_response(
                {
                    "action": "modify_request",
                    "request_body": {
                        "messages": [{"role": "user", "content": "scrubbed"}],
                        "aws_secret_access_key": "attacker-supplied",
                        "azure_ad_token": "attacker-supplied",
                        "vertex_credentials": {"private_key": "attacker-supplied"},
                    },
                }
            )
        ]
    )
    out = await _run_pre_call(g, _request_data())
    assert out["messages"] == [{"role": "user", "content": "scrubbed"}]
    for key in ("aws_secret_access_key", "azure_ad_token", "vertex_credentials"):
        assert key not in out, key


async def test_pre_call_sends_live_body_not_snapshot():
    g = _make_guardrail(decisions=[_decision_response({"action": "allow"})])
    await _run_pre_call(g, _request_data())
    messages = _sent_payload(g)["request_body"]["messages"]
    assert messages == [{"role": "user", "content": "my api key is sk-user-secret"}]


async def test_pre_call_payload_includes_extended_identity_and_hierarchy_metadata():
    g = _make_guardrail(decisions=[_decision_response({"action": "allow"})])
    await _run_pre_call(g, _request_data())
    metadata = _sent_payload(g)["metadata"]

    assert metadata["litellm_session_id"] == "session-123"
    assert metadata["user_api_key_project_id"] == "project-1"
    assert metadata["user_api_key_project_alias"] == "project-alias-1"
    assert metadata["user_api_key_org_alias"] == "org-alias-1"
    assert metadata["agent_id"] == "agent-1"
    assert metadata["tags"] == ["team:eng", "env:prod"]
    assert metadata["user_api_key_auth_metadata"] == {"cost_center": "eng-42", "team_cost_center": "team-eng-42"}


async def test_pre_call_payload_falls_back_to_metadata_session_id():
    """A caller can declare a session purely via `metadata.session_id` (litellm's own

    documented alternative to the `x-litellm-session-id` header, see the "missing_session_id"
    reject-policy error message in litellm_pre_call_utils.py) without ever populating the
    call-level `litellm_session_id` field this guardrail also reads.
    """
    g = _make_guardrail(decisions=[_decision_response({"action": "allow"})])
    data = _request_data()
    del data["litellm_session_id"]
    data["metadata"]["session_id"] = "client-declared-session"
    await _run_pre_call(g, data)
    metadata = _sent_payload(g)["metadata"]

    assert metadata["litellm_session_id"] == "client-declared-session"


async def test_pre_call_payload_strips_reserved_keys_from_hierarchy_metadata():
    g = _make_guardrail(decisions=[_decision_response({"action": "allow"})])
    await _run_pre_call(g, _request_data())
    metadata = _sent_payload(g)["metadata"]

    assert metadata["project_metadata"] == {"cost_center": "project-eng-42"}
    assert metadata["organization_metadata"] == {"cost_center": "org-eng-42"}
    for reserved_key in (
        "guardrails",
        "tags",
        "policies",
        "disable_global_guardrails",
        "opted_out_global_guardrails",
        "model_config",
    ):
        assert reserved_key not in metadata["user_api_key_auth_metadata"]
        assert reserved_key not in metadata["project_metadata"]
        assert reserved_key not in metadata["organization_metadata"]


async def test_pre_call_payload_does_not_duplicate_team_metadata():
    """team_metadata is already merged into user_api_key_auth_metadata by litellm core

    (litellm_pre_call_utils.py's add_management_endpoint_metadata_to_request_metadata layers
    the team's own custom metadata onto that same dict), so it must not be forwarded again
    under its own key.
    """
    g = _make_guardrail(decisions=[_decision_response({"action": "allow"})])
    await _run_pre_call(g, _request_data())
    metadata = _sent_payload(g)["metadata"]

    assert "team_metadata" not in metadata
    assert metadata["user_api_key_auth_metadata"]["team_cost_center"] == "team-eng-42"


async def test_pre_call_payload_omits_hierarchy_metadata_when_absent():
    g = _make_guardrail(decisions=[_decision_response({"action": "allow"})])
    data = _request_data()
    data["metadata"].pop("user_api_key_auth", None)
    data["metadata"].pop("tags", None)
    await _run_pre_call(g, data)
    metadata = _sent_payload(g)["metadata"]

    for absent_field in ("tags", "project_metadata", "organization_metadata"):
        assert absent_field not in metadata


async def test_all_headers_forwarded_without_credentials_by_default():
    g = _make_guardrail(decisions=[_decision_response({"action": "allow"})])
    await _run_pre_call(g, _request_data())
    headers = _sent_payload(g)["request_headers"]
    assert headers["content-type"] == "application/json"
    assert headers["x-request-id"] == "req-9"
    assert "authorization" not in headers
    assert "sk-live-raw" not in json.dumps(_sent_payload(g))


async def test_additional_headers_opts_into_raw_values():
    g = _make_guardrail(
        decisions=[_decision_response({"action": "allow"})],
        additional_headers="Authorization , x-missing",
    )
    await _run_pre_call(g, _request_data())
    headers = _sent_payload(g)["request_headers"]
    assert headers["authorization"] == "Bearer sk-live-raw"
    assert headers["x-request-id"] == "req-9"
    assert "x-missing" not in headers


async def test_pre_call_block_raises_with_status():
    g = _make_guardrail(
        decisions=[_decision_response({"action": "block", "message": "policy violation", "response_status": 422})]
    )
    with pytest.raises(GuardrailRaisedException) as exc_info:
        await _run_pre_call(g, _request_data())
    assert exc_info.value.message == "policy violation"
    assert exc_info.value.status_code == 422
    assert exc_info.value.blocked_content is True


async def test_pre_call_modify_request_applies_content_keys_only():
    modified_tools = [
        {
            "type": "function",
            "function": {"name": "get_weather", "description": "[sanitized]", "parameters": {"type": "object"}},
        }
    ]
    g = _make_guardrail(
        decisions=[
            _decision_response(
                {
                    "action": "modify_request",
                    "request_body": {
                        "messages": [{"role": "user", "content": "my api key is [REDACTED]"}],
                        "tools": modified_tools,
                        "temperature": 0.9,
                        "model": "attacker-model",
                        "guardrails": [],
                        "metadata": {"user_api_key_hash": "forged"},
                        "stream": True,
                        "secret_fields": {"raw_headers": {}},
                    },
                }
            )
        ]
    )
    data = _request_data()
    out = await _run_pre_call(g, data)

    assert out["messages"] == [{"role": "user", "content": "my api key is [REDACTED]"}]
    assert out["tools"] == modified_tools
    assert out["temperature"] == 0.9
    assert out["model"] == "gpt-5.6"
    assert out["guardrails"] == ["thirdlaw-guard"]
    assert out["metadata"]["user_api_key_hash"] == "hash-1"
    assert "stream" not in out
    assert out["secret_fields"]["raw_headers"]["authorization"] == "Bearer sk-live-raw"
    assert data["messages"] == [{"role": "user", "content": "my api key is sk-user-secret"}]


async def test_pre_call_records_single_guardrail_trace():
    g = _make_guardrail(decisions=[_decision_response({"action": "allow"})])
    data = _request_data()
    await _run_pre_call(g, data)
    traces = data["metadata"]["standard_logging_guardrail_information"]
    assert len(traces) == 1
    assert traces[0]["guardrail_status"] == "success"
    assert traces[0]["guardrail_response"]["action"] == "allow"


async def test_pre_call_block_records_intervened_trace():
    g = _make_guardrail(decisions=[_decision_response({"action": "block", "message": "no"})])
    data = _request_data()
    with pytest.raises(GuardrailRaisedException):
        await _run_pre_call(g, data)
    traces = data["metadata"]["standard_logging_guardrail_information"]
    assert len(traces) == 1
    assert traces[0]["guardrail_status"] == "guardrail_intervened"


async def test_during_call_block_raises():
    g = _make_guardrail(decisions=[_decision_response({"action": "block", "message": "denied"})])
    with pytest.raises(GuardrailRaisedException):
        await g.async_moderation_hook(data=_request_data(), user_api_key_dict=UserAPIKeyAuth(), call_type="completion")


async def test_during_call_modify_is_ignored():
    g = _make_guardrail(
        decisions=[_decision_response({"action": "modify_request", "request_body": {"messages": [{"role": "user"}]}})]
    )
    data = _request_data()
    out = await g.async_moderation_hook(data=data, user_api_key_dict=UserAPIKeyAuth(), call_type="completion")
    assert out is data
    assert data["messages"] == [{"role": "user", "content": "my api key is sk-user-secret"}]
    assert _sent_payload(g)["event_type"] == "during_call"


async def test_post_call_payload_includes_response_and_prefers_snapshot_body():
    g = _make_guardrail(decisions=[_decision_response({"action": "allow"})])
    response = _model_response()
    out = await g.async_post_call_success_hook(
        data=_request_data(), user_api_key_dict=UserAPIKeyAuth(), response=response
    )
    assert out is response
    payload = _sent_payload(g)
    assert payload["event_type"] == "post_call"
    assert payload["response_body"]["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "get_weather"
    assert payload["request_body"]["messages"] == [{"role": "user", "content": "snapshot message"}]


async def test_post_call_modify_response_rewrites_content_and_tool_calls():
    g = _make_guardrail(
        decisions=[
            _decision_response(
                {
                    "action": "modify_response",
                    "response_body": {
                        "choices": [
                            {
                                "index": 0,
                                "finish_reason": "tool_calls",
                                "message": {
                                    "role": "assistant",
                                    "content": "calling tool [sanitized]",
                                    "tool_calls": [
                                        {
                                            "id": "tool-1",
                                            "type": "function",
                                            "function": {
                                                "name": "get_weather",
                                                "arguments": '{"city": "sf", "token": "[REDACTED]"}',
                                            },
                                        }
                                    ],
                                },
                            }
                        ]
                    },
                }
            )
        ]
    )
    out = await g.async_post_call_success_hook(
        data=_request_data(), user_api_key_dict=UserAPIKeyAuth(), response=_model_response()
    )
    assert isinstance(out, ModelResponse)
    assert out.choices[0].message.content == "calling tool [sanitized]"
    assert out.choices[0].message.tool_calls[0].function.arguments == '{"city": "sf", "token": "[REDACTED]"}'
    assert out.id == "chatcmpl-1"
    assert out.usage.total_tokens == 15


async def test_post_call_modify_response_merges_dict_responses():
    g = _make_guardrail(
        decisions=[
            _decision_response(
                {
                    "action": "modify_response",
                    "response_body": {"content": [{"type": "text", "text": "[MASKED]"}]},
                }
            )
        ]
    )
    anthropic_response = {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "the secret is sk-leak"}],
    }
    out = await g.async_post_call_success_hook(
        data=_request_data(), user_api_key_dict=UserAPIKeyAuth(), response=anthropic_response
    )
    assert out == {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "[MASKED]"}],
    }


async def test_post_call_modify_response_rewrites_responses_api_output():
    """ResponsesAPIResponse is a Pydantic model, not a ModelResponse or a raw dict."""
    g = _make_guardrail(
        decisions=[
            _decision_response(
                {
                    "action": "modify_response",
                    "response_body": {
                        "output": [
                            {
                                "id": "msg-1",
                                "type": "message",
                                "role": "assistant",
                                "status": "completed",
                                "content": [{"type": "output_text", "text": "[REDACTED]", "annotations": []}],
                            }
                        ]
                    },
                }
            )
        ]
    )
    out = await g.async_post_call_success_hook(
        data=_request_data(), user_api_key_dict=UserAPIKeyAuth(), response=_responses_api_response()
    )
    assert isinstance(out, ResponsesAPIResponse)
    assert out.model_dump()["output"][0]["content"][0]["text"] == "[REDACTED]"


@pytest.mark.parametrize("response_factory", [_model_response, _responses_api_response])
async def test_post_call_modify_response_carries_hidden_params(response_factory):
    """model_validate() builds a fresh instance that starts with empty _hidden_params --
    the original's must be carried across explicitly or response-header forwarding breaks.
    """
    response = response_factory()
    response._hidden_params["additional_headers"] = {"x-request-id": "abc123"}
    g = _make_guardrail(
        decisions=[_decision_response({"action": "modify_response", "response_body": {"model": "gpt-5.6-redacted"}})]
    )
    out = await g.async_post_call_success_hook(
        data=_request_data(), user_api_key_dict=UserAPIKeyAuth(), response=response
    )
    assert out._hidden_params.get("additional_headers") == {"x-request-id": "abc123"}


async def test_post_call_block_raises():
    g = _make_guardrail(decisions=[_decision_response({"action": "block", "message": "leaked secret"})])
    with pytest.raises(GuardrailRaisedException) as exc_info:
        await g.async_post_call_success_hook(
            data=_request_data(), user_api_key_dict=UserAPIKeyAuth(), response=_model_response()
        )
    assert exc_info.value.message == "leaked secret"


async def test_post_call_malformed_modified_response_fails_closed():
    g = _make_guardrail(
        decisions=[_decision_response({"action": "modify_response", "response_body": {"choices": "garbage"}})]
    )
    with pytest.raises(GuardrailRaisedException):
        await g.async_post_call_success_hook(
            data=_request_data(), user_api_key_dict=UserAPIKeyAuth(), response=_model_response()
        )


async def test_modify_request_without_a_body_leaves_the_request_alone():
    """A decision that says modify but carries nothing must not be read as "replace with empty"."""
    g = _make_guardrail(decisions=[_decision_response({"action": "modify_request"})])
    data = _request_data()
    out = await _run_pre_call(g, data)
    assert out["messages"] == data["messages"]
    assert out["tools"] == data["tools"]


async def test_modify_response_on_pre_call_is_ignored():
    """The response does not exist yet on pre_call, so the decision has nothing to apply to and
    must not be mistaken for a request rewrite."""
    g = _make_guardrail(
        decisions=[
            _decision_response(
                {
                    "action": "modify_response",
                    "response_body": {"choices": []},
                    "request_body": {"messages": [{"role": "user", "content": "smuggled"}]},
                }
            )
        ]
    )
    data = _request_data()
    original_messages = list(data["messages"])
    out = await _run_pre_call(g, data)
    assert out["messages"] == original_messages


async def test_modify_request_on_during_call_is_ignored():
    """during_call runs beside the LLM call, so the request has already gone out."""
    g = _make_guardrail(
        decisions=[
            _decision_response(
                {"action": "modify_request", "request_body": {"messages": [{"role": "user", "content": "late"}]}}
            )
        ]
    )
    data = _request_data()
    out = await g.async_moderation_hook(data=data, user_api_key_dict=UserAPIKeyAuth(), call_type="completion")
    assert out["messages"] == data["messages"]


async def test_post_call_skips_a_response_shape_it_cannot_serialize():
    """An unscannable response type must not raise and take down the call."""
    g = _make_guardrail(decisions=[_decision_response({"action": "allow"})])
    response = object()
    out = await g.async_post_call_success_hook(
        data=_request_data(), user_api_key_dict=UserAPIKeyAuth(), response=response
    )
    assert out is response
    assert g.async_handler.post.await_count == 0


async def test_modify_response_on_an_unsupported_shape_returns_the_original():
    """The scan ran and the service asked for a rewrite, but there is no way to apply it to this
    shape, so the original is returned rather than a half-applied one."""

    class _Unsupported:
        def model_dump(self, *args: object, **kwargs: object) -> list[int]:
            return [1, 2, 3]

    g = _make_guardrail(decisions=[_decision_response({"action": "modify_response", "response_body": {"choices": []}})])
    response = _Unsupported()
    out = await g.async_post_call_success_hook(
        data=_request_data(), user_api_key_dict=UserAPIKeyAuth(), response=response
    )
    assert out is response


async def test_a_rewrite_that_fails_validation_fails_closed():
    """A malformed rewrite must block rather than reach the client half-applied."""
    g = _make_guardrail(
        decisions=[
            _decision_response({"action": "modify_response", "response_body": {"choices": [{"index": "not-an-int"}]}})
        ]
    )
    with pytest.raises(GuardrailRaisedException) as exc_info:
        await g.async_post_call_success_hook(
            data=_request_data(), user_api_key_dict=UserAPIKeyAuth(), response=_model_response()
        )
    assert "malformed modified response" in exc_info.value.message


async def test_a_responses_rewrite_that_fails_validation_fails_closed():
    g = _make_guardrail(
        decisions=[_decision_response({"action": "modify_response", "response_body": {"created_at": "nonsense"}})]
    )
    with pytest.raises(GuardrailRaisedException) as exc_info:
        await g.async_post_call_success_hook(
            data=_request_data(), user_api_key_dict=UserAPIKeyAuth(), response=_responses_api_response()
        )
    assert "malformed modified response" in exc_info.value.message


async def test_headers_fall_back_to_metadata_when_there_is_no_proxy_server_request():
    """Non-proxy call paths carry the inbound headers on metadata instead."""
    g = _make_guardrail(decisions=[_decision_response({"action": "allow"})])
    data = _request_data()
    del data["proxy_server_request"]
    data["metadata"]["headers"] = {"x-demo-trace": "trace-42"}
    await _run_pre_call(g, data)
    assert _sent_payload(g)["request_headers"]["x-demo-trace"] == "trace-42"


async def test_an_unserializable_request_body_is_posted_as_absent_not_raised():
    """A body the guardrail cannot serialize must not fail live traffic; the scan still runs."""

    class _Unserializable:
        def __repr__(self) -> str:
            raise RuntimeError("no repr for you")

    g = _make_guardrail(decisions=[_decision_response({"action": "allow"})])
    data = _request_data()
    data["messages"] = _Unserializable()
    out = await _run_pre_call(g, data)
    assert out is data
    assert "request_body" not in _sent_payload(g)


class _DictDumping:
    """Serializes for the scan but is not a shape the rewrite knows how to rebuild."""

    def model_dump(self, *args: object, **kwargs: object) -> JsonDict:
        return {"choices": [{"message": {"content": "hi"}}]}


async def test_a_rewrite_for_a_shape_it_cannot_rebuild_returns_the_original():
    """The scan runs because the response serializes, but there is no constructor for this shape,
    so the original is returned rather than a dict standing in for a typed response."""
    g = _make_guardrail(
        decisions=[
            _decision_response(
                {"action": "modify_response", "response_body": {"choices": [{"message": {"content": "nope"}}]}}
            )
        ]
    )
    response = _DictDumping()
    out = await g.async_post_call_success_hook(
        data=_request_data(), user_api_key_dict=UserAPIKeyAuth(), response=response
    )
    assert out is response


async def test_a_response_that_cannot_be_serialized_skips_the_scan():
    """A response whose dump raises must not take the call down with it."""

    class _Exploding:
        def model_dump(self, *args: object, **kwargs: object) -> JsonDict:
            raise RuntimeError("cannot dump")

    g = _make_guardrail(decisions=[_decision_response({"action": "allow"})])
    response = _Exploding()
    out = await g.async_post_call_success_hook(
        data=_request_data(), user_api_key_dict=UserAPIKeyAuth(), response=response
    )
    assert out is response
    assert g.async_handler.post.await_count == 0


def test_the_guardrail_advertises_its_config_model():
    """The proxy reads the config schema off the class to render and validate the guardrail, so a
    missing model silently drops every documented default."""
    model = ThirdlawGuardrail.get_config_model()
    assert model is not None
    assert model.ui_friendly_name() == "ThirdLaw"
    optional_params_model = model.model_fields["optional_params"].annotation
    assert ThirdlawGuardrailConfigModelOptionalParams in getattr(optional_params_model, "__args__", ())
    assert ThirdlawGuardrailConfigModelOptionalParams().send_stream_chunks is False


def _connect_error() -> httpx.ConnectError:
    return httpx.ConnectError("connection refused", request=httpx.Request("POST", _ENDPOINT))


async def test_unreachable_fail_closed_raises():
    g = _make_guardrail()
    g.async_handler.post.side_effect = _connect_error()
    with pytest.raises(GuardrailRaisedException) as exc_info:
        await _run_pre_call(g, _request_data())
    assert exc_info.value.blocked_content is False


async def test_unreachable_fail_open_passes_through():
    g = _make_guardrail(unreachable_fallback="fail_open")
    g.async_handler.post.side_effect = _connect_error()
    data = _request_data()
    out = await _run_pre_call(g, data)
    assert out is data
    traces = data["metadata"]["standard_logging_guardrail_information"]
    assert traces[0]["guardrail_status"] == "guardrail_failed_to_respond"


async def test_http_500_fails_closed_even_with_fail_open():
    g = _make_guardrail(
        unreachable_fallback="fail_open",
        decisions=[_decision_response({"detail": "boom"}, status_code=500)],
    )
    with pytest.raises(GuardrailRaisedException):
        await _run_pre_call(g, _request_data())


async def test_http_503_respects_fail_open():
    g = _make_guardrail(
        unreachable_fallback="fail_open",
        decisions=[_decision_response({"detail": "overloaded"}, status_code=503)],
    )
    data = _request_data()
    assert await _run_pre_call(g, data) is data


_EVALUATION_TIMEOUT_MESSAGE = (
    "ThirdLaw did not finish evaluating this request within 10 seconds, so the request was not evaluated."
)


def _fail_closed_detail(reason: str = "evaluation_timeout") -> JsonDict:
    return {
        "detail": {
            "error": "evaluation_unavailable",
            "reason": reason,
            "message": _EVALUATION_TIMEOUT_MESSAGE,
        }
    }


def _html_error_response(status_code: int = 503) -> httpx.Response:
    return httpx.Response(
        status_code=status_code,
        content=b"<html><body>503 Service Temporarily Unavailable</body></html>",
        headers={"content-type": "text/html"},
        request=httpx.Request("POST", _ENDPOINT),
    )


async def _raised_message(g: ThirdlawGuardrail) -> str:
    with pytest.raises(GuardrailRaisedException) as exc_info:
        await _run_pre_call(g, _request_data())
    return exc_info.value.message


async def test_fail_closed_503_surfaces_the_service_reason_not_the_status_line():
    g = _make_guardrail(decisions=[_decision_response(_fail_closed_detail(), status_code=503)])
    message = await _raised_message(g)
    assert _EVALUATION_TIMEOUT_MESSAGE in message
    assert "reason=evaluation_timeout" in message
    assert "Service Unavailable" not in message


async def test_fail_closed_503_records_the_service_reason_on_the_guardrail_trace():
    g = _make_guardrail(decisions=[_decision_response(_fail_closed_detail(), status_code=503)])
    data = _request_data()
    with pytest.raises(GuardrailRaisedException):
        await _run_pre_call(g, data)
    traces = data["metadata"]["standard_logging_guardrail_information"]
    assert traces[0]["guardrail_status"] == "guardrail_failed_to_respond"
    assert traces[0]["guardrail_response"]["error"] == f"{_EVALUATION_TIMEOUT_MESSAGE} (reason=evaluation_timeout)"


async def test_non_json_503_body_falls_back_to_the_status_line():
    g = _make_guardrail()
    g.async_handler.post.return_value = _html_error_response()
    assert "503 Service Unavailable" in await _raised_message(g)


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ({"error": "evaluation_unavailable"}, "503 Service Unavailable"),
        ({"detail": {"error": "evaluation_unavailable"}}, "503 Service Unavailable"),
        ({"detail": {"message": ""}}, "503 Service Unavailable"),
        ({"detail": "policy engine is draining"}, "policy engine is draining"),
    ],
)
async def test_503_detail_shapes_a_service_may_send(body: JsonDict, expected: str):
    g = _make_guardrail(decisions=[_decision_response(body, status_code=503)])
    assert expected in await _raised_message(g)


async def test_structured_503_passes_through_under_fail_open_and_still_records_the_reason():
    g = _make_guardrail(
        unreachable_fallback="fail_open",
        decisions=[_decision_response(_fail_closed_detail(reason="evaluation_capacity_exceeded"), status_code=503)],
    )
    data = _request_data()
    assert await _run_pre_call(g, data) is data
    traces = data["metadata"]["standard_logging_guardrail_information"]
    assert traces[0]["guardrail_status"] == "guardrail_failed_to_respond"
    assert "reason=evaluation_capacity_exceeded" in traces[0]["guardrail_response"]["error"]


def _stream_chunks() -> list[ModelResponseStream]:
    return [
        ModelResponseStream(
            id="chunk-1",
            model="gpt-5.6",
            choices=[StreamingChoices(index=0, delta=Delta(role="assistant", content="the secret "))],
        ),
        ModelResponseStream(
            id="chunk-1",
            model="gpt-5.6",
            choices=[StreamingChoices(index=0, delta=Delta(content="is sk-leak"))],
        ),
        ModelResponseStream(
            id="chunk-1",
            model="gpt-5.6",
            choices=[StreamingChoices(index=0, delta=Delta(content=None), finish_reason="stop")],
        ),
    ]


async def _aiter(items: Sequence[object]) -> AsyncIterator[object]:
    for item in items:
        yield item


async def _collect(agen: AsyncIterator[object]) -> list[object]:
    return [item async for item in agen]


async def test_streaming_buffered_allow_replays_original_chunks():
    g = _make_guardrail(decisions=[_decision_response({"action": "allow"})])
    chunks = _stream_chunks()
    out = await _collect(
        g.async_post_call_streaming_iterator_hook(
            user_api_key_dict=UserAPIKeyAuth(), response=_aiter(chunks), request_data=_request_data()
        )
    )
    assert out == chunks
    payload = _sent_payload(g)
    assert payload["event_type"] == "post_call"
    assert payload["response_body"]["choices"][0]["message"]["content"] == "the secret is sk-leak"


async def test_streaming_buffered_modify_emits_rewritten_response():
    g = _make_guardrail(
        decisions=[
            _decision_response(
                {
                    "action": "modify_response",
                    "response_body": {
                        "choices": [
                            {
                                "index": 0,
                                "finish_reason": "stop",
                                "message": {"role": "assistant", "content": "the secret is [REDACTED]"},
                            }
                        ]
                    },
                }
            )
        ]
    )
    out = await _collect(
        g.async_post_call_streaming_iterator_hook(
            user_api_key_dict=UserAPIKeyAuth(), response=_aiter(_stream_chunks()), request_data=_request_data()
        )
    )
    emitted_text = "".join(
        choice.delta.content or ""
        for chunk in out
        if isinstance(chunk, ModelResponseStream)
        for choice in chunk.choices
    )
    assert emitted_text == "the secret is [REDACTED]"
    assert "sk-leak" not in emitted_text


def _redacting_modify_decision() -> httpx.Response:
    return _decision_response(
        {
            "action": "modify_response",
            "response_body": {
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "the secret is [REDACTED]"},
                    }
                ]
            },
        }
    )


async def test_masking_guardrail_does_not_buffer_and_replay_originals():
    """Buffered replay hands back the unredacted chunks, so a masking guardrail must not buffer."""
    g = _make_guardrail(decisions=[_redacting_modify_decision()], mask_response_content=True)
    out = await _collect(
        g.async_post_call_streaming_iterator_hook(
            user_api_key_dict=UserAPIKeyAuth(), response=_aiter(_stream_chunks()), request_data=_request_data()
        )
    )
    emitted = "".join(
        choice.delta.content or ""
        for chunk in out
        if isinstance(chunk, ModelResponseStream)
        for choice in chunk.choices
    )
    assert emitted == "the secret is sk-leak"
    assert "[REDACTED]" not in emitted


async def test_streaming_buffered_block_raises_streaming_callback_error():
    from litellm.proxy.proxy_server import StreamingCallbackError

    g = _make_guardrail(decisions=[_decision_response({"action": "block", "message": "leaked secret"})])
    with pytest.raises(StreamingCallbackError, match="leaked secret"):
        await _collect(
            g.async_post_call_streaming_iterator_hook(
                user_api_key_dict=UserAPIKeyAuth(), response=_aiter(_stream_chunks()), request_data=_request_data()
            )
        )


async def test_streaming_buffered_holds_chunks_until_decision():
    call_order: list[str] = []

    async def _recording_stream() -> AsyncIterator[object]:
        for chunk in _stream_chunks():
            call_order.append("chunk_consumed")
            yield chunk

    g = _make_guardrail()

    async def _post(*args, **kwargs):
        call_order.append("guardrail_called")
        return _decision_response({"action": "allow"})

    g.async_handler.post.side_effect = _post
    out = await _collect(
        g.async_post_call_streaming_iterator_hook(
            user_api_key_dict=UserAPIKeyAuth(), response=_recording_stream(), request_data=_request_data()
        )
    )
    assert len(out) == 3
    assert call_order == ["chunk_consumed", "chunk_consumed", "chunk_consumed", "guardrail_called"]


async def test_streaming_sampled_interim_block_terminates_stream():
    from litellm.proxy.proxy_server import StreamingCallbackError

    g = _make_guardrail(
        streaming_buffer_until_moderated=False,
        streaming_end_of_stream_only=False,
        streaming_sampling_rate=1,
        decisions=[_decision_response({"action": "block", "message": "bad interim"})],
    )
    agen = g.async_post_call_streaming_iterator_hook(
        user_api_key_dict=UserAPIKeyAuth(), response=_aiter(_stream_chunks()), request_data=_request_data()
    )
    first = await agen.__anext__()
    assert isinstance(first, ModelResponseStream)
    with pytest.raises(StreamingCallbackError, match="bad interim"):
        await agen.__anext__()


def _truncated_tool_use_sse_frames() -> list[bytes]:
    """A tool_use block whose argument JSON was cut off by max_tokens.

    The secret only ever exists inside the unparseable fragment, so if assembly drops it the
    scan never sees it while the original frames still reach the client.
    """
    events = [
        (
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": "msg_tool",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-sonnet-5",
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 9, "output_tokens": 0},
                },
            },
        ),
        (
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "tool_use", "id": "toolu_1", "name": "lookup", "input": {}},
            },
        ),
        (
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "input_json_delta", "partial_json": '{"query": "sk-live-TRUNC'},
            },
        ),
        (
            "message_delta",
            {"type": "message_delta", "delta": {"stop_reason": "max_tokens"}, "usage": {"output_tokens": 5}},
        ),
        ("message_stop", {"type": "message_stop"}),
    ]
    return [f"event: {name}\ndata: {json.dumps(body)}\n\n".encode() for name, body in events]


async def test_a_truncated_tool_argument_still_reaches_the_scan():
    """Text the model emitted inside a tool argument reaches the client in the replayed frames,
    so dropping it from the assembled body would let a caller pick truncation as a bypass."""
    g = _make_guardrail(decisions=[_decision_response({"action": "allow"})])
    await _collect(
        g.async_post_call_streaming_iterator_hook(
            user_api_key_dict=UserAPIKeyAuth(),
            response=_aiter(_truncated_tool_use_sse_frames()),
            request_data=_request_data(),
        )
    )
    assert "sk-live-TRUNC" in json.dumps(_sent_payload(g)["response_body"])


async def test_a_truncated_tool_argument_can_still_be_blocked():
    """The scan seeing the text is only half the fix; the block decision has to land."""
    g = _make_guardrail(decisions=[_decision_response({"action": "block", "message": "secret in tool args"})])
    out = await _collect(
        g.async_post_call_streaming_iterator_hook(
            user_api_key_dict=UserAPIKeyAuth(),
            response=_aiter(_truncated_tool_use_sse_frames()),
            request_data=_request_data(),
        )
    )
    blob = b"".join(item for item in out if isinstance(item, bytes))
    assert b"secret in tool args" in blob
    assert b"sk-live-TRUNC" not in blob


def _anthropic_sse_frames() -> list[bytes]:
    events = [
        (
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": "msg_abc",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-sonnet-5",
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 9, "output_tokens": 0},
                },
            },
        ),
        (
            "content_block_start",
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        ),
        (
            "content_block_delta",
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "hello there"}},
        ),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        (
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": 4},
            },
        ),
        ("message_stop", {"type": "message_stop"}),
    ]
    return [f"event: {name}\ndata: {json.dumps(payload)}\n\n".encode() for name, payload in events]


async def test_streaming_raw_sse_allow_replays_frames():
    g = _make_guardrail(decisions=[_decision_response({"action": "allow"})])
    frames = _anthropic_sse_frames()
    out = await _collect(
        g.async_post_call_streaming_iterator_hook(
            user_api_key_dict=UserAPIKeyAuth(), response=_aiter(frames), request_data=_request_data()
        )
    )
    assert out == frames
    posted = _sent_payload(g)["response_body"]
    assert "choices" not in posted
    assert posted["type"] == "message"
    assert posted["id"] == "msg_abc"
    assert posted["content"] == [{"type": "text", "text": "hello there"}]
    assert posted["stop_reason"] == "end_turn"
    assert posted["usage"] == {"input_tokens": 9, "output_tokens": 4}


def _anthropic_thinking_sse_frames() -> list[bytes]:
    """A turn carrying everything the chat-completions shape has no field for."""
    events = [
        (
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": "msg_thinking",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-sonnet-5",
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {
                        "input_tokens": 9,
                        "output_tokens": 0,
                        "cache_creation_input_tokens": 120,
                        "cache_read_input_tokens": 400,
                    },
                },
            },
        ),
        (
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "thinking", "thinking": "", "signature": ""},
            },
        ),
        (
            "content_block_delta",
            {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "weighing it"}},
        ),
        (
            "content_block_delta",
            {"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": "sig-xyz"}},
        ),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        (
            "content_block_start",
            {"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}},
        ),
        (
            "content_block_delta",
            {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "hello there"}},
        ),
        ("content_block_stop", {"type": "content_block_stop", "index": 1}),
        (
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "stop_sequence", "stop_sequence": "END"},
                "usage": {"output_tokens": 4},
            },
        ),
        ("message_stop", {"type": "message_stop"}),
    ]
    return [f"event: {name}\ndata: {json.dumps(payload)}\n\n".encode() for name, payload in events]


async def test_a_messages_stream_keeps_what_the_chat_shape_has_no_field_for():
    """Folding the stream into a ModelResponse dropped thinking blocks, the signature,
    stop_sequence and the cache token split before the scan ever saw them."""
    g = _make_guardrail(decisions=[_decision_response({"action": "allow"})])
    await _collect(
        g.async_post_call_streaming_iterator_hook(
            user_api_key_dict=UserAPIKeyAuth(),
            response=_aiter(_anthropic_thinking_sse_frames()),
            request_data=_request_data(),
        )
    )
    posted = _sent_payload(g)["response_body"]
    assert posted["content"][0] == {"type": "thinking", "thinking": "weighing it", "signature": "sig-xyz"}
    assert posted["content"][1] == {"type": "text", "text": "hello there"}
    assert posted["stop_sequence"] == "END"
    assert posted["usage"]["cache_creation_input_tokens"] == 120
    assert posted["usage"]["cache_read_input_tokens"] == 400


async def test_a_messages_stream_rewrite_is_re_emitted_as_anthropic_frames():
    g = _make_guardrail(
        decisions=[
            _decision_response(
                {
                    "action": "modify_response",
                    "response_body": {"content": [{"type": "text", "text": "hello [REDACTED]"}]},
                }
            )
        ]
    )
    out = await _collect(
        g.async_post_call_streaming_iterator_hook(
            user_api_key_dict=UserAPIKeyAuth(),
            response=_aiter(_anthropic_sse_frames()),
            request_data=_request_data(),
        )
    )
    joined = b"".join(item for item in out if isinstance(item, bytes))
    assert b"hello [REDACTED]" in joined
    assert b"hello there" not in joined


async def test_a_chat_shaped_rewrite_of_a_messages_stream_fails_closed():
    """The overlay is a shallow merge, so "choices" would land beside the untouched "content"
    and the re-emitted stream would carry the very text the rewrite asked to redact."""
    from litellm.proxy.proxy_server import StreamingCallbackError

    g = _make_guardrail(decisions=[_redacting_modify_decision()])
    with pytest.raises(StreamingCallbackError, match="answered a /v1/messages stream"):
        await _collect(
            g.async_post_call_streaming_iterator_hook(
                user_api_key_dict=UserAPIKeyAuth(),
                response=_aiter(_anthropic_sse_frames()),
                request_data=_request_data(),
            )
        )


async def test_streaming_raw_sse_block_emits_anthropic_error_frame():
    g = _make_guardrail(decisions=[_decision_response({"action": "block", "message": "leaked secret"})])
    out = await _collect(
        g.async_post_call_streaming_iterator_hook(
            user_api_key_dict=UserAPIKeyAuth(),
            response=_aiter(_anthropic_sse_frames()),
            request_data=_request_data(),
        )
    )
    assert len(out) == 1
    assert isinstance(out[0], bytes)
    assert b"event: error" in out[0]
    assert b"leaked secret" in out[0]


def _unscannable_chunks() -> list[dict]:
    """A stream shape with no assembler and no wire format of its own.

    Text-completion streams arrive like this: plain dicts that are neither ModelResponseStream,
    raw Anthropic SSE frames, nor Responses API events, so there is no format to refuse them in.
    """
    return [
        {"text": "the secret ", "index": 0},
        {"text": "is sk-leak", "index": 0},
    ]


def _truncated_responses_chunks() -> list[dict]:
    """A /v1/responses stream that died before its terminal event.

    response.created also carries a body, but an empty one, so only a terminal event yields
    something scannable. Without one the turn cannot be scanned at all.
    """
    return [
        {"type": "response.output_text.delta", "delta": "the secret "},
        {"type": "response.output_text.delta", "delta": "is sk-leak"},
    ]


async def test_unscannable_stream_fails_closed_by_default():
    from litellm.proxy.proxy_server import StreamingCallbackError

    g = _make_guardrail(decisions=[])
    with pytest.raises(StreamingCallbackError, match="could not be assembled for scanning"):
        await _collect(
            g.async_post_call_streaming_iterator_hook(
                user_api_key_dict=UserAPIKeyAuth(),
                response=_aiter(_unscannable_chunks()),
                request_data=_request_data(),
            )
        )
    assert g.async_handler.post.await_count == 0


async def test_an_opaque_sse_stream_is_not_refused_in_anthropic_frames():
    """A Google :streamGenerateContent stream is raw SSE but not Anthropic, so an Anthropic
    error frame would refuse it in a format its client cannot parse."""
    from litellm.proxy.proxy_server import StreamingCallbackError

    google_frames = [b'data: {"candidates": [{"content": {"parts": [{"text": "hi"}]}}]}\n\n']
    g = _make_guardrail(decisions=[])
    with pytest.raises(StreamingCallbackError, match="could not be assembled for scanning"):
        await _collect(
            g.async_post_call_streaming_iterator_hook(
                user_api_key_dict=UserAPIKeyAuth(),
                response=_aiter(google_frames),
                request_data=_request_data(),
            )
        )
    assert g.async_handler.post.await_count == 0


async def test_modify_request_on_a_stream_is_ignored():
    """The request is long gone by the time a stream finishes, so a request rewrite has nothing
    to apply to and the original chunks are released."""
    g = _make_guardrail(
        decisions=[
            _decision_response(
                {"action": "modify_request", "request_body": {"messages": [{"role": "user", "content": "late"}]}}
            )
        ]
    )
    chunks = _stream_chunks()
    out = await _collect(
        g.async_post_call_streaming_iterator_hook(
            user_api_key_dict=UserAPIKeyAuth(), response=_aiter(chunks), request_data=_request_data()
        )
    )
    assert out == chunks


async def test_a_chat_stream_rewrite_that_fails_validation_blocks_the_stream():
    """A rewrite the assembler cannot turn back into a response must not reach the client."""
    from litellm.proxy.proxy_server import StreamingCallbackError

    g = _make_guardrail(
        decisions=[
            _decision_response({"action": "modify_response", "response_body": {"choices": [{"index": "not-an-int"}]}})
        ]
    )
    with pytest.raises((StreamingCallbackError, GuardrailRaisedException)):
        await _collect(
            g.async_post_call_streaming_iterator_hook(
                user_api_key_dict=UserAPIKeyAuth(),
                response=_aiter(_stream_chunks()),
                request_data=_request_data(),
            )
        )


async def test_a_messages_stream_rewrite_answered_with_choices_is_refused():
    """The Messages body carries its text in "content"; answering with "choices" means the service
    replied in the wrong body shape, which would silently blank the response."""
    from litellm.proxy.proxy_server import StreamingCallbackError

    g = _make_guardrail(decisions=[_decision_response({"action": "modify_response", "response_body": {"choices": []}})])
    with pytest.raises(StreamingCallbackError, match="carries its text in"):
        await _collect(
            g.async_post_call_streaming_iterator_hook(
                user_api_key_dict=UserAPIKeyAuth(),
                response=_aiter(_anthropic_sse_frames()),
                request_data=_request_data(),
            )
        )


async def test_an_opaque_sse_stream_posts_no_buffered_stream_beside_the_body():
    """There is no chunk shape to post for a surface with no assembler, so the stream fields stay
    off the payload instead of carrying a half-understood blob."""
    google_frames = [b'data: {"candidates": [{"content": {"parts": [{"text": "hi"}]}}]}\n\n']
    g = _make_guardrail(send_stream_chunks=True, unscannable_stream_fallback="fail_open", decisions=[])
    await _collect(
        g.async_post_call_streaming_iterator_hook(
            user_api_key_dict=UserAPIKeyAuth(),
            response=_aiter(google_frames),
            request_data=_request_data(),
        )
    )
    assert g.async_handler.post.await_count == 0


async def test_a_sampled_stream_blocks_on_the_final_decision():
    """The interim checks passed but the assembled response did not, so the terminal decision has
    to stop the stream even though chunks already went out."""
    from litellm.proxy.proxy_server import StreamingCallbackError

    g = _make_guardrail(
        streaming_buffer_until_moderated=False,
        streaming_end_of_stream_only=False,
        streaming_sampling_rate=100,
        decisions=[_decision_response({"action": "block", "message": "final says no"})],
    )
    with pytest.raises(StreamingCallbackError, match="final says no"):
        await _collect(
            g.async_post_call_streaming_iterator_hook(
                user_api_key_dict=UserAPIKeyAuth(),
                response=_aiter(_stream_chunks()),
                request_data=_request_data(),
            )
        )


async def test_a_sampled_stream_cannot_apply_a_late_modify_response():
    """Chunks were already delivered, so a rewrite arriving at end of stream is dropped with a
    warning rather than silently appearing to have been applied."""
    g = _make_guardrail(
        streaming_buffer_until_moderated=False,
        streaming_end_of_stream_only=False,
        streaming_sampling_rate=100,
        decisions=[_redacting_modify_decision()],
    )
    chunks = _stream_chunks()
    out = await _collect(
        g.async_post_call_streaming_iterator_hook(
            user_api_key_dict=UserAPIKeyAuth(), response=_aiter(chunks), request_data=_request_data()
        )
    )
    assert out == chunks


async def test_an_earlier_guardrails_refusal_is_passed_through_not_replaced():
    """post_call guardrails compose, so this hook can be handed the terminal error frames a
    preceding one emitted. Replacing them would hide the rejection the client is owed."""
    upstream_refusal = [b'event: error\ndata: {"type": "error", "error": {"message": "blocked by presidio"}}\n\n']
    g = _make_guardrail(decisions=[])
    out = await _collect(
        g.async_post_call_streaming_iterator_hook(
            user_api_key_dict=UserAPIKeyAuth(),
            response=_aiter(upstream_refusal),
            request_data=_request_data(),
        )
    )
    assert out == upstream_refusal
    assert g.async_handler.post.await_count == 0


def _responses_stream_chunks(text: str = "the secret is sk-leak") -> list[object]:
    """A complete /v1/responses stream, the shape that crashed stream_chunk_builder.

    The terminal response.completed event carries the finished body; every event before it
    carries either an empty body or a delta.
    """
    body = ResponsesAPIResponse(
        id="resp_encrypted_abc",
        created_at=1,
        model="claude-haiku-4-5",
        object="response",
        output=[
            {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
        parallel_tool_calls=True,
        tool_choice="auto",
        tools=[],
        top_p=1.0,
    )
    return [
        ResponseCreatedEvent(
            type=ResponsesAPIStreamEvents.RESPONSE_CREATED,
            response=ResponsesAPIResponse(
                id="resp_encrypted_abc",
                created_at=1,
                model="claude-haiku-4-5",
                object="response",
                output=[],
                parallel_tool_calls=True,
                tool_choice="auto",
                tools=[],
                top_p=1.0,
            ),
        ),
        OutputTextDeltaEvent(
            type=ResponsesAPIStreamEvents.OUTPUT_TEXT_DELTA,
            item_id="msg_1",
            output_index=0,
            content_index=0,
            delta=text,
        ),
        ResponseCompletedEvent(type=ResponsesAPIStreamEvents.RESPONSE_COMPLETED, response=body),
    ]


async def test_a_responses_stream_is_scanned_in_the_native_responses_body_shape():
    """stream_chunk_builder raised KeyError: 'model' on these events, refusing the whole stream.

    The body posted to the service must be the Responses shape its non-streaming half already
    sends, not a chat-completions body with a fabricated "choices".
    """
    g = _make_guardrail(decisions=[_decision_response({"action": "allow"})])
    chunks = _responses_stream_chunks()
    out = await _collect(
        g.async_post_call_streaming_iterator_hook(
            user_api_key_dict=UserAPIKeyAuth(), response=_aiter(chunks), request_data=_request_data()
        )
    )
    assert out == chunks
    assert g.async_handler.post.await_count == 1
    posted = g.async_handler.post.await_args.kwargs["json"]["response_body"]
    assert "choices" not in posted
    assert posted["object"] == "response"
    assert posted["id"] == "resp_encrypted_abc"
    assert posted["output"][0]["content"][0]["text"] == "the secret is sk-leak"


async def test_a_responses_stream_block_arrives_as_a_responses_error_event():
    """A raised exception reaches the client as the proxy's {"error": ...} blob, which has no
    top-level "type" and so is not a Responses event at all."""
    g = _make_guardrail(decisions=[_decision_response({"action": "block", "message": "leaked secret"})])
    out = await _collect(
        g.async_post_call_streaming_iterator_hook(
            user_api_key_dict=UserAPIKeyAuth(),
            response=_aiter(_responses_stream_chunks()),
            request_data=_request_data(),
        )
    )
    assert [getattr(item, "type", None) for item in out] == [ResponsesAPIStreamEvents.ERROR]
    assert out[0].error.message == "leaked secret"


async def test_a_responses_stream_rewrite_arrives_as_responses_events():
    g = _make_guardrail(
        decisions=[
            _decision_response(
                {
                    "action": "modify_response",
                    "response_body": {
                        "output": [
                            {
                                "id": "msg_1",
                                "type": "message",
                                "role": "assistant",
                                "status": "completed",
                                "content": [
                                    {"type": "output_text", "text": "the secret is [REDACTED]", "annotations": []}
                                ],
                            }
                        ]
                    },
                }
            )
        ]
    )
    out = await _collect(
        g.async_post_call_streaming_iterator_hook(
            user_api_key_dict=UserAPIKeyAuth(),
            response=_aiter(_responses_stream_chunks()),
            request_data=_request_data(),
        )
    )
    emitted = "".join(item.delta for item in out if getattr(item, "type", None) == "response.output_text.delta")
    assert emitted == "the secret is [REDACTED]"
    assert "sk-leak" not in emitted
    completed = [item for item in out if getattr(item, "type", None) == "response.completed"]
    assert len(completed) == 1
    assert completed[0].response.id == "resp_encrypted_abc"


async def test_a_responses_rewrite_cannot_rename_the_encrypted_response_id():
    """litellm hands out an encrypted response id and the client chains the next turn off it
    with previous_response_id, so a rewrite must not be able to replace it."""
    g = _make_guardrail(
        decisions=[
            _decision_response(
                {"action": "modify_response", "response_body": {"id": "resp_attacker_supplied"}},
            )
        ]
    )
    out = await _collect(
        g.async_post_call_streaming_iterator_hook(
            user_api_key_dict=UserAPIKeyAuth(),
            response=_aiter(_responses_stream_chunks()),
            request_data=_request_data(),
        )
    )
    ids = {getattr(item, "response", None) and item.response.id for item in out}
    assert "resp_attacker_supplied" not in ids
    assert "resp_encrypted_abc" in ids


async def test_a_truncated_responses_stream_is_refused_as_a_responses_error_event():
    g = _make_guardrail(decisions=[])
    out = await _collect(
        g.async_post_call_streaming_iterator_hook(
            user_api_key_dict=UserAPIKeyAuth(),
            response=_aiter(_truncated_responses_chunks()),
            request_data=_request_data(),
        )
    )
    assert [getattr(item, "type", None) for item in out] == [ResponsesAPIStreamEvents.ERROR]
    assert "could not be assembled for scanning" in out[0].error.message
    assert g.async_handler.post.await_count == 0


async def test_a_responses_error_event_continues_the_streams_sequence_numbering():
    """An error event restarting at zero after N events would arrive out of order."""
    chunks = _responses_stream_chunks()
    chunks[-1].__dict__["sequence_number"] = 11
    g = _make_guardrail(decisions=[_decision_response({"action": "block", "message": "leaked secret"})])
    out = await _collect(
        g.async_post_call_streaming_iterator_hook(
            user_api_key_dict=UserAPIKeyAuth(), response=_aiter(chunks), request_data=_request_data()
        )
    )
    assert out[0].sequence_number == 12


async def test_delta_text_missing_from_the_terminal_body_is_posted_beside_it():
    """Reasoning summaries reach the client through deltas some providers never repeat in the
    finished body. A scan of that body alone would miss them, so they ride along in their own
    field and the body itself stays the shape the non-streaming route posts."""
    chunks = _responses_stream_chunks()
    chunks.insert(
        2,
        ReasoningSummaryTextDeltaEvent(
            type=ResponsesAPIStreamEvents.REASONING_SUMMARY_TEXT_DELTA,
            item_id="rs_1",
            output_index=0,
            summary_index=0,
            delta="thinking about ",
        ),
    )
    chunks.insert(
        3,
        ReasoningSummaryTextDeltaEvent(
            type=ResponsesAPIStreamEvents.REASONING_SUMMARY_TEXT_DELTA,
            item_id="rs_1",
            output_index=0,
            summary_index=0,
            delta="sk-other-leak",
        ),
    )
    g = _make_guardrail(decisions=[_decision_response({"action": "allow"})])
    await _collect(
        g.async_post_call_streaming_iterator_hook(
            user_api_key_dict=UserAPIKeyAuth(), response=_aiter(chunks), request_data=_request_data()
        )
    )
    posted = _sent_payload(g)
    assert posted["streamed_deltas_not_in_body"] == ["thinking about sk-other-leak"]
    assert "choices" not in posted["response_body"]
    assert posted["response_body"]["output"][0]["content"][0]["text"] == "the secret is sk-leak"


async def test_the_deltas_field_is_omitted_when_the_body_already_carries_every_delta():
    """Only text the body does not account for goes beside it; a clean turn posts no field at all,
    so the service sees the same payload shape it did before the field existed."""
    g = _make_guardrail(decisions=[_decision_response({"action": "allow"})])
    await _collect(
        g.async_post_call_streaming_iterator_hook(
            user_api_key_dict=UserAPIKeyAuth(),
            response=_aiter(_responses_stream_chunks()),
            request_data=_request_data(),
        )
    )
    assert "streamed_deltas_not_in_body" not in _sent_payload(g)


async def test_the_deltas_field_never_appears_on_a_chat_or_messages_stream():
    """Only a /v1/responses turn can leave delta text out of its body; the other surfaces fold
    every delta into what they post."""
    for chunks in (_stream_chunks(), _anthropic_sse_frames()):
        g = _make_guardrail(decisions=[_decision_response({"action": "allow"})])
        await _collect(
            g.async_post_call_streaming_iterator_hook(
                user_api_key_dict=UserAPIKeyAuth(), response=_aiter(chunks), request_data=_request_data()
            )
        )
        assert "streamed_deltas_not_in_body" not in _sent_payload(g)


async def test_a_responses_stream_is_posted_as_chunks_beside_the_body():
    """The body is what the proxy folded; the chunks are what the client received. Both go, so the
    service can fold on its own side, and a sequence number the bridge stamps outside the model
    survives the trip."""
    chunks = _responses_stream_chunks()
    chunks[1].__dict__["sequence_number"] = 4
    g = _make_guardrail(send_stream_chunks=True, decisions=[_decision_response({"action": "allow"})])
    await _collect(
        g.async_post_call_streaming_iterator_hook(
            user_api_key_dict=UserAPIKeyAuth(), response=_aiter(chunks), request_data=_request_data()
        )
    )
    posted = _sent_payload(g)
    assert [c["type"] for c in posted["response_chunks"]] == [
        "response.created",
        "response.output_text.delta",
        "response.completed",
    ]
    assert posted["response_chunks"][1]["delta"] == "the secret is sk-leak"
    assert posted["response_chunks"][1]["sequence_number"] == 4
    assert "response_sse" not in posted
    assert posted["response_body"]["object"] == "response"


async def test_a_messages_stream_is_posted_as_the_raw_sse_text_beside_the_body():
    frames = _anthropic_sse_frames()
    g = _make_guardrail(send_stream_chunks=True, decisions=[_decision_response({"action": "allow"})])
    await _collect(
        g.async_post_call_streaming_iterator_hook(
            user_api_key_dict=UserAPIKeyAuth(), response=_aiter(frames), request_data=_request_data()
        )
    )
    posted = _sent_payload(g)
    assert posted["response_sse"] == b"".join(frames).decode()
    assert "response_chunks" not in posted
    assert posted["response_body"]["type"] == "message"


async def test_a_chat_stream_is_posted_as_chunks_beside_the_body():
    chunks = _stream_chunks()
    g = _make_guardrail(send_stream_chunks=True, decisions=[_decision_response({"action": "allow"})])
    await _collect(
        g.async_post_call_streaming_iterator_hook(
            user_api_key_dict=UserAPIKeyAuth(), response=_aiter(chunks), request_data=_request_data()
        )
    )
    posted = _sent_payload(g)
    assert len(posted["response_chunks"]) == len(chunks)
    assert posted["response_chunks"][0]["choices"][0]["delta"]["content"] == "the secret "
    assert "response_sse" not in posted
    assert posted["response_body"]["choices"][0]["message"]["content"] == "the secret is sk-leak"


async def test_the_stream_is_not_posted_when_send_stream_chunks_is_off():
    g = _make_guardrail(decisions=[_decision_response({"action": "allow"})], send_stream_chunks=False)
    await _collect(
        g.async_post_call_streaming_iterator_hook(
            user_api_key_dict=UserAPIKeyAuth(),
            response=_aiter(_responses_stream_chunks()),
            request_data=_request_data(),
        )
    )
    posted = _sent_payload(g)
    assert "response_chunks" not in posted
    assert "response_sse" not in posted
    assert posted["response_body"]["object"] == "response"


async def test_only_the_final_scan_carries_the_stream_not_the_interim_ones():
    """An interim scan sees a partial stream with no terminal event, which folds to nothing on the
    service side, so the chunks ride only on the end-of-stream post."""
    g = _make_guardrail(
        send_stream_chunks=True,
        streaming_buffer_until_moderated=False,
        streaming_end_of_stream_only=False,
        streaming_sampling_rate=1,
        decisions=[_decision_response({"action": "allow"})] * 6,
    )
    chunks = _stream_chunks()
    await _collect(
        g.async_post_call_streaming_iterator_hook(
            user_api_key_dict=UserAPIKeyAuth(), response=_aiter(chunks), request_data=_request_data()
        )
    )
    posts = [call.kwargs["json"] for call in g.async_handler.post.call_args_list]
    assert len(posts) == len(chunks) + 1
    assert all("response_chunks" not in post for post in posts[:-1])
    assert len(posts[-1]["response_chunks"]) == len(chunks)


def test_the_initializer_sends_stream_chunks_only_on_an_explicit_true():
    """An operator who never sets send_stream_chunks must not have the whole buffered stream
    posted out: the config schema documents the default as False, so an omitted value is off."""
    omitted = initialize_guardrail(
        LitellmParams(guardrail="thirdlaw", mode="post_call", api_base=_API_BASE, api_key="k"),
        {"guardrail_name": "thirdlaw-omitted"},
    )
    explicit_off = initialize_guardrail(
        LitellmParams(
            guardrail="thirdlaw", mode="post_call", api_base=_API_BASE, api_key="k", send_stream_chunks=False
        ),
        {"guardrail_name": "thirdlaw-off"},
    )
    explicit_on = initialize_guardrail(
        LitellmParams(guardrail="thirdlaw", mode="post_call", api_base=_API_BASE, api_key="k", send_stream_chunks=True),
        {"guardrail_name": "thirdlaw-on"},
    )
    assert omitted.send_stream_chunks is False
    assert explicit_off.send_stream_chunks is False
    assert explicit_on.send_stream_chunks is True


@pytest.mark.parametrize("typo", ["fail_close", "failopen", "FAIL_OPEN", ""])
async def test_unscannable_stream_fails_closed_on_a_mistyped_fallback(typo):
    """Literal is not enforced at runtime, so anything but the exact opt-in must block."""
    from litellm.proxy.proxy_server import StreamingCallbackError

    g = _make_guardrail(decisions=[], unscannable_stream_fallback=typo)
    with pytest.raises(StreamingCallbackError, match="could not be assembled for scanning"):
        await _collect(
            g.async_post_call_streaming_iterator_hook(
                user_api_key_dict=UserAPIKeyAuth(),
                response=_aiter(_unscannable_chunks()),
                request_data=_request_data(),
            )
        )


async def test_unscannable_stream_passes_through_when_opted_into_fail_open():
    chunks = _unscannable_chunks()
    g = _make_guardrail(decisions=[], unscannable_stream_fallback="fail_open")
    out = await _collect(
        g.async_post_call_streaming_iterator_hook(
            user_api_key_dict=UserAPIKeyAuth(), response=_aiter(chunks), request_data=_request_data()
        )
    )
    assert out == chunks
    assert g.async_handler.post.await_count == 0


@pytest.mark.parametrize(
    ("service_status", "expected_status"),
    [(200, 400), (204, 400), (302, 400), (403, 403), (451, 451), (503, 503), (None, 400)],
)
async def test_block_never_travels_as_a_success_status(service_status, expected_status):
    """A block is a refusal, so a success status from the service must not reach the caller."""
    body: JsonDict = {"action": "block", "message": "nope"}
    if service_status is not None:
        body["response_status"] = service_status
    g = _make_guardrail(decisions=[_decision_response(body)])
    with pytest.raises(GuardrailRaisedException) as excinfo:
        await g.async_pre_call_hook(
            user_api_key_dict=UserAPIKeyAuth(),
            cache=DualCache(),
            data=_request_data(),
            call_type="completion",
        )
    assert excinfo.value.status_code == expected_status
