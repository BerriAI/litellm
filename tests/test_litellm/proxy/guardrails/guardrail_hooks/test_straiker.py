import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from litellm.exceptions import GuardrailRaisedException, ModifyResponseException
from litellm.proxy.guardrails.guardrail_hooks.straiker import initialize_guardrail
from litellm.proxy.guardrails.guardrail_hooks.straiker.straiker import (
    StraikerGuardrail,
    _response_finish_reason,
)
from litellm.proxy.guardrails.guardrail_registry import (
    guardrail_class_registry,
    guardrail_initializer_registry,
)
from litellm.types.proxy.guardrails.guardrail_hooks.straiker import (
    StraikerGuardrailConfigModel,
    StraikerGuardrailConfigModelOptionalParams,
)
from litellm.types.utils import (
    ChatCompletionMessageToolCall,
    Choices,
    Function,
    Message,
    ModelResponse,
    Usage,
)


def _mock_response(action: str, turn_id: str = "turn-1", schema_version: str = "1", **extra) -> MagicMock:
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = 200
    resp.json.return_value = {
        "schema_version": schema_version,
        "action": action,
        "turn_id": turn_id,
        **extra,
    }
    resp.text = ""
    return resp


def _make_guardrail(**overrides) -> StraikerGuardrail:
    defaults = dict(
        api_key="test-key",
        api_base="https://test.straiker.ai",
        max_retries=0,
        guardrail_name="straiker",
        event_hook="pre_call",
        async_handler=MagicMock(spec=httpx.AsyncClient),
    )
    defaults.update(overrides)
    g = StraikerGuardrail(**defaults)
    g.async_handler.post = AsyncMock()
    return g


def _logging_obj() -> MagicMock:
    obj = MagicMock()
    obj.litellm_call_id = "call-123"
    obj.litellm_trace_id = "trace-456"
    obj.call_type = "acompletion"
    return obj


def _posted_payload(g: StraikerGuardrail) -> dict:
    return json.loads(g.async_handler.post.call_args.kwargs["content"])


def test_registry_membership():
    assert "straiker" in guardrail_initializer_registry
    assert guardrail_class_registry["straiker"] is StraikerGuardrail


def test_config_model_wiring():
    assert StraikerGuardrailConfigModel.ui_friendly_name() == "Straiker"
    assert StraikerGuardrail.get_config_model() is StraikerGuardrailConfigModel
    fields = StraikerGuardrailConfigModel.model_fields
    assert "api_key" in fields
    assert "api_base" in fields
    assert "default_app" in fields
    assert "source" not in fields
    assert "optional_params" in fields
    assert "timeout" not in fields
    assert "verbose" not in fields


def test_init_rejects_empty_api_key():
    with pytest.raises(ValueError):
        StraikerGuardrail(api_key="")


def test_init_rejects_invalid_fallback():
    with pytest.raises(ValueError):
        StraikerGuardrail(api_key="k", unreachable_fallback="nope")


def test_supported_hooks_limited_to_pre_and_post():
    from litellm.types.guardrails import GuardrailEventHooks

    assert StraikerGuardrail.get_supported_event_hooks() == [
        GuardrailEventHooks.pre_call,
        GuardrailEventHooks.post_call,
    ]


def test_during_call_mode_rejected_at_init():
    with pytest.raises(ValueError):
        StraikerGuardrail(api_key="k", event_hook="during_call")


def test_streaming_attrs_hardcoded_to_buffered():
    g = _make_guardrail()
    assert g.streaming_buffer_until_moderated is True
    assert g.streaming_end_of_stream_only is True


def test_streaming_flags_not_configurable():
    fields = StraikerGuardrailConfigModelOptionalParams.model_fields
    assert "streaming_buffer_until_moderated" not in fields
    assert "streaming_end_of_stream_only" not in fields
    assert "streaming_sampling_rate" not in fields


def test_initializer_builds_working_callback():
    from litellm.types.guardrails import LitellmParams

    params = LitellmParams(guardrail="straiker", mode="pre_call", api_key="abc", api_base="https://x.straiker.ai")
    callback = initialize_guardrail(params, {"guardrail_name": "straiker"})
    assert isinstance(callback, StraikerGuardrail)
    assert callback.api_base == "https://x.straiker.ai"


def test_initializer_maps_default_app_to_source():
    from litellm.types.guardrails import LitellmParams

    params = LitellmParams(
        guardrail="straiker",
        mode="pre_call",
        api_key="abc",
        default_app="My App",
    )
    callback = initialize_guardrail(params, {"guardrail_name": "straiker"})
    assert callback.source == "My App"


def test_initializer_reads_optional_params_flattened_like_ui():
    from litellm.types.guardrails import LitellmParams

    params = LitellmParams(
        guardrail="straiker",
        mode="pre_call",
        api_key="abc",
        api_base="https://x.straiker.ai",
        timeout=9.5,
        verbose=True,
        unreachable_fallback="fail_open",
    )
    callback = initialize_guardrail(params, {"guardrail_name": "straiker"})
    assert isinstance(callback, StraikerGuardrail)
    assert callback.timeout == 9.5
    assert callback.verbose is True
    assert callback.unreachable_fallback == "fail_open"
    assert callback.api_base == "https://x.straiker.ai"


def test_initializer_reads_nested_optional_params():
    from types import SimpleNamespace

    from litellm.types.guardrails import LitellmParams

    params = LitellmParams.model_construct(
        guardrail="straiker",
        mode="pre_call",
        api_key="abc",
        api_base="https://x.straiker.ai",
        optional_params=SimpleNamespace(
            timeout=7.25,
            verbose=True,
            unreachable_fallback="fail_open",
        ),
    )
    callback = initialize_guardrail(params, {"guardrail_name": "straiker"})
    assert isinstance(callback, StraikerGuardrail)
    assert callback.timeout == 7.25
    assert callback.verbose is True
    assert callback.unreachable_fallback == "fail_open"


def test_initializer_reads_dict_optional_params():
    from litellm.types.guardrails import LitellmParams

    params = LitellmParams.model_construct(
        guardrail="straiker",
        mode="pre_call",
        api_key="abc",
        api_base="https://x.straiker.ai",
        optional_params={"timeout": 7.25, "verbose": True, "unreachable_fallback": "fail_open"},
    )
    callback = initialize_guardrail(params, {"guardrail_name": "straiker"})
    assert isinstance(callback, StraikerGuardrail)
    assert callback.timeout == 7.25
    assert callback.verbose is True
    assert callback.unreachable_fallback == "fail_open"


@pytest.mark.asyncio
async def test_request_envelope_transport_and_shape():
    g = _make_guardrail()
    g.async_handler.post.return_value = _mock_response("NONE")
    inputs = {"texts": ["hello world"], "model": "gpt-4o-mini"}
    request_data = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "hello world"}],
        "metadata": {"user_api_key_alias": "team-key", "agent_id": "chatbot-app", "app_name": "Chatbot"},
    }

    out = await g.apply_guardrail(
        inputs=inputs, request_data=request_data, input_type="request", logging_obj=_logging_obj()
    )

    assert out is inputs
    url = g.async_handler.post.call_args.args[0]
    assert url == "https://test.straiker.ai/api/v1/detect/webhook"
    headers = g.async_handler.post.call_args.kwargs["headers"]
    assert headers["X-Straiker-Webhook-Format"] == "litellm"
    assert headers["Authorization"] == "Bearer test-key"

    payload = _posted_payload(g)
    assert payload["schema_version"] == "1"
    assert payload["event"]["type"] == "pre_call"
    assert payload["event"]["id"] == "call-123:request"
    assert payload["request"]["texts"] == ["hello world"]
    assert payload["context"]["litellm_call_id"] == "call-123"
    assert payload["identity"]["litellm_key"] == "team-key"
    assert payload["application"] == {"source": "chatbot-app", "name": "Chatbot"}
    assert "session_id" not in payload["application"]
    assert "user_name" not in payload["application"]
    assert "user_role" not in payload["application"]
    assert "response" not in payload
    assert "metadata" not in payload


@pytest.mark.asyncio
async def test_request_envelope_ignores_unsupported_opaque_items():
    g = _make_guardrail()
    g.async_handler.post.return_value = _mock_response("NONE")

    await g.apply_guardrail(
        inputs={
            "texts": ["hello"],
            "tools": [
                object(),
                {
                    "type": "function",
                    "function": {"name": "get_weather", "parameters": {"type": "object"}},
                },
            ],
        },
        request_data={"model": "m", "messages": [{"role": "user", "content": "hello"}]},
        input_type="request",
        logging_obj=_logging_obj(),
    )

    assert _posted_payload(g)["request"]["tools"] == [
        {
            "type": "function",
            "function": {"name": "get_weather", "parameters": {"type": "object"}},
        }
    ]


@pytest.mark.asyncio
async def test_webhook_metadata_session_id_and_opaque_passthrough():
    g = _make_guardrail()
    g.async_handler.post.return_value = _mock_response("NONE")
    await g.apply_guardrail(
        inputs={"texts": ["x"]},
        request_data={
            "model": "m",
            "litellm_session_id": "sess-from-litellm",
            "metadata": {
                "agent_id": "chatbot-app",
                "app_name": "Chatbot",
                "user_api_key_alias": "team-key",
                "custom_tag": "experiment-7",
                "client_ip": "10.0.0.1",
            },
        },
        input_type="request",
        logging_obj=_logging_obj(),
    )
    payload = _posted_payload(g)
    assert payload["application"] == {"source": "chatbot-app", "name": "Chatbot"}
    assert payload["identity"]["litellm_key"] == "team-key"
    assert payload["context"]["session_id"] == "sess-from-litellm"
    assert "session_id" not in payload["metadata"]
    assert payload["metadata"] == {
        "custom_tag": "experiment-7",
        "client_ip": "10.0.0.1",
    }


@pytest.mark.asyncio
async def test_webhook_metadata_never_forwards_proxy_internal_keys():
    g = _make_guardrail()
    g.async_handler.post.return_value = _mock_response("NONE")
    await g.apply_guardrail(
        inputs={"texts": ["x"]},
        request_data={
            "model": "m",
            "metadata": {
                "custom_tag": "experiment-7",
                "user_api_key": "sk-hashed-secret",
                "user_api_end_user_max_budget": 12.5,
            },
        },
        input_type="request",
        logging_obj=_logging_obj(),
    )
    assert _posted_payload(g)["metadata"] == {"custom_tag": "experiment-7"}


@pytest.mark.asyncio
async def test_default_metadata_injected_and_config_wins_on_clash():
    g = _make_guardrail(metadata={"tenant": "acme", "custom_tag": "config-value"})
    g.async_handler.post.return_value = _mock_response("NONE")
    await g.apply_guardrail(
        inputs={"texts": ["x"]},
        request_data={
            "model": "m",
            "metadata": {"custom_tag": "request-value", "client_ip": "10.0.0.1"},
        },
        input_type="request",
        logging_obj=_logging_obj(),
    )
    assert _posted_payload(g)["metadata"] == {
        "client_ip": "10.0.0.1",
        "custom_tag": "config-value",
        "tenant": "acme",
    }


@pytest.mark.asyncio
async def test_default_metadata_present_without_request_metadata():
    g = _make_guardrail(metadata={"tenant": "acme"})
    g.async_handler.post.return_value = _mock_response("NONE")
    await g.apply_guardrail(
        inputs={"texts": ["x"]},
        request_data={"model": "m"},
        input_type="request",
        logging_obj=_logging_obj(),
    )
    assert _posted_payload(g)["metadata"] == {"tenant": "acme"}


@pytest.mark.asyncio
async def test_context_session_id_from_request_metadata():
    g = _make_guardrail()
    g.async_handler.post.return_value = _mock_response("NONE")
    await g.apply_guardrail(
        inputs={"texts": ["x"]},
        request_data={"model": "m", "metadata": {"session_id": "sess-meta"}},
        input_type="request",
        logging_obj=_logging_obj(),
    )
    payload = _posted_payload(g)
    assert payload["context"]["session_id"] == "sess-meta"
    assert "metadata" not in payload


@pytest.mark.asyncio
async def test_context_mode_from_string_event_hook():
    g = _make_guardrail(event_hook="pre_call")
    g.async_handler.post.return_value = _mock_response("NONE")
    await g.apply_guardrail(
        inputs={"texts": ["x"]},
        request_data={"model": "m"},
        input_type="request",
        logging_obj=_logging_obj(),
    )
    assert _posted_payload(g)["context"]["mode"] == ["pre_call"]


@pytest.mark.asyncio
async def test_context_mode_from_list_event_hook():
    from litellm.types.guardrails import GuardrailEventHooks

    g = _make_guardrail(event_hook=[GuardrailEventHooks.pre_call, GuardrailEventHooks.post_call])
    g.async_handler.post.return_value = _mock_response("NONE")
    await g.apply_guardrail(
        inputs={"texts": ["x"]},
        request_data={"model": "m"},
        input_type="request",
        logging_obj=_logging_obj(),
    )
    assert _posted_payload(g)["context"]["mode"] == ["pre_call", "post_call"]


@pytest.mark.asyncio
async def test_context_mode_from_tagged_mode_is_flattened_and_deduped():
    from litellm.types.guardrails import Mode

    g = _make_guardrail(
        event_hook=Mode(tags={"team-a": "pre_call", "team-b": ["post_call", "pre_call"]}, default="post_call")
    )
    g.async_handler.post.return_value = _mock_response("NONE")
    await g.apply_guardrail(
        inputs={"texts": ["x"]},
        request_data={"model": "m"},
        input_type="request",
        logging_obj=_logging_obj(),
    )
    assert _posted_payload(g)["context"]["mode"] == ["post_call", "pre_call"]


@pytest.mark.asyncio
async def test_context_mode_omitted_when_event_hook_absent():
    g = _make_guardrail(event_hook=None)
    g.async_handler.post.return_value = _mock_response("NONE")
    await g.apply_guardrail(
        inputs={"texts": ["x"]},
        request_data={"model": "m"},
        input_type="request",
        logging_obj=_logging_obj(),
    )
    assert "mode" not in _posted_payload(g)["context"]


@pytest.mark.asyncio
async def test_identity_key_and_team_coalesce_alias_over_id():
    g = _make_guardrail()
    g.async_handler.post.return_value = _mock_response("NONE")
    await g.apply_guardrail(
        inputs={"texts": ["x"]},
        request_data={
            "model": "m",
            "metadata": {
                "user_api_key_alias": "prod-key",
                "user_api_key_hash": "hash-abc",
                "user_api_key_team_alias": "growth",
                "user_api_key_team_id": "team-9",
            },
        },
        input_type="request",
        logging_obj=_logging_obj(),
    )
    identity = _posted_payload(g)["identity"]
    assert identity["litellm_key"] == "prod-key"
    assert identity["litellm_team"] == "growth"
    assert "key" not in identity
    assert "team" not in identity


@pytest.mark.asyncio
async def test_identity_key_and_team_fall_back_to_hash_and_id():
    g = _make_guardrail()
    g.async_handler.post.return_value = _mock_response("NONE")
    await g.apply_guardrail(
        inputs={"texts": ["x"]},
        request_data={
            "model": "m",
            "metadata": {
                "user_api_key_hash": "hash-abc",
                "user_api_key_team_id": "team-9",
            },
        },
        input_type="request",
        logging_obj=_logging_obj(),
    )
    identity = _posted_payload(g)["identity"]
    assert identity["litellm_key"] == "hash-abc"
    assert identity["litellm_team"] == "team-9"


@pytest.mark.asyncio
async def test_identity_end_user_from_resolved_metadata():
    g = _make_guardrail()
    g.async_handler.post.return_value = _mock_response("NONE")

    await g.apply_guardrail(
        inputs={"texts": ["x"]},
        request_data={
            "model": "m",
            "metadata": {
                "user_api_key_end_user_id": "eu-meta",
                "user_api_key_user_id": "default_user_id",
            },
            "user": "eu-body",
        },
        input_type="request",
        logging_obj=_logging_obj(),
    )
    identity = _posted_payload(g)["identity"]
    assert identity["end_user_id"] == "eu-meta"
    assert identity["litellm_user_id"] == "default_user_id"
    assert _posted_payload(g)["application"] == {"source": g.source}


@pytest.mark.asyncio
async def test_identity_end_user_absent_without_resolved_metadata():
    g = _make_guardrail()
    g.async_handler.post.return_value = _mock_response("NONE")
    await g.apply_guardrail(
        inputs={"texts": ["x"]},
        request_data={"model": "m", "user": "eu-body", "metadata": {"user_api_key_user_id": "default_user_id"}},
        input_type="request",
        logging_obj=_logging_obj(),
    )
    assert "end_user_id" not in _posted_payload(g)["identity"]


@pytest.mark.asyncio
async def test_application_source_from_agent_id():
    g = _make_guardrail(source="litellm")
    g.async_handler.post.return_value = _mock_response("NONE")
    await g.apply_guardrail(
        inputs={"texts": ["x"]},
        request_data={"model": "m", "metadata": {"agent_id": "analytics-app", "app_name": "Analytics"}},
        input_type="request",
        logging_obj=_logging_obj(),
    )
    assert _posted_payload(g)["application"] == {"source": "analytics-app", "name": "Analytics"}


@pytest.mark.asyncio
async def test_request_block_raises_guardrail_exception_with_reason():
    g = _make_guardrail()
    g.async_handler.post.return_value = _mock_response("BLOCKED", blocked_reason="prompt injection")
    with pytest.raises(GuardrailRaisedException) as exc:
        await g.apply_guardrail(
            inputs={"texts": ["attack"]}, request_data={"model": "m"}, input_type="request", logging_obj=_logging_obj()
        )
    assert "prompt injection" in str(exc.value)


@pytest.mark.asyncio
async def test_guardrail_intervened_writes_back_modified_text_only():
    g = _make_guardrail()
    g.async_handler.post.return_value = _mock_response("GUARDRAIL_INTERVENED", texts=["[redacted]"])
    inputs = {"texts": ["my ssn is 123"], "images": ["img-a"]}
    out = await g.apply_guardrail(
        inputs=inputs, request_data={"model": "m"}, input_type="request", logging_obj=_logging_obj()
    )
    assert out["texts"] == ["[redacted]"]
    assert out["images"] == ["img-a"]


@pytest.mark.asyncio
async def test_streamed_response_intervention_converts_to_block():
    g = _make_guardrail()
    g.async_handler.post.return_value = _mock_response("GUARDRAIL_INTERVENED", texts=["[redacted]"])
    response = ModelResponse(
        choices=[Choices(finish_reason="stop", index=0, message=Message(content="secret", role="assistant"))],
        model="gpt-4o-mini",
    )
    request_data = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "p"}],
        "stream": True,
        "response": response,
    }
    with pytest.raises(ModifyResponseException):
        await g.apply_guardrail(
            inputs={"texts": ["secret"], "model": "gpt-4o-mini"},
            request_data=request_data,
            input_type="response",
            logging_obj=_logging_obj(),
        )


@pytest.mark.asyncio
async def test_non_streamed_response_intervention_redacts():
    g = _make_guardrail()
    g.async_handler.post.return_value = _mock_response("GUARDRAIL_INTERVENED", texts=["[redacted]"])
    response = ModelResponse(
        choices=[Choices(finish_reason="stop", index=0, message=Message(content="secret", role="assistant"))],
        model="gpt-4o-mini",
    )
    request_data = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "p"}],
        "response": response,
    }
    out = await g.apply_guardrail(
        inputs={"texts": ["secret"], "model": "gpt-4o-mini"},
        request_data=request_data,
        input_type="response",
        logging_obj=_logging_obj(),
    )
    assert out["texts"] == ["[redacted]"]


@pytest.mark.asyncio
async def test_guardrail_intervened_without_texts_blocks():
    g = _make_guardrail()
    g.async_handler.post.return_value = _mock_response("GUARDRAIL_INTERVENED")
    with pytest.raises(GuardrailRaisedException):
        await g.apply_guardrail(
            inputs={"texts": ["my ssn is 123"]},
            request_data={"model": "m"},
            input_type="request",
            logging_obj=_logging_obj(),
        )


@pytest.mark.asyncio
async def test_streamed_via_proxy_server_request_body_converts_to_block():
    g = _make_guardrail()
    g.async_handler.post.return_value = _mock_response("GUARDRAIL_INTERVENED", texts=["[redacted]"])
    response = ModelResponse(
        choices=[Choices(finish_reason="stop", index=0, message=Message(content="secret", role="assistant"))],
        model="gpt-4o-mini",
    )
    request_data = {
        "model": "gpt-4o-mini",
        "proxy_server_request": {"body": {"stream": True}},
        "response": response,
    }
    with pytest.raises(ModifyResponseException):
        await g.apply_guardrail(
            inputs={"texts": ["secret"], "model": "gpt-4o-mini"},
            request_data=request_data,
            input_type="response",
            logging_obj=_logging_obj(),
        )


@pytest.mark.asyncio
async def test_response_envelope_and_block_replaces_response():
    g = _make_guardrail(verbose=True)
    g.async_handler.post.return_value = _mock_response("BLOCKED")
    response = ModelResponse(
        choices=[Choices(finish_reason="stop", index=0, message=Message(content="secret", role="assistant"))],
        model="gpt-4o-mini",
    )
    request_data = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "original prompt"}],
        "stream": True,
        "response": response,
    }
    with pytest.raises(ModifyResponseException) as exc:
        await g.apply_guardrail(
            inputs={"texts": ["secret"], "model": "gpt-4o-mini"},
            request_data=request_data,
            input_type="response",
            logging_obj=_logging_obj(),
        )

    assert exc.value.original_response is response
    payload = _posted_payload(g)
    assert payload["event"]["type"] == "post_call"
    assert payload["event"]["stream"]["phase"] == "assembled"
    assert payload["response"]["texts"] == ["secret"]
    assert payload["response"]["finish_reason"] == "stop"
    assert payload["request"]["structured_messages"] == [{"role": "user", "content": "original prompt"}]


@pytest.mark.asyncio
async def test_post_call_resolves_request_from_responses_input_when_messages_absent():
    g = _make_guardrail()
    g.async_handler.post.return_value = _mock_response("NONE")
    response = ModelResponse(
        choices=[Choices(finish_reason="stop", index=0, message=Message(content="answer", role="assistant"))],
        model="gpt-4o-mini",
    )
    request_data = {
        "model": "gpt-4o-mini",
        "input": "responses-surface prompt",
        "response": response,
    }

    await g.apply_guardrail(
        inputs={"texts": ["answer"], "model": "gpt-4o-mini"},
        request_data=request_data,
        input_type="response",
        logging_obj=_logging_obj(),
    )

    payload = _posted_payload(g)
    assert payload["event"]["type"] == "post_call"
    messages = payload["request"]["structured_messages"]
    assert any(m.get("content") == "responses-surface prompt" for m in messages)


@pytest.mark.asyncio
async def test_post_call_fail_closed_raises_modify_response_exception():
    g = _make_guardrail(unreachable_fallback="fail_closed")
    g.async_handler.post.side_effect = httpx.ConnectError("boom")
    response = ModelResponse(
        choices=[Choices(finish_reason="stop", index=0, message=Message(content="secret", role="assistant"))],
        model="gpt-4o-mini",
    )
    request_data = {"model": "gpt-4o-mini", "response": response}
    with pytest.raises(ModifyResponseException) as exc:
        await g.apply_guardrail(
            inputs={"texts": ["secret"], "model": "gpt-4o-mini"},
            request_data=request_data,
            input_type="response",
            logging_obj=_logging_obj(),
        )
    assert exc.value.original_response is response


@pytest.mark.asyncio
async def test_usage_tokens_on_post_call():
    g = _make_guardrail()
    g.async_handler.post.return_value = _mock_response("NONE")
    response = ModelResponse(
        choices=[Choices(finish_reason="stop", index=0, message=Message(content="hi", role="assistant"))],
        model="gpt-4o-mini",
        usage=Usage(prompt_tokens=11, completion_tokens=7, total_tokens=18),
    )
    await g.apply_guardrail(
        inputs={"texts": ["hi"], "model": "gpt-4o-mini"},
        request_data={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hey"}], "response": response},
        input_type="response",
        logging_obj=_logging_obj(),
    )
    usage = _posted_payload(g)["usage"]
    assert usage == {"input_tokens": 11, "output_tokens": 7}


@pytest.mark.asyncio
async def test_usage_absent_on_pre_call():
    g = _make_guardrail()
    g.async_handler.post.return_value = _mock_response("NONE")
    await g.apply_guardrail(
        inputs={"texts": ["hi"]},
        request_data={"model": "gpt-4o-mini"},
        input_type="request",
        logging_obj=_logging_obj(),
    )
    assert "usage" not in _posted_payload(g)


@pytest.mark.asyncio
async def test_allow_returns_inputs_unchanged():
    g = _make_guardrail()
    g.async_handler.post.return_value = _mock_response("NONE")
    inputs = {"texts": ["fine"]}
    out = await g.apply_guardrail(
        inputs=inputs, request_data={"model": "m"}, input_type="request", logging_obj=_logging_obj()
    )
    assert out is inputs


@pytest.mark.asyncio
async def test_unreachable_fail_closed_blocks():
    g = _make_guardrail(unreachable_fallback="fail_closed")
    g.async_handler.post.side_effect = httpx.ConnectError("boom")
    with pytest.raises(GuardrailRaisedException):
        await g.apply_guardrail(
            inputs={"texts": ["x"]}, request_data={"model": "m"}, input_type="request", logging_obj=_logging_obj()
        )


@pytest.mark.asyncio
async def test_unreachable_fail_open_passes_through():
    g = _make_guardrail(unreachable_fallback="fail_open")
    g.async_handler.post.side_effect = httpx.ConnectError("boom")
    inputs = {"texts": ["x"]}
    out = await g.apply_guardrail(
        inputs=inputs, request_data={"model": "m"}, input_type="request", logging_obj=_logging_obj()
    )
    assert out is inputs


@pytest.mark.asyncio
async def test_fail_on_error_false_allows_on_bad_status():
    g = _make_guardrail(unreachable_fallback="fail_closed", fail_on_error=False)
    bad = MagicMock(spec=httpx.Response)
    bad.status_code = 400
    bad.text = "bad request"
    g.async_handler.post.return_value = bad
    inputs = {"texts": ["x"]}
    out = await g.apply_guardrail(
        inputs=inputs, request_data={"model": "m"}, input_type="request", logging_obj=_logging_obj()
    )
    assert out is inputs


@pytest.mark.asyncio
async def test_non_retryable_status_fail_closed_blocks():
    g = _make_guardrail(unreachable_fallback="fail_closed", fail_on_error=True)
    bad = MagicMock(spec=httpx.Response)
    bad.status_code = 401
    bad.text = "unauthorized"
    g.async_handler.post.return_value = bad
    with pytest.raises(GuardrailRaisedException):
        await g.apply_guardrail(
            inputs={"texts": ["x"]}, request_data={"model": "m"}, input_type="request", logging_obj=_logging_obj()
        )


@pytest.mark.asyncio
async def test_payload_size_guard_fails_closed():
    g = _make_guardrail(max_payload_bytes=10)
    inputs = {"texts": ["x" * 5000]}
    with pytest.raises(GuardrailRaisedException):
        await g.apply_guardrail(
            inputs=inputs, request_data={"model": "m"}, input_type="request", logging_obj=_logging_obj()
        )
    g.async_handler.post.assert_not_called()


@pytest.mark.asyncio
async def test_payload_size_guard_blocks_even_with_fail_open():
    g = _make_guardrail(max_payload_bytes=10, unreachable_fallback="fail_open")
    inputs = {"texts": ["x" * 5000]}
    with pytest.raises(GuardrailRaisedException):
        await g.apply_guardrail(
            inputs=inputs, request_data={"model": "m"}, input_type="request", logging_obj=_logging_obj()
        )
    g.async_handler.post.assert_not_called()


@pytest.mark.asyncio
async def test_invalid_response_schema_blocks_even_with_fail_open():
    g = _make_guardrail(unreachable_fallback="fail_open")
    bad = MagicMock(spec=httpx.Response)
    bad.status_code = 200
    bad.json.return_value = {"action": "NOT_A_VALID_ACTION"}
    bad.text = ""
    g.async_handler.post.return_value = bad
    with pytest.raises(GuardrailRaisedException):
        await g.apply_guardrail(
            inputs={"texts": ["x"]}, request_data={"model": "m"}, input_type="request", logging_obj=_logging_obj()
        )


@pytest.mark.asyncio
async def test_unreachable_http_status_fail_open_passes():
    g = _make_guardrail(unreachable_fallback="fail_open")
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = 503
    resp.text = "service unavailable"
    g.async_handler.post.return_value = resp
    inputs = {"texts": ["x"]}
    out = await g.apply_guardrail(
        inputs=inputs, request_data={"model": "m"}, input_type="request", logging_obj=_logging_obj()
    )
    assert out is inputs


@pytest.mark.asyncio
async def test_unreachable_http_status_fail_closed_blocks():
    g = _make_guardrail(unreachable_fallback="fail_closed")
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = 503
    resp.text = "service unavailable"
    g.async_handler.post.return_value = resp
    with pytest.raises(GuardrailRaisedException):
        await g.apply_guardrail(
            inputs={"texts": ["x"]}, request_data={"model": "m"}, input_type="request", logging_obj=_logging_obj()
        )


@pytest.mark.asyncio
async def test_post_call_preserves_anthropic_tool_blocks_in_request_messages():
    g = _make_guardrail()
    g.async_handler.post.return_value = _mock_response("NONE")
    anthropic_messages = [
        {"role": "user", "content": "What's the weather in Paris?"},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "get_weather",
                    "input": {"city": "Paris"},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_1",
                    "content": "18C, cloudy",
                }
            ],
        },
    ]
    response = {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "Mild and cloudy."}],
        "stop_reason": "end_turn",
        "model": "claude-sonnet-5",
    }
    await g.apply_guardrail(
        inputs={"texts": ["Mild and cloudy."], "model": "claude-sonnet-5"},
        request_data={
            "model": "claude-sonnet-5",
            "messages": anthropic_messages,
            "response": response,
        },
        input_type="response",
        logging_obj=_logging_obj(),
    )
    payload = _posted_payload(g)
    assert payload["request"]["structured_messages"] == anthropic_messages
    assert payload["response"]["finish_reason"] == "end_turn"


@pytest.mark.asyncio
async def test_pre_call_preserves_anthropic_tool_blocks_in_structured_messages():
    g = _make_guardrail()
    g.async_handler.post.return_value = _mock_response("NONE")
    anthropic_messages = [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "get_weather",
                    "input": {"city": "Paris"},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_1",
                    "content": "18C",
                }
            ],
        },
    ]
    await g.apply_guardrail(
        inputs={"structured_messages": anthropic_messages, "model": "claude-sonnet-5"},
        request_data={"model": "claude-sonnet-5", "messages": anthropic_messages},
        input_type="request",
        logging_obj=_logging_obj(),
    )
    assert _posted_payload(g)["request"]["structured_messages"] == anthropic_messages


@pytest.mark.asyncio
async def test_response_finish_reason_from_openai_choices_still_works():
    g = _make_guardrail()
    g.async_handler.post.return_value = _mock_response("NONE")
    response = ModelResponse(
        choices=[Choices(finish_reason="tool_calls", index=0, message=Message(content=None, role="assistant"))],
        model="gpt-4o-mini",
    )
    await g.apply_guardrail(
        inputs={
            "texts": [],
            "tool_calls": [
                ChatCompletionMessageToolCall(
                    id="c1",
                    type="function",
                    function=Function(name="f", arguments="{}"),
                )
            ],
        },
        request_data={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}], "response": response},
        input_type="response",
        logging_obj=_logging_obj(),
    )
    payload = _posted_payload(g)
    assert payload["response"]["finish_reason"] == "tool_calls"
    assert payload["response"]["tool_calls"] == [
        {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}
    ]


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        (None, None),
        ({"choices": "invalid"}, None),
        ({"choices": [{"finish_reason": "length"}]}, "length"),
        ({"choices": [{"stop_reason": "end_turn"}]}, "end_turn"),
        ({"choices": [{}]}, None),
        (SimpleNamespace(stop_reason="end_turn"), "end_turn"),
    ],
)
def test_response_finish_reason_handles_supported_shapes(response, expected):
    assert _response_finish_reason(response) == expected
