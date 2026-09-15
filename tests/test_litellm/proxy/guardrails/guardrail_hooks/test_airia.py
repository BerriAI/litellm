import uuid
from typing import Final, Literal, cast
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

import litellm
from litellm.exceptions import GuardrailRaisedException
from litellm.proxy.guardrails.guardrail_hooks.airia.airia import AiriaGuardrail
from litellm.proxy.guardrails.init_guardrails import init_guardrails_v2
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.proxy.guardrails.guardrail_hooks.airia import AiriaGuardrailConfigModel
from litellm.types.utils import GenericGuardrailAPIInputs

API_BASE = "https://gateway.airia.ai"
API_KEY = "ak-test-key"


def _make_guardrail(**kwargs) -> AiriaGuardrail:
    """Build a guardrail; construction defaults still exercise the real HTTP client."""
    kwargs.setdefault("api_base", API_BASE)
    kwargs.setdefault("api_key", API_KEY)
    kwargs.setdefault("guardrail_name", "airia-guard")
    return AiriaGuardrail(**kwargs)


def _make_guardrail_with_response(payload: object, status_code: int = 200, **kwargs) -> AiriaGuardrail:
    """Build a guardrail with a fake HTTP client injected at construction, per CLAUDE.md's
    dependency-injection rule: no test mutates `async_handler` after the fact."""
    handler = MagicMock()
    handler.post = AsyncMock(return_value=_mock_response(payload, status_code))
    return _make_guardrail(async_handler=handler, **kwargs)


def _mock_response(payload: object, status_code: int = 200) -> MagicMock:
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = payload
    response.raise_for_status = MagicMock()
    return response


def _inputs(**overrides: object) -> GenericGuardrailAPIInputs:
    return cast(
        GenericGuardrailAPIInputs, {"texts": ["hello"], "model": "gpt-4o", **overrides}
    )  # cast-ok: test fixture


def test_airia_guardrail_config(monkeypatch: pytest.MonkeyPatch):
    """The guardrail registers through init_guardrails_v2 under the `airia` key."""
    monkeypatch.setattr(litellm, "guardrail_name_config_map", {})
    monkeypatch.setattr(litellm, "callbacks", [])
    monkeypatch.setenv("AIRIA_GATEWAY_URL", API_BASE)
    monkeypatch.setenv("AIRIA_API_KEY", API_KEY)

    init_guardrails_v2(
        all_guardrails=[
            {
                "guardrail_name": "airia-guard",
                "litellm_params": {
                    "guardrail": "airia",
                    "mode": "pre_call",
                    "default_on": True,
                    "timeout": 45.0,
                },
            }
        ],
        config_file_path="",
    )

    registered = [c for c in litellm.callbacks if isinstance(c, AiriaGuardrail)]
    assert len(registered) == 1
    assert registered[0].guardrail_name == "airia-guard"
    assert registered[0].default_on is True
    assert registered[0].event_hook == "pre_call"
    assert registered[0].async_handler.timeout.read == 45.0


def test_during_call_is_not_a_supported_event_hook():
    """during_call runs concurrently with the upstream call, so a block can land too late."""
    hooks = _make_guardrail().supported_event_hooks

    assert hooks is not None
    assert GuardrailEventHooks.during_call not in hooks
    assert list(hooks) == [GuardrailEventHooks.pre_call, GuardrailEventHooks.post_call]


@pytest.mark.parametrize("missing", ["api_base", "api_key"])
def test_missing_credentials_raise_at_construction(missing: str, monkeypatch: pytest.MonkeyPatch):
    """Fail at startup rather than on the first request."""
    monkeypatch.delenv("AIRIA_GATEWAY_URL", raising=False)
    monkeypatch.delenv("AIRIA_API_KEY", raising=False)

    kwargs = {"api_base": API_BASE, "api_key": API_KEY}
    kwargs[missing] = None

    with pytest.raises(ValueError, match="AiriaGuardrail requires"):
        _make_guardrail(**kwargs)


def test_credentials_fall_back_to_environment(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("AIRIA_GATEWAY_URL", API_BASE)
    monkeypatch.setenv("AIRIA_API_KEY", API_KEY)

    guardrail = _make_guardrail(api_base=None, api_key=None)

    assert guardrail.api_base == API_BASE
    assert guardrail.api_key == API_KEY


def test_trailing_slash_is_stripped_from_api_base():
    guardrail = _make_guardrail(api_base=f"{API_BASE}/")

    assert guardrail.api_base == API_BASE


def test_custom_timeout_from_kwargs():
    guardrail = _make_guardrail(timeout=45.0)

    assert guardrail.async_handler.timeout.read == 45.0
    assert guardrail.async_handler.timeout.connect == 5.0


def test_timeout_falls_back_to_environment(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("AIRIA_TIMEOUT", "30")

    guardrail = _make_guardrail()

    assert guardrail.async_handler.timeout.read == 30.0


@pytest.mark.asyncio
async def test_request_payload_and_auth_header():
    guardrail = _make_guardrail_with_response({"action": "NONE"})

    inputs = _inputs(
        images=["data:image/png;base64,AAAA"],
        structured_messages=[{"role": "user", "content": "hello"}],
        tools=[{"type": "function", "function": {"name": "f"}}],
        tool_calls=[{"id": "c1", "function": {"name": "f", "arguments": "{}"}}],
    )
    await guardrail.apply_guardrail(inputs=inputs, request_data={"litellm_call_id": "call-123"}, input_type="request")

    call = guardrail.async_handler.post.call_args
    assert call.args[0] == f"{API_BASE}/v1/guardrails/litellm"
    assert call.kwargs["headers"] == {"Authorization": f"Bearer {API_KEY}"}

    payload = call.kwargs["json"]
    assert payload["input_type"] == "request"
    assert payload["texts"] == ["hello"]
    assert payload["structured_messages"] == [{"role": "user", "content": "hello"}]
    assert payload["images"] == ["data:image/png;base64,AAAA"]
    assert payload["tools"] == [{"type": "function", "function": {"name": "f"}}]
    assert payload["tool_calls"] == [{"id": "c1", "function": {"name": "f", "arguments": "{}"}}]
    assert payload["model"] == "gpt-4o"
    assert payload["litellm_call_id"] == "call-123"


@pytest.mark.asyncio
@pytest.mark.parametrize("input_type", ["request", "response"])
async def test_input_type_is_forwarded_verbatim(input_type: Literal["request", "response"]):
    guardrail = _make_guardrail_with_response({"action": "NONE"})

    await guardrail.apply_guardrail(inputs=_inputs(), request_data={}, input_type=input_type)

    assert guardrail.async_handler.post.call_args.kwargs["json"]["input_type"] == input_type


@pytest.mark.asyncio
async def test_call_id_falls_back_to_a_generated_uuid():
    guardrail = _make_guardrail_with_response({"action": "NONE"})

    await guardrail.apply_guardrail(inputs=_inputs(), request_data={}, input_type="request")

    call_id = guardrail.async_handler.post.call_args.kwargs["json"]["litellm_call_id"]
    assert uuid.UUID(call_id).version == 4


@pytest.mark.asyncio
async def test_action_none_returns_inputs_unchanged():
    guardrail = _make_guardrail_with_response({"action": "NONE"})
    inputs = _inputs(structured_messages=[{"role": "user", "content": "hello"}])

    result = await guardrail.apply_guardrail(inputs=inputs, request_data={}, input_type="request")

    assert result["texts"] == ["hello"]
    assert result["structured_messages"] == [{"role": "user", "content": "hello"}]


@pytest.mark.asyncio
async def test_action_blocked_raises_with_blocked_content_set():
    guardrail = _make_guardrail_with_response({"action": "BLOCKED", "blocked_reason": "Contains a secret"})

    with pytest.raises(GuardrailRaisedException) as excinfo:
        await guardrail.apply_guardrail(inputs=_inputs(), request_data={}, input_type="request")

    assert excinfo.value.blocked_content is True
    assert "Contains a secret" in str(excinfo.value)


@pytest.mark.asyncio
async def test_action_blocked_without_a_reason_uses_the_default_message():
    guardrail = _make_guardrail_with_response({"action": "BLOCKED"})

    with pytest.raises(GuardrailRaisedException) as excinfo:
        await guardrail.apply_guardrail(inputs=_inputs(), request_data={}, input_type="request")

    assert excinfo.value.blocked_content is True
    assert "Blocked by your organization's content policy." in str(excinfo.value)


@pytest.mark.asyncio
async def test_intervened_replaces_both_text_bearing_fields():
    """Substituting only one field would leave the other still carrying the unredacted text."""
    guardrail = _make_guardrail_with_response(
        {
            "action": "GUARDRAIL_INTERVENED",
            "texts": ["my email is [EmailAddress1]"],
            "structured_messages": [{"role": "user", "content": "my email is [EmailAddress1]"}],
        }
    )
    inputs = _inputs(
        texts=["my email is a@b.com"],
        structured_messages=[{"role": "user", "content": "my email is a@b.com"}],
    )

    result = await guardrail.apply_guardrail(inputs=inputs, request_data={}, input_type="request")

    assert result["texts"] == ["my email is [EmailAddress1]"]
    assert result["structured_messages"] == [{"role": "user", "content": "my email is [EmailAddress1]"}]
    assert "a@b.com" not in str(result)


@pytest.mark.asyncio
async def test_intervened_leaves_a_field_alone_when_airia_omits_it():
    """An omitted field means "not rewritten", not "rewritten to empty"."""
    guardrail = _make_guardrail_with_response({"action": "GUARDRAIL_INTERVENED", "texts": ["redacted"]})
    inputs = _inputs(texts=["original"], structured_messages=[{"role": "user", "content": "keep me"}])

    result = await guardrail.apply_guardrail(inputs=inputs, request_data={}, input_type="request")

    assert result["texts"] == ["redacted"]
    assert result["structured_messages"] == [{"role": "user", "content": "keep me"}]


@pytest.mark.asyncio
async def test_intervened_with_a_non_list_field_is_treated_as_a_block():
    """A rewrite this version cannot apply must not let the original text through."""
    guardrail = _make_guardrail_with_response({"action": "GUARDRAIL_INTERVENED", "texts": "not a list"})

    with pytest.raises(GuardrailRaisedException) as excinfo:
        await guardrail.apply_guardrail(inputs=_inputs(texts=["secret"]), request_data={}, input_type="request")

    assert excinfo.value.blocked_content is True


@pytest.mark.asyncio
async def test_intervened_with_a_non_list_structured_messages_is_treated_as_a_block():
    guardrail = _make_guardrail_with_response(
        {"action": "GUARDRAIL_INTERVENED", "structured_messages": {"role": "user"}}
    )

    with pytest.raises(GuardrailRaisedException) as excinfo:
        await guardrail.apply_guardrail(inputs=_inputs(), request_data={}, input_type="request")

    assert excinfo.value.blocked_content is True


@pytest.mark.asyncio
async def test_intervened_without_any_rewritten_field_is_treated_as_a_block():
    """An intervention the proxy cannot apply must not let the original content through."""
    guardrail = _make_guardrail_with_response({"action": "GUARDRAIL_INTERVENED", "tool_calls": []})

    with pytest.raises(GuardrailRaisedException) as excinfo:
        await guardrail.apply_guardrail(inputs=_inputs(texts=["secret"]), request_data={}, input_type="request")

    assert excinfo.value.blocked_content is True


@pytest.mark.asyncio
@pytest.mark.parametrize("body", [["not", "an", "object"], "just a string", 42, None])
async def test_a_non_object_response_body_fails_closed_but_is_not_reported_as_a_verdict(body: object):
    """response.json() is not guaranteed to return a dict; body.get(...) on a list or scalar
    would otherwise raise AttributeError and surface as an internal error, not a clean refusal."""
    guardrail = _make_guardrail_with_response(body)

    with pytest.raises(GuardrailRaisedException) as excinfo:
        await guardrail.apply_guardrail(inputs=_inputs(), request_data={}, input_type="request")

    assert excinfo.value.blocked_content is False
    assert "could not evaluate" in str(excinfo.value)


@pytest.mark.asyncio
async def test_intervened_with_a_texts_rewrite_of_the_wrong_length_is_treated_as_a_block():
    """A rewrite with a different item count than what was sent would misalign the positional
    write-back downstream; accepting it could drop content or leave other items unredacted."""
    guardrail = _make_guardrail_with_response({"action": "GUARDRAIL_INTERVENED", "texts": ["only one redacted item"]})

    with pytest.raises(GuardrailRaisedException) as excinfo:
        await guardrail.apply_guardrail(
            inputs=_inputs(texts=["first secret", "second secret"]), request_data={}, input_type="request"
        )

    assert excinfo.value.blocked_content is True


@pytest.mark.asyncio
async def test_intervened_returns_a_copy_and_leaves_the_caller_object_untouched():
    guardrail = _make_guardrail_with_response({"action": "GUARDRAIL_INTERVENED", "texts": ["redacted"]})
    inputs = _inputs(texts=["original"])

    result = await guardrail.apply_guardrail(inputs=inputs, request_data={}, input_type="request")

    assert result["texts"] == ["redacted"]
    assert inputs["texts"] == ["original"]


@pytest.mark.asyncio
async def test_streaming_redaction_actually_reaches_the_client():
    """block_only (the framework default) drops guardrail rewrites on the streaming path, so a
    REDACT verdict would silently leak the original text. Runs the real UnifiedLLMGuardrails
    streaming iterator end to end (not just this guardrail's config attributes) and checks what
    the client would actually receive.
    """
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.guardrails.guardrail_hooks.unified_guardrail.unified_guardrail import (
        UnifiedLLMGuardrails,
    )
    from litellm.types.utils import Delta, ModelResponseStream, StreamingChoices

    guardrail: Final = _make_guardrail_with_response(
        {"action": "GUARDRAIL_INTERVENED", "texts": ["Contact me at [EmailAddress1]"]},
        event_hook="post_call",
        default_on=True,
    )

    async def source_stream():
        for content in ("Contact ", "me at ", "ada@example.com"):
            yield ModelResponseStream(
                model="gpt-4o-mini", choices=[StreamingChoices(index=0, delta=Delta(content=content))]
            )
        yield ModelResponseStream(
            model="gpt-4o-mini",
            choices=[StreamingChoices(index=0, delta=Delta(content=None), finish_reason="stop")],
        )

    assembled = ""
    async for chunk in UnifiedLLMGuardrails().async_post_call_streaming_iterator_hook(
        user_api_key_dict=UserAPIKeyAuth(api_key="test", request_route="/chat/completions"),
        response=source_stream(),
        request_data={  # mutable-ok: one-shot request_data the framework requires as a plain dict
            "messages": [{"role": "user", "content": "share your contact"}]
        },
        guardrail_to_apply=guardrail,
    ):
        for choice in chunk.choices:
            if choice.delta.content:
                assembled += choice.delta.content

    assert "[EmailAddress1]" in assembled
    assert "ada@example.com" not in assembled
    guardrail.async_handler.post.assert_called_once()


def test_config_model_is_exposed():
    assert AiriaGuardrail.get_config_model() is AiriaGuardrailConfigModel
    assert AiriaGuardrailConfigModel.ui_friendly_name() == "Airia Guardrail"


@pytest.mark.asyncio
async def test_unrecognized_action_is_treated_as_a_block():
    """An action this version cannot interpret must not become an allow."""
    guardrail = _make_guardrail_with_response({"action": "SOME_FUTURE_ACTION"})

    with pytest.raises(GuardrailRaisedException) as excinfo:
        await guardrail.apply_guardrail(inputs=_inputs(), request_data={}, input_type="request")

    assert excinfo.value.blocked_content is True


@pytest.mark.asyncio
async def test_missing_action_is_treated_as_a_block():
    """A response with no action at all is not an allow either."""
    guardrail = _make_guardrail_with_response({})

    with pytest.raises(GuardrailRaisedException) as excinfo:
        await guardrail.apply_guardrail(inputs=_inputs(), request_data={}, input_type="request")

    assert excinfo.value.blocked_content is True


@pytest.mark.asyncio
async def test_transport_failure_fails_closed_but_is_not_reported_as_a_verdict():
    """Still refused, but blocked_content stays False: could not evaluate is not a verdict."""
    handler: Final = MagicMock()
    handler.post = AsyncMock(side_effect=httpx.ConnectError("connection refused"))
    guardrail: Final = _make_guardrail(async_handler=handler)

    with pytest.raises(GuardrailRaisedException) as excinfo:
        await guardrail.apply_guardrail(inputs=_inputs(), request_data={}, input_type="request")

    assert excinfo.value.blocked_content is False
    assert "could not evaluate" in str(excinfo.value)


@pytest.mark.asyncio
async def test_non_2xx_fails_closed_but_is_not_reported_as_a_verdict():
    response: Final = _mock_response({}, status_code=503)
    response.raise_for_status.side_effect = httpx.HTTPStatusError("503", request=MagicMock(), response=MagicMock())
    handler: Final = MagicMock()
    handler.post = AsyncMock(return_value=response)
    guardrail: Final = _make_guardrail(async_handler=handler)

    with pytest.raises(GuardrailRaisedException) as excinfo:
        await guardrail.apply_guardrail(inputs=_inputs(), request_data={}, input_type="request")

    assert excinfo.value.blocked_content is False
