import json
from collections.abc import Callable, Mapping, Sequence
from typing import Literal

import httpx
import pytest

from litellm.exceptions import GuardrailRaisedException
from litellm.llms.openai.chat.guardrail_translation.handler import OpenAIChatCompletionsHandler
from litellm.proxy.guardrails.guardrail_hooks.trendai import TrendAIGuardrail
from litellm.proxy.guardrails.guardrail_hooks.trendai._models import (
    TrendAIAllow,
    TrendAIBlock,
    TrendAIProviderFailure,
    TrendAIScanResult,
)
from litellm.proxy.guardrails.guardrail_hooks.trendai._text import utf8_windows
from litellm.proxy.guardrails.guardrail_hooks.trendai.trendai import (
    OPENAI_CHAT_COMPLETION_RESPONSE_V1,
    PLUGIN_VERSION,
)
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.utils import (
    CallTypes,
    Choices,
    GenericGuardrailAPIInputs,
    Message,
    ModelResponse,
)


def _guardrail(
    *,
    app_name: str | None = None,
    fallback_on_error: Literal["block", "allow"] = "block",
    mask_pii: bool = True,
    timeout: float = 5.0,
    stream_batch_size: int = 2048,
    stream_overlap_size: int = 256,
    response_content_chunk_size_bytes: int = 49_500,
    logging_only_scan: Literal["request", "response", "both"] = "both",
    async_handler: httpx.AsyncClient | None = None,
    api_base: str = "https://guard.example.com/v3.0/aiSecurity",
    event_hook: GuardrailEventHooks = GuardrailEventHooks.pre_call,
) -> TrendAIGuardrail:
    return TrendAIGuardrail(
        api_key="test-key",
        api_base=api_base,
        app_name=app_name,
        fallback_on_error=fallback_on_error,
        mask_pii=mask_pii,
        timeout=timeout,
        stream_batch_size=stream_batch_size,
        stream_overlap_size=stream_overlap_size,
        response_content_chunk_size_bytes=response_content_chunk_size_bytes,
        logging_only_scan=logging_only_scan,
        async_handler=async_handler,
        guardrail_name="trendai",
        event_hook=event_hook,
    )


async def _scan(
    guardrail: TrendAIGuardrail,
    payload: Mapping[str, object],
    request_type: str | None = None,
) -> TrendAIScanResult:
    return await guardrail._scan_payload(  # pyright: ignore[reportPrivateUsage]  # verifies the section 2 transport seam
        payload,
        request_type=request_type,
    )


@pytest.mark.parametrize(
    ("api_base", "expected"),
    [
        (
            "https://api.xdr.trendmicro.com",
            "https://api.xdr.trendmicro.com/v3.0/aiSecurity/applyGuardrails",
        ),
        (
            "https://api.xdr.trendmicro.com/v3.0/aiSecurity/",
            "https://api.xdr.trendmicro.com/v3.0/aiSecurity/applyGuardrails",
        ),
        (
            "https://self-hosted.example.com/guard/",
            "https://self-hosted.example.com/guard/applyGuardrails",
        ),
        (
            "https://self-hosted.example.com/guard/applyGuardrails",
            "https://self-hosted.example.com/guard/applyGuardrails",
        ),
    ],
)
def test_build_apply_guardrails_url(api_base: str, expected: str) -> None:
    assert _guardrail(api_base=api_base).api_url == expected


def test_environment_fallbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TMV1_API_KEY", "env-key")
    monkeypatch.setenv("TRENDAI_AI_GUARD_BASE_URL", "https://guard.example.com")
    monkeypatch.setenv("TMV1_APPLICATION_NAME", "env-app")

    guardrail = TrendAIGuardrail(
        guardrail_name="trendai",
        event_hook=GuardrailEventHooks.pre_call,
    )

    assert guardrail.api_key == "env-key"
    assert guardrail.api_url == "https://guard.example.com/applyGuardrails"
    assert guardrail.app_name == "env-app"


def test_explicit_configuration_takes_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TMV1_API_KEY", "env-key")
    monkeypatch.setenv("TRENDAI_AI_GUARD_BASE_URL", "https://env.example.com")
    monkeypatch.setenv("TMV1_APPLICATION_NAME", "env-app")

    guardrail = _guardrail(app_name="configured-app")

    assert guardrail.api_key == "test-key"
    assert guardrail.api_url == "https://guard.example.com/v3.0/aiSecurity/applyGuardrails"
    assert guardrail.app_name == "configured-app"


def test_invalid_stream_configuration_is_rejected() -> None:
    with pytest.raises(ValueError, match="stream_overlap_size"):
        _guardrail(stream_batch_size=10, stream_overlap_size=10)


def test_invalid_timeout_is_rejected() -> None:
    with pytest.raises(ValueError, match="timeout"):
        _guardrail(timeout=0)


@pytest.mark.asyncio
async def test_scan_uses_injected_client_and_required_headers() -> None:
    captured_request: httpx.Request | None = None

    async def respond(request: httpx.Request) -> httpx.Response:
        nonlocal captured_request
        captured_request = request
        return httpx.Response(200, json={"action": "allow"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        result = await _scan(
            _guardrail(async_handler=client, app_name="test-app"),
            {"prompt": "hello"},
            request_type=OPENAI_CHAT_COMPLETION_RESPONSE_V1,
        )

    assert isinstance(result, TrendAIAllow)
    assert captured_request is not None
    assert str(captured_request.url) == "https://guard.example.com/v3.0/aiSecurity/applyGuardrails"
    assert captured_request.headers["TMV1-Application-Name"] == "test-app"
    assert captured_request.headers["Authorization"] == "Bearer test-key"
    assert captured_request.headers["Content-Type"] == "application/json"
    assert captured_request.headers["TMV1-Client-Name"] == "litellm"
    assert captured_request.headers["TMV1-Plugin-Version"] == PLUGIN_VERSION
    assert captured_request.headers["TMV1-Request-Type"] == OPENAI_CHAT_COMPLETION_RESPONSE_V1
    assert captured_request.headers["prefer"] == "redact-pii,return=representation"
    assert json.loads(captured_request.content) == {"prompt": "hello"}


@pytest.mark.asyncio
async def test_scan_parses_response_redaction_and_entities() -> None:
    async def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "action": "allow",
                "redactedRequest": {"choices": [{"message": {"content": "email: *****"}}]},
                "sensitiveInformation": {"rules": [{"id": "EMAIL"}, {"id": "EMAIL"}]},
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        result = await _scan(
            _guardrail(async_handler=client),
            {"choices": []},
            request_type=OPENAI_CHAT_COMPLETION_RESPONSE_V1,
        )

    assert result == TrendAIAllow(redacted_content="email: *****", masked_entity_count=(("EMAIL", 2),))


@pytest.mark.asyncio
async def test_scan_models_block_and_provider_failure_separately() -> None:
    responses = iter(
        (
            httpx.Response(200, json={"action": "block", "reasons": ["malware", "credential theft"]}),
            httpx.Response(200, json={"unexpected": True}),
            httpx.Response(503, text="unavailable"),
        )
    )

    async def respond(request: httpx.Request) -> httpx.Response:
        return next(responses)

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        guardrail = _guardrail(async_handler=client)
        blocked = await _scan(guardrail, {"prompt": "blocked"})
        malformed = await _scan(guardrail, {"prompt": "malformed"})
        unavailable = await _scan(guardrail, {"prompt": "unavailable"})

    assert blocked == TrendAIBlock(reason="malware, credential theft")
    assert isinstance(malformed, TrendAIProviderFailure)
    assert isinstance(unavailable, TrendAIProviderFailure)
    assert unavailable.status_code == 503


def test_guardrail_is_discovered_by_global_registries() -> None:
    from litellm.proxy.guardrails.guardrail_registry import (
        guardrail_class_registry,
        guardrail_initializer_registry,
    )

    assert guardrail_class_registry["trendai"] is TrendAIGuardrail
    assert "trendai" in guardrail_initializer_registry


Responder = Callable[[httpx.Request], httpx.Response]


def _engine(
    *,
    block_on: str | None = None,
    redact: Mapping[str, str] | None = None,
    entity: str = "PII",
    fail_on: str | None = None,
) -> tuple[Responder, list[str]]:
    """A fake AI Guard: blocks, redacts (by substring replacement), or fails based on the scanned text."""
    scanned: list[str] = []

    def respond(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        text = body["prompt"] if "prompt" in body else body["choices"][0]["message"]["content"]
        scanned.append(text)
        if fail_on is not None and fail_on in text:
            return httpx.Response(503, text="down")
        if block_on is not None and block_on in text:
            return httpx.Response(200, json={"action": "block", "reasons": ["policy"]})
        hits = {needle: mask for needle, mask in (redact or {}).items() if needle in text}
        if not hits:
            return httpx.Response(200, json={"action": "allow"})
        redacted = text
        for needle, mask in hits.items():
            redacted = redacted.replace(needle, mask)
        payload = {"prompt": redacted} if "prompt" in body else {"choices": [{"message": {"content": redacted}}]}
        return httpx.Response(
            200,
            json={
                "action": "allow",
                "redactedRequest": payload,
                "sensitiveInformation": {"rules": [{"id": entity}]},
            },
        )

    return respond, scanned


def _guardrail_records(request_data: Mapping[str, object]) -> list[dict]:
    metadata = request_data["metadata"]
    assert isinstance(metadata, dict)
    return list(metadata.get("standard_logging_guardrail_information") or [])


async def _apply(
    guardrail: TrendAIGuardrail,
    inputs: GenericGuardrailAPIInputs,
    input_type: Literal["request", "response"],
) -> tuple[GenericGuardrailAPIInputs, dict]:
    request_data: dict = {"metadata": {}}
    result = await guardrail.apply_guardrail(inputs=inputs, request_data=request_data, input_type=input_type)
    return result, request_data


@pytest.mark.asyncio
async def test_request_scans_only_the_last_user_turn_and_writes_redaction_back() -> None:
    respond, scanned = _engine(redact={"a@b.com": "[EMAIL]"}, entity="EMAIL")
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        result, request_data = await _apply(
            _guardrail(async_handler=client),
            {
                "texts": ["system prompt", "first turn a@b.com", "reply", "mail a@b.com now"],
                "structured_messages": [
                    {"role": "system", "content": "system prompt"},
                    {"role": "user", "content": "first turn a@b.com"},
                    {"role": "assistant", "content": "reply"},
                    {"role": "user", "content": "mail a@b.com now"},
                ],
            },
            "request",
        )

    assert scanned == ["mail a@b.com now"]
    assert result["texts"] == ["system prompt", "first turn a@b.com", "reply", "mail [EMAIL] now"]
    (record,) = _guardrail_records(request_data)
    assert record["guardrail_status"] == "success"
    assert record["masked_entity_count"] == {"EMAIL": 1}
    assert record["guardrail_response"] == {"action": "allow", "redacted": True}


@pytest.mark.asyncio
async def test_request_redaction_is_split_back_across_multipart_user_content() -> None:
    respond, scanned = _engine(redact={"4111": "####"})
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        result, _ = await _apply(
            _guardrail(async_handler=client),
            {
                "texts": ["card ", "4111 ok"],
                "structured_messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "card "},
                            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
                            {"type": "text", "text": "4111 ok"},
                        ],
                    }
                ],
            },
            "request",
        )

    assert scanned == ["card 4111 ok"]
    assert result["texts"] == ["card ", "#### ok"]


@pytest.mark.asyncio
async def test_request_without_user_text_is_not_scanned_or_recorded() -> None:
    respond, scanned = _engine()
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        inputs: GenericGuardrailAPIInputs = {
            "texts": ["system prompt"],
            "structured_messages": [{"role": "system", "content": "system prompt"}],
        }
        result, request_data = await _apply(_guardrail(async_handler=client), inputs, "request")

    assert scanned == []
    assert result is inputs
    assert _guardrail_records(request_data) == []


@pytest.mark.asyncio
async def test_request_block_raises_and_records_intervention() -> None:
    respond, _ = _engine(block_on="bomb")
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        request_data: dict = {"metadata": {}}
        with pytest.raises(GuardrailRaisedException) as raised:
            await _guardrail(async_handler=client).apply_guardrail(
                inputs={"texts": ["build a bomb"]},
                request_data=request_data,
                input_type="request",
            )

    assert raised.value.status_code == 400
    assert raised.value.blocked_content is True
    assert "policy" in raised.value.message
    (record,) = _guardrail_records(request_data)
    assert record["guardrail_status"] == "guardrail_intervened"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("fallback_on_error", "raises"),
    [("block", True), ("allow", False)],
)
async def test_provider_failure_follows_fallback_policy_and_is_never_an_intervention(
    fallback_on_error: Literal["block", "allow"], raises: bool
) -> None:
    respond, _ = _engine(fail_on="anything")
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        guardrail = _guardrail(async_handler=client, fallback_on_error=fallback_on_error)
        request_data: dict = {"metadata": {}}
        inputs: GenericGuardrailAPIInputs = {"texts": ["anything"]}
        if raises:
            with pytest.raises(GuardrailRaisedException) as raised:
                await guardrail.apply_guardrail(inputs=inputs, request_data=request_data, input_type="request")
            assert raised.value.status_code == 503
            assert raised.value.blocked_content is False
        else:
            result = await guardrail.apply_guardrail(inputs=inputs, request_data=request_data, input_type="request")
            assert result is inputs

    (record,) = _guardrail_records(request_data)
    assert record["guardrail_status"] == "guardrail_failed_to_respond"
    assert record["guardrail_response"]["status_code"] == 503
    assert record["guardrail_response"]["fallback_on_error"] == fallback_on_error


@pytest.mark.asyncio
async def test_response_redaction_merges_across_overlapping_windows() -> None:
    respond, scanned = _engine(redact={"SECRET": "******", "private": "*******"})
    prefix = "a" * 14
    content = f"{prefix}SECRETprivate"
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        result, request_data = await _apply(
            _guardrail(async_handler=client, response_content_chunk_size_bytes=20, stream_overlap_size=6),
            {"texts": [content], "model": "gpt-5.4"},
            "response",
        )

    assert scanned == [f"{prefix}SECRET", "SECRETprivate"]
    assert result["texts"] == [f"{prefix}*************"]
    (record,) = _guardrail_records(request_data)
    assert record["guardrail_status"] == "success"
    assert record["masked_entity_count"] == {"PII": 1}


def _redacted_choice(content: str) -> httpx.Response:
    return httpx.Response(
        200, json={"action": "allow", "redactedRequest": {"choices": [{"message": {"content": content}}]}}
    )


@pytest.mark.asyncio
async def test_later_overlap_scan_cannot_undo_an_earlier_redaction() -> None:
    """Window two sees ``SECRET`` in its overlap unmasked, flags only ``tail``, and must not restore ``SECRET``."""
    responses = iter((_redacted_choice("aaaaa******"), _redacted_choice("SECRET####")))

    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: next(responses))) as client:
        result, _ = await _apply(
            _guardrail(async_handler=client, response_content_chunk_size_bytes=11, stream_overlap_size=6),
            {"texts": ["aaaaaSECRETtail"]},
            "response",
        )

    assert result["texts"] == ["aaaaa******####"]


@pytest.mark.asyncio
async def test_response_block_in_a_later_window_stops_scanning() -> None:
    respond, scanned = _engine(block_on="zzz")
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        request_data: dict = {"metadata": {}}
        with pytest.raises(GuardrailRaisedException, match="policy"):
            await _guardrail(
                async_handler=client, response_content_chunk_size_bytes=5, stream_overlap_size=0
            ).apply_guardrail(
                inputs={"texts": ["aaaaabbbbbzzzzzccccc"]},
                request_data=request_data,
                input_type="response",
            )

    assert scanned == ["aaaaa", "bbbbb", "zzzzz"]
    assert _guardrail_records(request_data)[0]["guardrail_status"] == "guardrail_intervened"


@pytest.mark.asyncio
async def test_response_scans_every_choice() -> None:
    respond, scanned = _engine(redact={"SECRET": "******"})
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        result, _ = await _apply(
            _guardrail(async_handler=client),
            {"texts": ["first SECRET", "second SECRET"]},
            "response",
        )

    assert scanned == ["first SECRET", "second SECRET"]
    assert result["texts"] == ["first ******", "second ******"]


@pytest.mark.asyncio
async def test_unmergeable_response_redaction_blocks_instead_of_leaking() -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"action": "allow", "redactedRequest": {"choices": [{"message": {"content": "short"}}]}},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        request_data: dict = {"metadata": {}}
        with pytest.raises(GuardrailRaisedException, match="redaction"):
            await _guardrail(async_handler=client, fallback_on_error="allow").apply_guardrail(
                inputs={"texts": ["a much longer sensitive response"]},
                request_data=request_data,
                input_type="response",
            )

    assert _guardrail_records(request_data)[-1]["guardrail_status"] == "guardrail_failed_to_respond"


@pytest.mark.asyncio
async def test_request_redaction_flows_through_the_chat_completions_handler() -> None:
    respond, _ = _engine(redact={"a@b.com": "[EMAIL]"})
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        data: dict = {
            "model": "gpt-5.4",
            "messages": [
                {"role": "system", "content": "be terse"},
                {"role": "user", "content": [{"type": "text", "text": "mail a@b.com"}]},
            ],
            "metadata": {},
        }
        out = await OpenAIChatCompletionsHandler().process_input_messages(
            data=data, guardrail_to_apply=_guardrail(async_handler=client)
        )

    assert out["messages"][0]["content"] == "be terse"
    assert out["messages"][1]["content"][0]["text"] == "mail [EMAIL]"


@pytest.mark.parametrize(
    ("content", "chunk_size_bytes", "overlap", "expected"),
    [
        ("abcdéX", 5, 0, ["abcd", "éX"]),
        ("abcéX", 5, 0, ["abcé", "X"]),
        ("abcdéXYZ", 5, 2, ["abcd", "cdéX", "éXYZ"]),
        ("abcdef", 5, 100, ["abcde", "bcdef"]),
        ("", 5, 2, []),
        ("é", 1, 0, ["é"]),
    ],
)
def test_utf8_windows_respect_byte_limits_and_character_overlap(
    content: str, chunk_size_bytes: int, overlap: int, expected: Sequence[str]
) -> None:
    windows = utf8_windows(content, chunk_size_bytes=chunk_size_bytes, overlap_chars=overlap)

    assert [window.text for window in windows] == list(expected)
    assert all(content[window.start : window.start + len(window.text)] == window.text for window in windows)


def _logged_call(user_text: str, assistant_text: str) -> tuple[dict, ModelResponse]:
    response = ModelResponse(choices=[Choices(message=Message(role="assistant", content=assistant_text))])
    kwargs: dict = {
        "model": "gpt-5.4",
        "messages": [{"role": "user", "content": user_text}],
        "litellm_call_id": "call-1",
        "litellm_params": {"metadata": {}},
        "optional_params": {},
        "standard_logging_object": {"guardrail_information": None},
    }
    return kwargs, response


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("scope", "expected_scans"),
    [
        ("request", ["user a@b.com"]),
        ("response", ["assistant SECRET"]),
        ("both", ["user a@b.com", "assistant SECRET"]),
    ],
)
async def test_logging_only_scan_scope_selects_which_side_is_scanned(
    scope: Literal["request", "response", "both"], expected_scans: Sequence[str]
) -> None:
    respond, scanned = _engine(redact={"a@b.com": "[EMAIL]", "SECRET": "******"})
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        guardrail = _guardrail(
            async_handler=client, logging_only_scan=scope, event_hook=GuardrailEventHooks.logging_only
        )
        kwargs, response = _logged_call("user a@b.com", "assistant SECRET")
        out_kwargs, out_response = await guardrail.async_logging_hook(kwargs, response, CallTypes.acompletion.value)

    assert scanned == list(expected_scans)
    assert out_kwargs["messages"] == [{"role": "user", "content": "user a@b.com"}]
    assert out_response is response
    assert response.choices[0].message.content == "assistant SECRET"
    entries = out_kwargs["standard_logging_object"]["guardrail_information"]
    assert [entry["guardrail_status"] for entry in entries] == ["success"] * len(expected_scans)
    assert all(entry["guardrail_mode"] == "logging_only" for entry in entries)
