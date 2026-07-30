"""
Unit tests for the Veto guardrail integration.

Network calls are always mocked. Exercises the verdict → hook-action
mapping (allow / redact / block) across the pre-call, moderation, and
post-call hooks plus the text-in/text-out ``apply_guardrail`` surface.
"""

from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from fastapi.exceptions import HTTPException

from litellm.proxy.guardrails.guardrail_hooks.veto.veto import VetoGuardrail
from litellm.types.utils import (
    Choices,
    Delta,
    Message,
    ModelResponse,
    ModelResponseStream,
    StreamingChoices,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def veto_guardrail():
    return VetoGuardrail(
        api_base="https://api.test.vetocheck.local",
        api_key="vt_live_test_key",
        guardrail_name="test-veto",
        event_hook="pre_call",
        default_on=True,
    )


def _verdict(
    action: str, redacted: str = "", findings=None, degraded=None
) -> MagicMock:
    """A mock HTTP response whose .json() returns a Veto verdict."""
    mock = MagicMock()
    mock.json.return_value = {
        "allowed": action != "block",
        "action": action,
        "findings": findings or [],
        "redacted": redacted,
        "latency_ms": 1.0,
        "degraded": degraded or [],
    }
    return mock


@contextmanager
def _patch_check(guardrail, verdict_mock):
    """Patch the shared litellm async HTTP client that _check fetches, and
    yield the AsyncMock standing in for the handler's .post."""
    post_mock = AsyncMock(return_value=verdict_mock)
    handler = MagicMock()
    handler.post = post_mock
    with patch(
        "litellm.proxy.guardrails.guardrail_hooks.veto.veto.get_async_httpx_client",
        return_value=handler,
    ):
        yield post_mock


def _model_response(content: str) -> ModelResponse:
    return ModelResponse(
        choices=[Choices(message=Message(content=content, role="assistant"))]
    )


def _stream_chunk(text: str) -> ModelResponseStream:
    return ModelResponseStream(
        model="gpt-x",
        choices=[StreamingChoices(index=0, delta=Delta(content=text))],
    )


async def _aiter(items):
    for it in items:
        yield it


async def _collect(agen):
    return [c async for c in agen]


@contextmanager
def _patch_stream(guardrail, responses):
    """Patch the shared client; ``post`` returns each item in ``responses``
    in order — a verdict mock to return, or an Exception instance to raise
    (drives the fail-closed path)."""
    post_mock = AsyncMock(side_effect=responses)
    handler = MagicMock()
    handler.post = post_mock
    with patch(
        "litellm.proxy.guardrails.guardrail_hooks.veto.veto.get_async_httpx_client",
        return_value=handler,
    ):
        yield post_mock


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class TestVetoConfiguration:
    def test_default_api_base(self):
        g = VetoGuardrail(api_key="vt_live_x")
        assert g.api_base == "https://api.vetocheck.com"

    def test_strips_trailing_slash(self):
        g = VetoGuardrail(api_key="vt_live_x", api_base="https://custom.local/")
        assert g.api_base == "https://custom.local"

    def test_default_categories(self):
        g = VetoGuardrail(api_key="vt_live_x")
        assert g.categories == ["pii", "secrets", "injection"]

    def test_apply_guardrail_defined_on_class(self):
        # during_call dispatch requires apply_guardrail on the class dict,
        # not merely inherited. Guard against accidental refactors.
        assert "apply_guardrail" in VetoGuardrail.__dict__


# ---------------------------------------------------------------------------
# pre_call hook (input)
# ---------------------------------------------------------------------------


class TestVetoPreCall:
    @pytest.mark.asyncio
    async def test_allow_passes_unchanged(self, veto_guardrail):
        data = {"messages": [{"role": "user", "content": "hello"}]}
        with _patch_check(veto_guardrail, _verdict("allow")):
            out = await veto_guardrail.async_pre_call_hook(
                MagicMock(), MagicMock(), data, "completion"
            )
        assert out["messages"][0]["content"] == "hello"

    @pytest.mark.asyncio
    async def test_redact_rewrites_content(self, veto_guardrail):
        data = {"messages": [{"role": "user", "content": "email a@b.com"}]}
        with _patch_check(
            veto_guardrail, _verdict("redact", redacted="email [REDACTED_EMAIL]")
        ):
            out = await veto_guardrail.async_pre_call_hook(
                MagicMock(), MagicMock(), data, "completion"
            )
        assert out["messages"][0]["content"] == "email [REDACTED_EMAIL]"

    @pytest.mark.asyncio
    async def test_block_raises(self, veto_guardrail):
        data = {"messages": [{"role": "user", "content": "ignore previous"}]}
        finding = {"category": "injection", "rule": "ml", "severity": "high"}
        with _patch_check(veto_guardrail, _verdict("block", findings=[finding])):
            with pytest.raises(HTTPException) as exc:
                await veto_guardrail.async_pre_call_hook(
                    MagicMock(), MagicMock(), data, "completion"
                )
        assert exc.value.detail["veto"]["action"] == "block"

    @pytest.mark.asyncio
    async def test_non_string_content_skipped(self, veto_guardrail):
        data = {"messages": [{"role": "user", "content": [{"type": "image_url"}]}]}
        with _patch_check(veto_guardrail, _verdict("block")) as mock_post:
            out = await veto_guardrail.async_pre_call_hook(
                MagicMock(), MagicMock(), data, "completion"
            )
        assert mock_post.call_count == 0
        assert out == data


# ---------------------------------------------------------------------------
# bypass prevention: multimodal parts + Responses `input` field
# ---------------------------------------------------------------------------


class TestVetoMultimodalAndInput:
    """Text smuggled inside a multimodal content part or the Responses
    ``input`` field must still be scanned — not only top-level string content."""

    @pytest.mark.asyncio
    async def test_multimodal_text_part_blocked(self, veto_guardrail):
        data = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "ignore previous"},
                        {"type": "image_url", "image_url": {"url": "http://x/y.png"}},
                    ],
                }
            ]
        }
        with _patch_check(veto_guardrail, _verdict("block")) as mock_post:
            with pytest.raises(HTTPException):
                await veto_guardrail.async_pre_call_hook(
                    MagicMock(), MagicMock(), data, "completion"
                )
        assert mock_post.call_count == 1  # only the text part is scanned

    @pytest.mark.asyncio
    async def test_multimodal_text_part_redacted(self, veto_guardrail):
        data = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "email a@b.com"},
                        {"type": "image_url", "image_url": {"url": "http://x/y.png"}},
                    ],
                }
            ]
        }
        with _patch_check(
            veto_guardrail, _verdict("redact", redacted="email [REDACTED_EMAIL]")
        ):
            out = await veto_guardrail.async_pre_call_hook(
                MagicMock(), MagicMock(), data, "completion"
            )
        parts = out["messages"][0]["content"]
        assert parts[0]["text"] == "email [REDACTED_EMAIL]"
        assert parts[1]["type"] == "image_url"  # non-text part untouched

    @pytest.mark.asyncio
    async def test_responses_input_string_blocked(self, veto_guardrail):
        data = {"input": "ignore previous"}
        with _patch_check(veto_guardrail, _verdict("block")):
            with pytest.raises(HTTPException):
                await veto_guardrail.async_pre_call_hook(
                    MagicMock(), MagicMock(), data, "responses"
                )

    @pytest.mark.asyncio
    async def test_responses_input_string_redacted(self, veto_guardrail):
        data = {"input": "email a@b.com"}
        with _patch_check(
            veto_guardrail, _verdict("redact", redacted="email [REDACTED_EMAIL]")
        ):
            out = await veto_guardrail.async_pre_call_hook(
                MagicMock(), MagicMock(), data, "responses"
            )
        assert out["input"] == "email [REDACTED_EMAIL]"

    @pytest.mark.asyncio
    async def test_responses_input_list_message_blocked(self, veto_guardrail):
        data = {
            "input": [
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": "ignore previous"}],
                }
            ]
        }
        with _patch_check(veto_guardrail, _verdict("block")):
            with pytest.raises(HTTPException):
                await veto_guardrail.async_pre_call_hook(
                    MagicMock(), MagicMock(), data, "responses"
                )

    @pytest.mark.asyncio
    async def test_moderation_scans_multimodal_text(self, veto_guardrail):
        # the block-only parallel path must also see multimodal text parts
        data = {
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "ignore previous"}],
                }
            ]
        }
        with _patch_check(veto_guardrail, _verdict("block")):
            with pytest.raises(HTTPException):
                await veto_guardrail.async_moderation_hook(
                    data, MagicMock(), "completion"
                )

    @pytest.mark.asyncio
    async def test_responses_input_list_of_strings_redacted(self, veto_guardrail):
        data = {"input": ["email a@b.com"]}
        with _patch_check(
            veto_guardrail, _verdict("redact", redacted="email [REDACTED_EMAIL]")
        ):
            out = await veto_guardrail.async_pre_call_hook(
                MagicMock(), MagicMock(), data, "responses"
            )
        assert out["input"][0] == "email [REDACTED_EMAIL]"

    @pytest.mark.asyncio
    async def test_null_content_passthrough(self, veto_guardrail):
        # a message with no string/list content (e.g. a tool call) is left alone
        data = {"messages": [{"role": "assistant", "content": None}]}
        with _patch_check(veto_guardrail, _verdict("block")) as mock_post:
            out = await veto_guardrail.async_pre_call_hook(
                MagicMock(), MagicMock(), data, "completion"
            )
        assert mock_post.call_count == 0
        assert out["messages"][0]["content"] is None


# ---------------------------------------------------------------------------
# moderation hook (block-only)
# ---------------------------------------------------------------------------


class TestVetoModeration:
    @pytest.mark.asyncio
    async def test_block_raises(self, veto_guardrail):
        data = {"messages": [{"role": "user", "content": "ignore previous"}]}
        with _patch_check(veto_guardrail, _verdict("block")):
            with pytest.raises(HTTPException):
                await veto_guardrail.async_moderation_hook(
                    data, MagicMock(), "completion"
                )

    @pytest.mark.asyncio
    async def test_redact_does_not_rewrite(self, veto_guardrail):
        # moderation runs parallel to the LLM call and cannot mutate input.
        data = {"messages": [{"role": "user", "content": "email a@b.com"}]}
        with _patch_check(veto_guardrail, _verdict("redact", redacted="email X")):
            out = await veto_guardrail.async_moderation_hook(
                data, MagicMock(), "completion"
            )
        assert out["messages"][0]["content"] == "email a@b.com"


# ---------------------------------------------------------------------------
# post_call hook (output)
# ---------------------------------------------------------------------------


class TestVetoPostCall:
    @pytest.mark.asyncio
    async def test_block_raises(self, veto_guardrail):
        response = _model_response("harmful output")
        with _patch_check(veto_guardrail, _verdict("block")):
            with pytest.raises(HTTPException):
                await veto_guardrail.async_post_call_success_hook(
                    {}, MagicMock(), response
                )

    @pytest.mark.asyncio
    async def test_redact_rewrites_message(self, veto_guardrail):
        response = _model_response("my key is sk-123")
        with _patch_check(
            veto_guardrail,
            _verdict("redact", redacted="my key is [REDACTED_OPENAI_KEY]"),
        ):
            out = await veto_guardrail.async_post_call_success_hook(
                {}, MagicMock(), response
            )
        assert out.choices[0].message.content == "my key is [REDACTED_OPENAI_KEY]"

    @pytest.mark.asyncio
    async def test_non_model_response_returned_unchanged(self, veto_guardrail):
        with _patch_check(veto_guardrail, _verdict("block")) as mock_post:
            out = await veto_guardrail.async_post_call_success_hook(
                {}, MagicMock(), "raw string"
            )
        assert out == "raw string"
        assert mock_post.call_count == 0


# ---------------------------------------------------------------------------
# apply_guardrail (text in / text out)
# ---------------------------------------------------------------------------


class TestVetoApplyGuardrail:
    @pytest.mark.asyncio
    async def test_allow_returns_input(self, veto_guardrail):
        with _patch_check(veto_guardrail, _verdict("allow")):
            out = await veto_guardrail.apply_guardrail(
                {"texts": ["hello"]}, {}, "request"
            )
        assert out["texts"] == ["hello"]

    @pytest.mark.asyncio
    async def test_redact_returns_masked(self, veto_guardrail):
        with _patch_check(veto_guardrail, _verdict("redact", redacted="masked")):
            out = await veto_guardrail.apply_guardrail(
                {"texts": ["secret"]}, {}, "request"
            )
        assert out["texts"] == ["masked"]

    @pytest.mark.asyncio
    async def test_block_raises(self, veto_guardrail):
        with _patch_check(veto_guardrail, _verdict("block")):
            with pytest.raises(HTTPException):
                await veto_guardrail.apply_guardrail(
                    {"texts": ["ignore previous"]}, {}, "request"
                )

    @pytest.mark.asyncio
    async def test_empty_text_not_scanned(self, veto_guardrail):
        with _patch_check(veto_guardrail, _verdict("block")) as mock_post:
            out = await veto_guardrail.apply_guardrail(
                {"texts": ["   "]}, {}, "request"
            )
        assert out["texts"] == ["   "]
        assert mock_post.call_count == 0


# ---------------------------------------------------------------------------
# request shape
# ---------------------------------------------------------------------------


class TestVetoRequest:
    @pytest.mark.asyncio
    async def test_bearer_auth_and_payload(self, veto_guardrail):
        data = {"messages": [{"role": "user", "content": "scan me"}]}
        verdict = _verdict("allow")
        with _patch_check(veto_guardrail, verdict) as mock_post:
            await veto_guardrail.async_pre_call_hook(
                MagicMock(), MagicMock(), data, "completion"
            )
        assert mock_post.call_args.kwargs["headers"]["Authorization"] == (
            "Bearer vt_live_test_key"
        )
        sent = mock_post.call_args.kwargs["json"]
        assert sent["text"] == "scan me"
        assert sent["categories"] == ["pii", "secrets", "injection"]
        assert mock_post.call_args.args[0].endswith("/v1/check")

    @pytest.mark.asyncio
    async def test_no_api_key_omits_auth_header(self):
        g = VetoGuardrail(guardrail_name="no-key", event_hook="pre_call")
        data = {"messages": [{"role": "user", "content": "scan me"}]}
        with _patch_check(g, _verdict("allow")) as mock_post:
            await g.async_pre_call_hook(MagicMock(), MagicMock(), data, "completion")
        assert "Authorization" not in mock_post.call_args.kwargs["headers"]

    @pytest.mark.asyncio
    async def test_gateway_error_fails_closed(self, veto_guardrail):
        # a non-2xx from the gateway must raise (fail closed), never silently
        # pass unscanned text through.
        err_resp = MagicMock()
        err_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "500", request=MagicMock(), response=MagicMock()
        )
        with _patch_check(veto_guardrail, err_resp):
            with pytest.raises(httpx.HTTPStatusError):
                await veto_guardrail.apply_guardrail(
                    {"texts": ["scan me"]}, {}, "request"
                )


# ---------------------------------------------------------------------------
# Config model (UI / typed params)
# ---------------------------------------------------------------------------


class TestVetoConfigModel:
    def test_get_config_model_returns_veto_model(self):
        from litellm.types.proxy.guardrails.guardrail_hooks.veto import (
            VetoGuardrailConfigModel,
        )

        assert VetoGuardrail.get_config_model() is VetoGuardrailConfigModel

    def test_ui_friendly_name(self):
        from litellm.types.proxy.guardrails.guardrail_hooks.veto import (
            VetoGuardrailConfigModel,
        )

        assert VetoGuardrailConfigModel.ui_friendly_name() == "Veto"

    def test_config_model_fields(self):
        from litellm.types.proxy.guardrails.guardrail_hooks.veto import (
            VetoGuardrailConfigModel,
        )

        fields = VetoGuardrailConfigModel.model_fields
        assert {"api_key", "api_base"}.issubset(fields.keys())


# ---------------------------------------------------------------------------
# streaming output hook (SPEC §3.5)
# ---------------------------------------------------------------------------


class TestVetoStreaming:
    @pytest.mark.asyncio
    async def test_allow_passes_through(self, veto_guardrail):
        # total text under the buffer → one final=true POST, all chunks yielded
        chunks = [_stream_chunk("hel"), _stream_chunk("lo")]
        with _patch_stream(veto_guardrail, [_verdict("allow")]) as post:
            out = await _collect(
                veto_guardrail.async_post_call_streaming_iterator_hook(
                    MagicMock(), _aiter(chunks), {"model": "gpt-x"}
                )
            )
        assert [c.choices[0].delta.content for c in out] == ["hel", "lo"]
        assert post.call_count == 1
        assert post.call_args.kwargs["json"]["final"] is True

    @pytest.mark.asyncio
    async def test_midstream_block_drops_held_buffer(self, veto_guardrail):
        # small buffer → first chunk triggers a flush; block drops the held
        # chunk (never yielded) and closes with a content_filter terminal.
        veto_guardrail.buffer_tokens = 1  # buffer_chars = 4
        chunks = [_stream_chunk("AKIA"), _stream_chunk("more")]
        with _patch_stream(veto_guardrail, [_verdict("block")]) as post:
            out = await _collect(
                veto_guardrail.async_post_call_streaming_iterator_hook(
                    MagicMock(), _aiter(chunks), {"model": "gpt-x"}
                )
            )
        assert len(out) == 1
        assert out[0].choices[0].finish_reason == "content_filter"
        assert post.call_count == 1

    @pytest.mark.asyncio
    async def test_final_block(self, veto_guardrail):
        # under buffer → block surfaces only at EOF; held tail is dropped.
        chunks = [_stream_chunk("hi")]
        with _patch_stream(veto_guardrail, [_verdict("block")]):
            out = await _collect(
                veto_guardrail.async_post_call_streaming_iterator_hook(
                    MagicMock(), _aiter(chunks), {"model": "gpt-x"}
                )
            )
        assert out[-1].choices[0].finish_reason == "content_filter"
        assert all(
            getattr(c.choices[0], "delta", None) is None
            or c.choices[0].delta.content != "hi"
            for c in out
        )

    @pytest.mark.asyncio
    async def test_fail_closed_on_gateway_error(self, veto_guardrail):
        # a thrown gateway call must close the stream, not pass tokens through
        chunks = [_stream_chunk("hi")]
        with _patch_stream(veto_guardrail, [RuntimeError("boom")]):
            out = await _collect(
                veto_guardrail.async_post_call_streaming_iterator_hook(
                    MagicMock(), _aiter(chunks), {"model": "gpt-x"}
                )
            )
        assert out[-1].choices[0].finish_reason == "content_filter"

    @pytest.mark.asyncio
    async def test_buffer_flush_releases_and_params(self, veto_guardrail):
        veto_guardrail.buffer_tokens = 1  # buffer_chars = 4
        chunks = [_stream_chunk("aaaa"), _stream_chunk("bb")]
        with _patch_stream(
            veto_guardrail, [_verdict("allow"), _verdict("allow")]
        ) as post:
            out = await _collect(
                veto_guardrail.async_post_call_streaming_iterator_hook(
                    MagicMock(), _aiter(chunks), {"model": "gpt-x"}
                )
            )
        assert [c.choices[0].delta.content for c in out] == ["aaaa", "bb"]
        assert post.call_count == 2  # one mid-stream flush + one final
        p0 = post.call_args_list[0].kwargs["params"]
        p1 = post.call_args_list[1].kwargs["params"]
        assert p0["stream_id"] == p1["stream_id"]  # stable across chunks
        assert p0["buffer"] == 1
        assert post.call_args_list[0].kwargs["json"]["final"] is False
        assert post.call_args_list[1].kwargs["json"]["final"] is True
        assert post.call_args_list[1].kwargs["json"]["chunk_index"] == 1

    @pytest.mark.asyncio
    async def test_eof_redact_treated_as_allow(self, veto_guardrail):
        # redact is EOF-only and can't unsay yielded tokens → streaming
        # honors only block; a final redact verdict releases the tail as-is.
        chunks = [_stream_chunk("email a@b.com")]
        with _patch_stream(veto_guardrail, [_verdict("redact", redacted="email X")]):
            out = await _collect(
                veto_guardrail.async_post_call_streaming_iterator_hook(
                    MagicMock(), _aiter(chunks), {"model": "gpt-x"}
                )
            )
        assert [c.choices[0].delta.content for c in out] == ["email a@b.com"]

    @pytest.mark.asyncio
    async def test_non_chat_chunk_passthrough(self, veto_guardrail):
        # raw bytes (Anthropic native SSE) pass through untouched; chat
        # chunks are still buffered + scanned.
        chunks = [b"raw bytes", _stream_chunk("hi")]
        with _patch_stream(veto_guardrail, [_verdict("allow")]) as post:
            out = await _collect(
                veto_guardrail.async_post_call_streaming_iterator_hook(
                    MagicMock(), _aiter(chunks), {"model": "gpt-x"}
                )
            )
        assert out[0] == b"raw bytes"
        assert out[-1].choices[0].delta.content == "hi"
        assert post.call_count == 1
