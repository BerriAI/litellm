"""
Unit tests for the Veto guardrail integration.

Network calls are always mocked. Exercises the verdict → hook-action
mapping (allow / redact / block) across the pre-call, moderation, and
post-call hooks plus the text-in/text-out ``apply_guardrail`` surface.
"""

from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from fastapi.exceptions import HTTPException

from litellm.proxy.guardrails.guardrail_hooks.veto.veto import VetoGuardrail
from litellm.types.utils import Choices, Message, ModelResponse

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
            assert await veto_guardrail.apply_guardrail("hello") == "hello"

    @pytest.mark.asyncio
    async def test_redact_returns_masked(self, veto_guardrail):
        with _patch_check(veto_guardrail, _verdict("redact", redacted="masked")):
            assert await veto_guardrail.apply_guardrail("secret") == "masked"

    @pytest.mark.asyncio
    async def test_block_raises(self, veto_guardrail):
        with _patch_check(veto_guardrail, _verdict("block")):
            with pytest.raises(HTTPException):
                await veto_guardrail.apply_guardrail("ignore previous")


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
