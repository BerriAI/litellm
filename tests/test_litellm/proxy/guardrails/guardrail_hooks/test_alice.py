import os
from unittest.mock import AsyncMock

import pytest
from httpx import Request, Response

import litellm
from litellm.exceptions import GuardrailRaisedException
from litellm.proxy.guardrails.guardrail_hooks.alice.alice import (
    GUARDRAIL_NAME,
    AliceGuardrail,
    AliceGuardrailMissingSecrets,
)
from litellm.proxy.guardrails.init_guardrails import init_guardrails_v2


def _guardrail(**overrides) -> AliceGuardrail:
    params = {
        "api_key": "test-key",
        "guardrail_name": "alice",
        "event_hook": "pre_call",
    }
    params.update(overrides)
    return AliceGuardrail(**params)


def _authenticated(**metadata) -> dict:
    """What the proxy writes for an authenticated virtual key."""
    return {"metadata": {"user_api_key_metadata": metadata}}


def _response(payload: dict, status_code: int = 200) -> Response:
    return Response(
        status_code=status_code,
        json=payload,
        request=Request("POST", "https://api.alice.io/v2/evaluate/message"),
    )


def test_alice_guardrail_config(monkeypatch: pytest.MonkeyPatch):
    """Should register through init_guardrails_v2 like any other provider."""
    monkeypatch.setattr(litellm, "guardrail_name_config_map", {})
    monkeypatch.setenv("ALICE_API_KEY", "test-key")

    init_guardrails_v2(
        all_guardrails=[
            {
                "guardrail_name": "alice",
                "litellm_params": {
                    "guardrail": "alice",
                    "mode": "pre_call",
                    "default_on": True,
                },
            }
        ],
        config_file_path="",
    )


class TestAliceGuardrailInitialization:
    def setup_method(self):
        for key in ("ALICE_API_KEY", "ALICE_API_BASE"):
            os.environ.pop(key, None)

    def test_missing_api_key_raises(self):
        with pytest.raises(AliceGuardrailMissingSecrets, match="API key"):
            AliceGuardrail(guardrail_name="alice", event_hook="pre_call")

    def test_reads_credentials_from_environment(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("ALICE_API_KEY", "env-key")
        monkeypatch.setenv("ALICE_API_BASE", "https://env.alice.test")

        guardrail = AliceGuardrail(guardrail_name="alice", event_hook="pre_call")

        assert guardrail.alice_api_key == "env-key"
        assert guardrail.api_base == "https://env.alice.test/v2/evaluate/message"

    def test_defaults_the_api_base(self):
        assert _guardrail().api_base == "https://api.alice.io/v2/evaluate/message"

    def test_trailing_slash_does_not_double_up(self):
        guardrail = _guardrail(api_base="https://api.alice.io/")

        assert guardrail.api_base == "https://api.alice.io/v2/evaluate/message"


class TestAliceApplicationResolution:
    """The application is named by the authenticated key, never by the caller."""

    @pytest.mark.asyncio
    async def test_reads_the_app_id_from_key_metadata(self):
        guardrail = _guardrail()
        guardrail.async_handler.post = AsyncMock(return_value=_response({"action": "", "detections": [], "errors": []}))

        await guardrail.apply_guardrail(
            inputs={"texts": ["hello"]},
            request_data=_authenticated(alice_app_id="payments-bot"),
            input_type="request",
        )

        assert guardrail.async_handler.post.call_args.kwargs["json"]["app_id"] == "payments-bot"

    @pytest.mark.asyncio
    async def test_ignores_an_app_id_the_caller_supplied(self):
        guardrail = _guardrail()
        guardrail.async_handler.post = AsyncMock(return_value=_response({"action": "", "detections": [], "errors": []}))
        request_data = {
            "metadata": {
                "user_api_key_metadata": {"alice_app_id": "payments-bot"},
                "alice_app_id": "forged",
            }
        }

        await guardrail.apply_guardrail(inputs={"texts": ["hello"]}, request_data=request_data, input_type="request")

        assert guardrail.async_handler.post.call_args.kwargs["json"]["app_id"] == "payments-bot"

    @pytest.mark.asyncio
    async def test_falls_back_to_the_key_alias(self):
        guardrail = _guardrail()
        guardrail.async_handler.post = AsyncMock(return_value=_response({"action": "", "detections": [], "errors": []}))
        request_data = {"metadata": {"user_api_key_metadata": {}, "user_api_key_alias": "billing-app"}}

        await guardrail.apply_guardrail(inputs={"texts": ["hello"]}, request_data=request_data, input_type="request")

        assert guardrail.async_handler.post.call_args.kwargs["json"]["app_id"] == "billing-app"

    @pytest.mark.asyncio
    async def test_refuses_when_no_key_names_an_application(self):
        guardrail = _guardrail()
        guardrail.async_handler.post = AsyncMock()

        with pytest.raises(GuardrailRaisedException, match="No Alice application"):
            await guardrail.apply_guardrail(inputs={"texts": ["hello"]}, request_data={}, input_type="request")

        guardrail.async_handler.post.assert_not_called()


class TestAliceVerdicts:
    @pytest.mark.asyncio
    async def test_allows_when_no_policy_matched(self):
        guardrail = _guardrail()
        guardrail.async_handler.post = AsyncMock(return_value=_response({"action": "", "detections": [], "errors": []}))

        result = await guardrail.apply_guardrail(
            inputs={"texts": ["hello"]},
            request_data=_authenticated(alice_app_id="app"),
            input_type="request",
        )

        assert result["texts"] == ["hello"]

    @pytest.mark.asyncio
    async def test_blocks_and_surfaces_the_configured_message(self):
        guardrail = _guardrail()
        guardrail.async_handler.post = AsyncMock(
            return_value=_response(
                {
                    "action": "BLOCK",
                    "action_text": "Blocked by your organization's policy",
                    "detections": [{"type": "self_harm", "score": 0.97}],
                    "errors": [],
                }
            )
        )

        with pytest.raises(GuardrailRaisedException) as error:
            await guardrail.apply_guardrail(
                inputs={"texts": ["bad"]},
                request_data=_authenticated(alice_app_id="app"),
                input_type="request",
            )

        assert "Blocked by your organization's policy" in str(error.value)

    @pytest.mark.asyncio
    async def test_masks_by_substituting_the_redacted_text(self):
        guardrail = _guardrail()
        guardrail.async_handler.post = AsyncMock(
            return_value=_response({"action": "MASK", "action_text": "my ssn is ***", "detections": [], "errors": []})
        )

        result = await guardrail.apply_guardrail(
            inputs={"texts": ["my ssn is 123-45-6789"]},
            request_data=_authenticated(alice_app_id="app"),
            input_type="request",
        )

        assert result["texts"] == ["my ssn is ***"]

    @pytest.mark.asyncio
    async def test_a_mask_with_nothing_to_substitute_blocks(self):
        guardrail = _guardrail()
        guardrail.async_handler.post = AsyncMock(
            return_value=_response({"action": "MASK", "detections": [], "errors": []})
        )

        with pytest.raises(GuardrailRaisedException):
            await guardrail.apply_guardrail(
                inputs={"texts": ["secret"]},
                request_data=_authenticated(alice_app_id="app"),
                input_type="request",
            )

    @pytest.mark.asyncio
    async def test_detect_allows_and_leaves_the_text_alone(self):
        guardrail = _guardrail()
        guardrail.async_handler.post = AsyncMock(
            return_value=_response(
                {
                    "action": "DETECT",
                    "correlation_id": "c1",
                    "detections": [{"type": "profanity", "score": 0.6}],
                    "errors": [],
                }
            )
        )

        result = await guardrail.apply_guardrail(
            inputs={"texts": ["mild"]},
            request_data=_authenticated(alice_app_id="app"),
            input_type="request",
        )

        assert result["texts"] == ["mild"]

    @pytest.mark.asyncio
    async def test_a_verdict_reporting_errors_is_a_failure_not_a_pass(self):
        guardrail = _guardrail()
        guardrail.async_handler.post = AsyncMock(
            return_value=_response({"action": "", "detections": [], "errors": [{"type": "timeout"}]})
        )

        with pytest.raises(GuardrailRaisedException, match="reported errors"):
            await guardrail.apply_guardrail(
                inputs={"texts": ["hello"]},
                request_data=_authenticated(alice_app_id="app"),
                input_type="request",
            )

    @pytest.mark.asyncio
    async def test_sends_a_completion_as_a_response(self):
        guardrail = _guardrail()
        guardrail.async_handler.post = AsyncMock(return_value=_response({"action": "", "detections": [], "errors": []}))

        await guardrail.apply_guardrail(
            inputs={"texts": ["the answer"]},
            request_data=_authenticated(alice_app_id="app"),
            input_type="response",
        )

        assert guardrail.async_handler.post.call_args.kwargs["json"]["message_type"] == "response"

    @pytest.mark.asyncio
    async def test_no_texts_reaches_no_evaluation(self):
        guardrail = _guardrail()
        guardrail.async_handler.post = AsyncMock()

        result = await guardrail.apply_guardrail(inputs={"texts": []}, request_data={}, input_type="request")

        assert result == {"texts": []}
        guardrail.async_handler.post.assert_not_called()


class TestAliceUnreachable:
    @pytest.mark.asyncio
    async def test_fails_closed_by_default(self):
        import httpx

        guardrail = _guardrail()
        guardrail.async_handler.post = AsyncMock(side_effect=httpx.ConnectError("refused"))

        with pytest.raises(GuardrailRaisedException, match="unavailable"):
            await guardrail.apply_guardrail(
                inputs={"texts": ["hello"]},
                request_data=_authenticated(alice_app_id="app"),
                input_type="request",
            )

    @pytest.mark.asyncio
    async def test_fails_open_when_configured(self):
        import httpx

        guardrail = _guardrail(unreachable_fallback="fail_open")
        guardrail.async_handler.post = AsyncMock(side_effect=httpx.ConnectError("refused"))

        result = await guardrail.apply_guardrail(
            inputs={"texts": ["hello"]},
            request_data=_authenticated(alice_app_id="app"),
            input_type="request",
        )

        assert result["texts"] == ["hello"]


def test_config_model_is_exposed_for_the_ui():
    config_model = AliceGuardrail.get_config_model()

    assert config_model is not None
    assert config_model.ui_friendly_name() == "Alice by ActiveFence"


def test_guardrail_name_constant():
    assert GUARDRAIL_NAME == "alice"
