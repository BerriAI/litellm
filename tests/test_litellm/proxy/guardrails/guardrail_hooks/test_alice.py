import os
from copy import deepcopy
from unittest.mock import AsyncMock

import httpx
import pytest
from httpx import Request, Response

import litellm
from litellm.exceptions import GuardrailRaisedException
from litellm.proxy.guardrails.guardrail_hooks.alice.alice import (
    GUARDRAIL_NAME,
    AliceGuardrail,
    AliceGuardrailMissingSecrets,
    _json_safe,
)
from litellm.proxy.guardrails.init_guardrails import init_guardrails_v2


def _guardrail(**overrides) -> AliceGuardrail:
    params = {"api_key": "test-key", "guardrail_name": "alice", "event_hook": "pre_call"}
    params.update(overrides)
    return AliceGuardrail(**params)


def _verdict(payload: dict, status_code: int = 200) -> Response:
    return Response(
        status_code=status_code,
        json=payload,
        request=Request("POST", "https://api.alice.io/v2/evaluate/litellm"),
    )


def test_alice_guardrail_config(monkeypatch: pytest.MonkeyPatch):
    """Should register through init_guardrails_v2 like any other provider."""
    monkeypatch.setattr(litellm, "guardrail_name_config_map", {})
    monkeypatch.setenv("ALICE_API_KEY", "test-key")

    init_guardrails_v2(
        all_guardrails=[
            {
                "guardrail_name": "alice",
                "litellm_params": {"guardrail": "alice", "mode": "pre_call", "default_on": True},
            }
        ],
        config_file_path="",
    )

    registered = [cb for cb in litellm.callbacks if isinstance(cb, AliceGuardrail)]
    assert len(registered) == 1
    assert registered[0].guardrail_name == "alice"


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
        assert guardrail.api_base == "https://env.alice.test/v2/evaluate/litellm"

    def test_defaults_the_api_base(self):
        assert _guardrail().api_base == "https://api.alice.io/v2/evaluate/litellm"

    def test_trailing_slash_does_not_double_up(self):
        assert _guardrail(api_base="https://api.alice.io/").api_base == ("https://api.alice.io/v2/evaluate/litellm")


class TestAliceForwarding:
    """The hook's arguments cross the wire as they were received — nothing selected, nothing renamed."""

    @pytest.mark.asyncio
    async def test_forwards_the_hook_arguments_verbatim(self):
        guardrail = _guardrail()
        guardrail.async_handler.post = AsyncMock(return_value=_verdict({"verdict": "ALLOW", "categories": []}))
        inputs = {"texts": ["hello"], "structured_messages": [{"role": "user", "content": "hello"}]}
        request_data = {"model": "gpt-4o", "metadata": {"user_api_key_alias": "payments-bot"}}
        # Snapshot before the call: @log_guardrail_information writes its own entry into
        # request_data["metadata"] afterwards, so the original is no longer what was sent.
        sent_inputs = deepcopy(inputs)
        sent_request_data = deepcopy(request_data)

        await guardrail.apply_guardrail(inputs=inputs, request_data=request_data, input_type="request")

        body = guardrail.async_handler.post.call_args.kwargs["json"]
        assert body["input_type"] == "request"
        assert body["inputs"] == sent_inputs
        assert body["request_data"] == sent_request_data

    @pytest.mark.asyncio
    async def test_sends_the_credential(self):
        guardrail = _guardrail()
        guardrail.async_handler.post = AsyncMock(return_value=_verdict({"verdict": "ALLOW", "categories": []}))

        await guardrail.apply_guardrail(inputs={"texts": ["hi"]}, request_data={}, input_type="request")

        assert guardrail.async_handler.post.call_args.kwargs["headers"]["af-api-key"] == "test-key"

    @pytest.mark.asyncio
    async def test_marks_a_completion_as_a_response(self):
        guardrail = _guardrail()
        guardrail.async_handler.post = AsyncMock(return_value=_verdict({"verdict": "ALLOW", "categories": []}))

        await guardrail.apply_guardrail(inputs={"texts": ["answer"]}, request_data={}, input_type="response")

        assert guardrail.async_handler.post.call_args.kwargs["json"]["input_type"] == "response"

    @pytest.mark.asyncio
    async def test_no_texts_reaches_no_evaluation(self):
        guardrail = _guardrail()
        guardrail.async_handler.post = AsyncMock()

        result = await guardrail.apply_guardrail(inputs={"texts": []}, request_data={}, input_type="request")

        assert result == {"texts": []}
        guardrail.async_handler.post.assert_not_called()

    @pytest.mark.asyncio
    async def test_makes_exactly_one_attempt(self):
        guardrail = _guardrail(unreachable_fallback="fail_open")
        guardrail.async_handler.post = AsyncMock(side_effect=httpx.ConnectError("refused"))

        await guardrail.apply_guardrail(inputs={"texts": ["hi"]}, request_data={}, input_type="request")

        assert guardrail.async_handler.post.call_count == 1


class TestAliceVerdicts:
    @pytest.mark.asyncio
    async def test_allow_leaves_the_inputs_untouched(self):
        guardrail = _guardrail()
        guardrail.async_handler.post = AsyncMock(return_value=_verdict({"verdict": "ALLOW", "categories": []}))

        result = await guardrail.apply_guardrail(inputs={"texts": ["hello"]}, request_data={}, input_type="request")

        assert result["texts"] == ["hello"]

    @pytest.mark.asyncio
    async def test_block_surfaces_the_policy_message(self):
        guardrail = _guardrail()
        guardrail.async_handler.post = AsyncMock(
            return_value=_verdict(
                {
                    "verdict": "BLOCK",
                    "categories": ["self_harm"],
                    "correlation_id": "c1",
                    "message": "Blocked by your organization's policy",
                }
            )
        )

        with pytest.raises(GuardrailRaisedException) as error:
            await guardrail.apply_guardrail(inputs={"texts": ["bad"]}, request_data={}, input_type="request")

        assert "Blocked by your organization's policy" in str(error.value)

    @pytest.mark.asyncio
    async def test_block_without_a_message_still_blocks(self):
        guardrail = _guardrail()
        guardrail.async_handler.post = AsyncMock(return_value=_verdict({"verdict": "BLOCK", "categories": []}))

        with pytest.raises(GuardrailRaisedException):
            await guardrail.apply_guardrail(inputs={"texts": ["bad"]}, request_data={}, input_type="request")

    @pytest.mark.asyncio
    async def test_mask_substitutes_by_position(self):
        guardrail = _guardrail()
        guardrail.async_handler.post = AsyncMock(
            return_value=_verdict(
                {
                    "verdict": "MASK",
                    "categories": ["pii"],
                    "replacements": [{"index": 1, "text": "my ssn is ***"}],
                }
            )
        )

        result = await guardrail.apply_guardrail(
            inputs={"texts": ["untouched", "my ssn is 123-45-6789"]},
            request_data={},
            input_type="request",
        )

        assert result["texts"] == ["untouched", "my ssn is ***"]

    @pytest.mark.asyncio
    async def test_mask_that_lands_nowhere_blocks(self):
        """A mask that wrote nothing would let the text through under a verdict that said not to."""
        guardrail = _guardrail()
        guardrail.async_handler.post = AsyncMock(
            return_value=_verdict({"verdict": "MASK", "categories": [], "replacements": [{"index": 9, "text": "***"}]})
        )

        with pytest.raises(GuardrailRaisedException):
            await guardrail.apply_guardrail(inputs={"texts": ["hello"]}, request_data={}, input_type="request")

    @pytest.mark.asyncio
    async def test_mask_leaves_structured_messages_identical(self):
        """A new structured_messages object makes the translation layer skip the texts write-back."""
        guardrail = _guardrail()
        guardrail.async_handler.post = AsyncMock(
            return_value=_verdict({"verdict": "MASK", "categories": [], "replacements": [{"index": 0, "text": "***"}]})
        )
        messages = [{"role": "user", "content": "secret"}]

        result = await guardrail.apply_guardrail(
            inputs={"texts": ["secret"], "structured_messages": messages},
            request_data={},
            input_type="request",
        )

        assert result["structured_messages"] is messages

    @pytest.mark.asyncio
    async def test_detect_allows_and_leaves_the_text_alone(self):
        guardrail = _guardrail()
        guardrail.async_handler.post = AsyncMock(
            return_value=_verdict({"verdict": "DETECT", "categories": ["profanity"], "correlation_id": "c1"})
        )

        result = await guardrail.apply_guardrail(inputs={"texts": ["mild"]}, request_data={}, input_type="request")

        assert result["texts"] == ["mild"]


class TestAliceUnreachable:
    @pytest.mark.parametrize(
        "failure",
        [
            pytest.param({"side_effect": httpx.ConnectError("refused")}, id="connect-error"),
            pytest.param({"return_value": _verdict({"verdict": "MAYBE"})}, id="unrecognized-verdict"),
            pytest.param({"return_value": _verdict({})}, id="no-verdict"),
        ],
    )
    @pytest.mark.asyncio
    async def test_fails_closed_by_default(self, failure: dict):
        guardrail = _guardrail()
        guardrail.async_handler.post = AsyncMock(**failure)

        with pytest.raises(GuardrailRaisedException, match="unavailable"):
            await guardrail.apply_guardrail(inputs={"texts": ["hello"]}, request_data={}, input_type="request")

    @pytest.mark.asyncio
    async def test_fails_open_when_configured(self):
        guardrail = _guardrail(unreachable_fallback="fail_open")
        guardrail.async_handler.post = AsyncMock(side_effect=httpx.ConnectError("refused"))

        result = await guardrail.apply_guardrail(inputs={"texts": ["hello"]}, request_data={}, input_type="request")

        assert result["texts"] == ["hello"]


class TestAliceSerialization:
    """`request_data` carries live objects, so it cannot be posted as it stands."""

    def test_drops_what_cannot_serialize_and_keeps_the_rest(self):
        class Span:
            pass

        result = _json_safe({"model": "x", "metadata": {"span": Span(), "user": "u1"}, "n": 1})

        assert result == {"model": "x", "metadata": {"span": None, "user": "u1"}, "n": 1}

    def test_survives_a_cycle(self):
        data: dict = {"a": 1}
        data["self"] = data

        assert _json_safe(data) == {"a": 1, "self": None}

    def test_dumps_pydantic_models(self):
        from pydantic import BaseModel

        class Model(BaseModel):
            name: str

        assert _json_safe({"m": Model(name="x")}) == {"m": {"name": "x"}}


def test_config_model_is_exposed_for_the_ui():
    config_model = AliceGuardrail.get_config_model()

    assert config_model is not None
    assert config_model.ui_friendly_name() == "Alice by ActiveFence"


def test_guardrail_name_constant():
    assert GUARDRAIL_NAME == "alice"
