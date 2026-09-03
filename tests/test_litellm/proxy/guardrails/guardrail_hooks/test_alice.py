import json
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


def _guardrail(**overrides: object) -> AliceGuardrail:
    params: dict[str, object] = {"api_key": "test-key", "guardrail_name": "alice", "event_hook": "pre_call"}
    params.update(overrides)
    return AliceGuardrail(**params)


def _verdict(payload: dict[str, object], status_code: int = 200) -> Response:
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
    """The hook's arguments cross the wire as they were received — nothing selected, nothing
    renamed — except the caller's raw credentials, which are stripped before request_data is
    serialized (see TestAliceCredentialStripping)."""

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
    async def test_nothing_selectable_reaches_no_evaluation(self):
        """No texts, images, tools, tool_calls, or structured_messages: genuinely nothing to send."""
        guardrail = _guardrail()
        guardrail.async_handler.post = AsyncMock()

        result = await guardrail.apply_guardrail(inputs={"texts": []}, request_data={}, input_type="request")

        assert result == {"texts": []}
        guardrail.async_handler.post.assert_not_called()

    @pytest.mark.asyncio
    async def test_tool_calls_only_still_reaches_alice(self):
        """A batch with empty texts but populated tool_calls is still a selection decision Alice
        should make, not the plugin — see the class docstring."""
        guardrail = _guardrail()
        guardrail.async_handler.post = AsyncMock(return_value=_verdict({"verdict": "ALLOW", "categories": []}))
        inputs = {"texts": [], "tool_calls": [{"id": "call_1", "function": {"name": "get_weather"}}]}

        await guardrail.apply_guardrail(inputs=inputs, request_data={}, input_type="request")

        guardrail.async_handler.post.assert_called_once()
        assert guardrail.async_handler.post.call_args.kwargs["json"]["inputs"]["tool_calls"] == inputs["tool_calls"]

    @pytest.mark.asyncio
    async def test_images_only_still_reaches_alice(self):
        guardrail = _guardrail()
        guardrail.async_handler.post = AsyncMock(return_value=_verdict({"verdict": "ALLOW", "categories": []}))

        await guardrail.apply_guardrail(
            inputs={"texts": [], "images": ["data:image/png;base64,abc"]}, request_data={}, input_type="request"
        )

        guardrail.async_handler.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_structured_messages_only_still_reaches_alice(self):
        guardrail = _guardrail()
        guardrail.async_handler.post = AsyncMock(return_value=_verdict({"verdict": "ALLOW", "categories": []}))

        await guardrail.apply_guardrail(
            inputs={"texts": [], "structured_messages": [{"role": "user", "content": []}]},
            request_data={},
            input_type="request",
        )

        guardrail.async_handler.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_makes_exactly_one_attempt(self):
        guardrail = _guardrail(unreachable_fallback="fail_open")
        guardrail.async_handler.post = AsyncMock(side_effect=httpx.ConnectError("refused"))

        await guardrail.apply_guardrail(inputs={"texts": ["hi"]}, request_data={}, input_type="request")

        assert guardrail.async_handler.post.call_count == 1


class TestAliceCredentialStripping:
    """request_data's raw-credential keys never leave the process."""

    @pytest.mark.asyncio
    async def test_secret_fields_and_api_key_are_stripped(self):
        guardrail = _guardrail()
        guardrail.async_handler.post = AsyncMock(return_value=_verdict({"verdict": "ALLOW", "categories": []}))
        request_data = {
            "model": "gpt-4o",
            "api_key": "sk-forwarded-provider-secret",
            "secret_fields": {"raw_headers": {"authorization": "Bearer caller-virtual-key"}},
            "metadata": {"user_api_key_alias": "payments-bot"},
        }

        await guardrail.apply_guardrail(inputs={"texts": ["hi"]}, request_data=request_data, input_type="request")

        sent_request_data = guardrail.async_handler.post.call_args.kwargs["json"]["request_data"]
        assert "secret_fields" not in sent_request_data
        assert "api_key" not in sent_request_data
        assert sent_request_data == {"model": "gpt-4o", "metadata": {"user_api_key_alias": "payments-bot"}}

    @pytest.mark.asyncio
    async def test_nested_credentials_are_stripped_at_every_depth(self):
        """Shaped after a real captured Claude Code payload: the caller's Authorization/x-api-key
        lives under several independent nesting paths, none of which are the root."""
        guardrail = _guardrail()
        guardrail.async_handler.post = AsyncMock(return_value=_verdict({"verdict": "ALLOW", "categories": []}))
        request_data = {
            "model": "claude-3-5-sonnet",
            "secret_fields": {"raw_headers": {"authorization": "Bearer caller-virtual-key"}},
            "provider_specific_header": {"extra_headers": {"authorization": "sk-ant-oat01-nested-oauth"}},
            "proxy_server_request": {
                "url": "/v1/messages",
                "headers": {"authorization": "Bearer inbound-caller-secret", "x-request-id": "req-1"},
                "body": {
                    "model": "claude-3-5-sonnet",
                    "metadata": {"headers": {"authorization": "Bearer body-metadata-secret"}},
                },
            },
            "metadata": {
                "user_api_key_alias": "payments-bot",
                "headers": {"authorization": "Bearer metadata-secret"},
                "requester_metadata": {"headers": {"authorization": "Bearer requester-metadata-secret"}},
            },
            "litellm_metadata": {"headers": {"authorization": "Bearer litellm-metadata-secret"}},
        }

        await guardrail.apply_guardrail(inputs={"texts": ["hi"]}, request_data=request_data, input_type="request")

        posted_body = guardrail.async_handler.post.call_args.kwargs["json"]
        serialized = json.dumps(posted_body)
        assert "authorization" not in serialized.lower()
        assert "caller-virtual-key" not in serialized
        assert "nested-oauth" not in serialized
        assert "inbound-caller-secret" not in serialized
        assert "body-metadata-secret" not in serialized
        assert "metadata-secret" not in serialized
        assert "requester-metadata-secret" not in serialized
        assert "litellm-metadata-secret" not in serialized

        sent_request_data = posted_body["request_data"]
        assert sent_request_data["model"] == "claude-3-5-sonnet"
        assert sent_request_data["proxy_server_request"]["url"] == "/v1/messages"
        assert "headers" not in sent_request_data["proxy_server_request"]
        assert sent_request_data["proxy_server_request"]["body"]["model"] == "claude-3-5-sonnet"
        assert "headers" not in sent_request_data["proxy_server_request"]["body"]["metadata"]
        assert sent_request_data["metadata"]["user_api_key_alias"] == "payments-bot"
        assert "headers" not in sent_request_data["metadata"]
        assert "requester_metadata" in sent_request_data["metadata"]
        assert "headers" not in sent_request_data["metadata"]["requester_metadata"]
        assert "headers" not in sent_request_data["litellm_metadata"]
        assert "secret_fields" not in sent_request_data
        assert "provider_specific_header" not in sent_request_data

    @pytest.mark.asyncio
    async def test_the_original_request_data_is_not_mutated(self):
        """Stripping must only affect the outbound copy — api_key still has to reach the
        provider, and secret_fields still has to reach the rest of the request pipeline."""
        guardrail = _guardrail()
        guardrail.async_handler.post = AsyncMock(return_value=_verdict({"verdict": "ALLOW", "categories": []}))
        request_data = {"api_key": "sk-forwarded-provider-secret", "secret_fields": {"raw_headers": {}}}

        await guardrail.apply_guardrail(inputs={"texts": ["hi"]}, request_data=request_data, input_type="request")

        assert request_data["api_key"] == "sk-forwarded-provider-secret"
        assert request_data["secret_fields"] == {"raw_headers": {}}


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
    async def test_mask_with_no_replacements_blocks(self):
        guardrail = _guardrail()
        guardrail.async_handler.post = AsyncMock(return_value=_verdict({"verdict": "MASK", "categories": []}))

        with pytest.raises(GuardrailRaisedException):
            await guardrail.apply_guardrail(inputs={"texts": ["hello"]}, request_data={}, input_type="request")

    @pytest.mark.asyncio
    async def test_mask_with_one_invalid_replacement_blocks_entirely(self):
        """A mixed valid/invalid replacement list must not let the valid half through: that
        would leave the content named by the invalid entry unmasked while looking like success."""
        guardrail = _guardrail()
        guardrail.async_handler.post = AsyncMock(
            return_value=_verdict(
                {
                    "verdict": "MASK",
                    "categories": ["pii"],
                    "replacements": [{"index": 0, "text": "***"}, {"index": 9, "text": "***"}],
                }
            )
        )

        with pytest.raises(GuardrailRaisedException):
            await guardrail.apply_guardrail(
                inputs={"texts": ["my ssn is 123-45-6789"]}, request_data={}, input_type="request"
            )

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


class TestAliceTransportFailures:
    """Every path out of the HTTP call, since each decides whether traffic flows unscreened."""

    @pytest.mark.asyncio
    async def test_a_timeout_is_unreachable(self):
        guardrail = _guardrail()
        guardrail.async_handler.post = AsyncMock(
            side_effect=litellm.exceptions.Timeout(message="slow", model="gpt-4o", llm_provider="openai")
        )

        with pytest.raises(GuardrailRaisedException, match="unavailable"):
            await guardrail.apply_guardrail(inputs={"texts": ["hello"]}, request_data={}, input_type="request")

    @pytest.mark.parametrize("status", [500, 502, 503, 504])
    @pytest.mark.asyncio
    async def test_upstream_5xx_is_unreachable(self, status: int):
        guardrail = _guardrail(unreachable_fallback="fail_open")
        guardrail.async_handler.post = AsyncMock(
            side_effect=httpx.HTTPStatusError(
                "server error",
                request=Request("POST", "https://api.alice.io/v2/evaluate/litellm"),
                response=_verdict({}, status_code=status),
            )
        )

        result = await guardrail.apply_guardrail(inputs={"texts": ["hello"]}, request_data={}, input_type="request")

        assert result["texts"] == ["hello"]

    @pytest.mark.asyncio
    async def test_a_500_fails_closed_by_default(self):
        guardrail = _guardrail()
        guardrail.async_handler.post = AsyncMock(
            side_effect=httpx.HTTPStatusError(
                "server error",
                request=Request("POST", "https://api.alice.io/v2/evaluate/litellm"),
                response=_verdict({}, status_code=500),
            )
        )

        with pytest.raises(GuardrailRaisedException, match="unavailable"):
            await guardrail.apply_guardrail(inputs={"texts": ["hello"]}, request_data={}, input_type="request")

    @pytest.mark.asyncio
    async def test_a_4xx_is_not_treated_as_unreachable(self):
        """A rejected credential is our misconfiguration, not an outage — it must not fail open."""
        guardrail = _guardrail(unreachable_fallback="fail_open")
        guardrail.async_handler.post = AsyncMock(
            side_effect=httpx.HTTPStatusError(
                "unauthorized",
                request=Request("POST", "https://api.alice.io/v2/evaluate/litellm"),
                response=_verdict({}, status_code=401),
            )
        )

        with pytest.raises(httpx.HTTPStatusError):
            await guardrail.apply_guardrail(inputs={"texts": ["hello"]}, request_data={}, input_type="request")

    @pytest.mark.asyncio
    async def test_a_non_object_body_fails_closed_by_default(self):
        guardrail = _guardrail()
        guardrail.async_handler.post = AsyncMock(
            return_value=Response(
                status_code=200,
                json=["not", "an", "object"],
                request=Request("POST", "https://api.alice.io/v2/evaluate/litellm"),
            )
        )

        with pytest.raises(GuardrailRaisedException, match="unavailable"):
            await guardrail.apply_guardrail(inputs={"texts": ["hello"]}, request_data={}, input_type="request")

    @pytest.mark.asyncio
    async def test_a_non_object_body_fails_open_when_configured(self):
        guardrail = _guardrail(unreachable_fallback="fail_open")
        guardrail.async_handler.post = AsyncMock(
            return_value=Response(
                status_code=200,
                json=["not", "an", "object"],
                request=Request("POST", "https://api.alice.io/v2/evaluate/litellm"),
            )
        )

        result = await guardrail.apply_guardrail(inputs={"texts": ["hello"]}, request_data={}, input_type="request")

        assert result["texts"] == ["hello"]

    @pytest.mark.asyncio
    async def test_malformed_json_fails_closed_by_default(self):
        guardrail = _guardrail()
        guardrail.async_handler.post = AsyncMock(
            return_value=Response(
                status_code=200,
                content=b"not json",
                request=Request("POST", "https://api.alice.io/v2/evaluate/litellm"),
            )
        )

        with pytest.raises(GuardrailRaisedException, match="unavailable"):
            await guardrail.apply_guardrail(inputs={"texts": ["hello"]}, request_data={}, input_type="request")

    @pytest.mark.asyncio
    async def test_malformed_json_fails_open_when_configured(self):
        guardrail = _guardrail(unreachable_fallback="fail_open")
        guardrail.async_handler.post = AsyncMock(
            return_value=Response(
                status_code=200,
                content=b"not json",
                request=Request("POST", "https://api.alice.io/v2/evaluate/litellm"),
            )
        )

        result = await guardrail.apply_guardrail(inputs={"texts": ["hello"]}, request_data={}, input_type="request")

        assert result["texts"] == ["hello"]

    @pytest.mark.asyncio
    async def test_an_undecodable_body_fails_closed_by_default(self):
        """UnicodeDecodeError is a sibling of JSONDecodeError under ValueError, not a subclass."""
        guardrail = _guardrail()
        guardrail.async_handler.post = AsyncMock(
            return_value=Response(
                status_code=200,
                content=b"\xff\xfe not utf-8",
                request=Request("POST", "https://api.alice.io/v2/evaluate/litellm"),
            )
        )

        with pytest.raises(GuardrailRaisedException, match="unavailable"):
            await guardrail.apply_guardrail(inputs={"texts": ["hello"]}, request_data={}, input_type="request")

    @pytest.mark.asyncio
    async def test_an_undecodable_body_fails_open_when_configured(self):
        guardrail = _guardrail(unreachable_fallback="fail_open")
        guardrail.async_handler.post = AsyncMock(
            return_value=Response(
                status_code=200,
                content=b"\xff\xfe not utf-8",
                request=Request("POST", "https://api.alice.io/v2/evaluate/litellm"),
            )
        )

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

    def test_drops_a_model_that_will_not_dump(self):
        class Stubborn:
            def model_dump(self, mode: str = "python") -> dict:
                raise RuntimeError("cannot serialise")

        assert _json_safe({"m": Stubborn()}) == {"m": None}

    def test_drops_a_bare_unserialisable_value(self):
        class Span:
            pass

        assert _json_safe(Span()) is None

    def test_dumps_pydantic_models(self):
        from pydantic import BaseModel

        class Model(BaseModel):
            name: str

        assert _json_safe({"m": Model(name="x")}) == {"m": {"name": "x"}}


def test_config_model_is_exposed_for_the_ui():
    config_model = AliceGuardrail.get_config_model()

    assert config_model is not None
    assert config_model.ui_friendly_name() == "Alice"


def test_guardrail_name_constant():
    assert GUARDRAIL_NAME == "alice"
