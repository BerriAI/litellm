"""
Test for anthropic_endpoints/endpoints.py, focusing on handling dictionary objects in streaming responses
"""

import json
import logging
import unittest
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Mapping
from typing import Final
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi import Request, Response
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient
from starlette.types import Message

from litellm import Router
from litellm._logging import verbose_proxy_logger
from litellm.caching.dual_cache import DualCache
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.utils import ModelResponseStream, TokenCountResponse


class TestAnthropicEndpoints(unittest.TestCase):
    @patch("litellm.litellm_core_utils.safe_json_dumps.safe_dumps")
    @pytest.mark.asyncio
    async def test_async_data_generator_anthropic_dict_handling(self, mock_safe_dumps):
        """Test async_data_generator_anthropic handles dictionary chunks properly"""
        # Setup
        mock_response = AsyncMock()
        mock_response.__aiter__.return_value = [
            {"type": "message_start", "message": {"id": "msg_123"}},
            "text chunk data",
            {"type": "content_block_delta", "delta": {"text": "more data"}},
            "text chunk data again",
        ]

        mock_user_api_key_dict = MagicMock()
        mock_request_data = {}
        mock_proxy_logging_obj = MagicMock()
        mock_proxy_logging_obj.async_post_call_streaming_hook = AsyncMock(
            side_effect=lambda **kwargs: kwargs["response"]
        )

        # Configure safe_dumps to return a properly formatted JSON string
        mock_safe_dumps.side_effect = lambda chunk: json.dumps(chunk)

        # Execute
        result = [
            chunk
            async for chunk in ProxyBaseLLMRequestProcessing.async_sse_data_generator(
                response=mock_response,
                user_api_key_dict=mock_user_api_key_dict,
                request_data=mock_request_data,
                proxy_logging_obj=mock_proxy_logging_obj,
            )
        ]

        # Verify
        expected_result = [
            'data: {"type": "message_start", "message": {"id": "msg_123"}}\n\n',
            "text chunk data",
            'data: {"type": "content_block_delta", "delta": {"text": "more data"}}\n\n',
            "text chunk data again",
        ]

        self.assertEqual(result, expected_result)

        # Assert safe_dumps was called for dictionary objects
        mock_safe_dumps.assert_any_call({"type": "message_start", "message": {"id": "msg_123"}})
        mock_safe_dumps.assert_any_call({"type": "content_block_delta", "delta": {"text": "more data"}})
        assert mock_safe_dumps.call_count == 2  # Called twice, once for each dict object


_BLOCK_CALL_ID: Final = "messages-block"
_BLOCK_TEXT: Final = "blocked by guardrail"


class _BlockEveryRequest(CustomGuardrail):
    async def async_pre_call_hook(
        self, user_api_key_dict: UserAPIKeyAuth, cache: DualCache, data: dict[str, object], call_type: str
    ) -> None:
        self.raise_passthrough_exception(violation_message=_BLOCK_TEXT, request_data=data)


class _RewriteEveryStreamedReply(CustomGuardrail):
    async def async_post_call_streaming_iterator_hook(
        self, user_api_key_dict: UserAPIKeyAuth, response: AsyncIterator[object], request_data: dict[str, object]
    ) -> AsyncGenerator[ModelResponseStream, None]:
        async for _ in response:
            yield ModelResponseStream(choices=[])


def _messages_request(body: Mapping[str, object]) -> Request:
    async def receive() -> Message:
        return {"type": "http.request", "body": json.dumps(body).encode(), "more_body": False}

    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/messages",
            "query_string": b"",
            "headers": [(b"content-type", b"application/json"), (b"x-litellm-call-id", _BLOCK_CALL_ID.encode())],
        },
        receive,
    )


async def _blocked_messages_reply(
    body: Mapping[str, object], fastapi_response: Response, rails: tuple[CustomGuardrail, ...]
) -> tuple[object, str]:
    import litellm
    import litellm.proxy.anthropic_endpoints.endpoints as ep
    import litellm.proxy.proxy_server as proxy_server
    from litellm.proxy.utils import ProxyLogging

    blocker: Final = _BlockEveryRequest(
        guardrail_name="deny-list", event_hook=GuardrailEventHooks.pre_call, default_on=True
    )
    with (
        patch.object(
            proxy_server, "proxy_logging_obj", new=ProxyLogging(user_api_key_cache=DualCache())
        ),  # test-quality-ok: module global imported at call time; no injection seam
        patch.object(litellm, "callbacks", new=[blocker, *rails]),  # test-quality-ok: guardrails register through this global
    ):
        response = await ep.anthropic_response(
            fastapi_response=fastapi_response,
            request=_messages_request(body),
            user_api_key_dict=UserAPIKeyAuth(),
        )
        if not isinstance(response, StreamingResponse):
            return response, ""
        streamed: Final = b"".join(
            [chunk if isinstance(chunk, bytes) else chunk.encode() async for chunk in response.body_iterator]
        ).decode()
        return response, streamed


class TestBlockedResponseUsage:
    """Blocked responses report the blocked LLM response's real usage."""

    def test_uses_original_response_usage(self):
        from litellm.llms.base_llm.guardrail_translation.utils import blocked_response_usage as _blocked_response_usage

        # original_response is the AnthropicMessagesResponse the LLM produced
        # before the guardrail blocked it; its usage is real.
        original = {"usage": {"input_tokens": 31, "output_tokens": 9}}
        assert _blocked_response_usage(original) == {
            "input_tokens": 31,
            "output_tokens": 9,
        }

    def test_zero_usage_when_no_original_response(self):
        from litellm.llms.base_llm.guardrail_translation.utils import blocked_response_usage as _blocked_response_usage

        # Pre-call blocks never invoked the LLM -> nothing consumed.
        assert _blocked_response_usage(None) == {
            "input_tokens": 0,
            "output_tokens": 0,
        }

    @pytest.mark.asyncio
    async def test_blocked_endpoint_response_carries_original_usage(self):
        """The /v1/messages block handler reports the blocked response's real
        usage, carried on ModifyResponseException.original_response."""
        from unittest.mock import AsyncMock, MagicMock

        import litellm.proxy.anthropic_endpoints.endpoints as ep
        import litellm.proxy.proxy_server as proxy_server
        from litellm.integrations.custom_guardrail import ModifyResponseException

        exc = ModifyResponseException(
            message="blocked by guardrail",
            model="claude-3-5-sonnet-20240620",
            request_data={"messages": [{"role": "user", "content": "hi"}]},
            guardrail_name="rubrik",
            original_response={"usage": {"input_tokens": 12, "output_tokens": 5}},
        )

        with (
            patch.object(ep, "read_request_body", new=AsyncMock(return_value={})),
            patch.object(
                ep.ProxyBaseLLMRequestProcessing,
                "base_process_llm_request",
                new=AsyncMock(side_effect=exc),
            ),
            patch.object(proxy_server, "proxy_logging_obj") as mock_logging,
        ):
            mock_logging.post_call_failure_hook = AsyncMock()
            response = await ep.anthropic_response(
                fastapi_response=MagicMock(),
                request=MagicMock(),
                user_api_key_dict=MagicMock(),
            )

        assert response["content"][0]["text"] == "blocked by guardrail"
        assert response["usage"] == {"input_tokens": 12, "output_tokens": 5}
        mock_logging.post_call_failure_hook.assert_awaited_once()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "rails",
        ((), (_RewriteEveryStreamedReply(guardrail_name="post-call-rewrite", default_on=True),)),
        ids=("no_rails", "post_call_rail"),
    )
    async def test_streaming_block_answers_anthropic_events_with_the_call_id(
        self, rails: tuple[CustomGuardrail, ...]
    ) -> None:
        response, streamed = await _blocked_messages_reply(
            {"model": "claude-haiku-5-5", "stream": True}, Response(), rails
        )

        frames: Final = tuple(frame for frame in streamed.split("\n\n") if frame.strip())
        events: Final = tuple(
            (frame.split("\n")[0].removeprefix("event: "), json.loads(frame.split("\n")[1].removeprefix("data: ")))
            for frame in frames
        )
        assert isinstance(response, StreamingResponse)
        assert response.media_type == "text/event-stream"
        assert response.headers["x-litellm-call-id"] == _BLOCK_CALL_ID
        assert [name for name, _ in events] == [
            "message_start",
            "content_block_start",
            "content_block_delta",
            "content_block_stop",
            "message_delta",
            "message_stop",
        ]
        assert all(payload["type"] == name for name, payload in events)
        assert events[2][1]["delta"] == {"type": "text_delta", "text": _BLOCK_TEXT}
        assert events[4][1]["delta"]["stop_reason"] == "end_turn"

    @pytest.mark.asyncio
    async def test_json_block_answers_with_the_call_id(self) -> None:
        fastapi_response: Final = Response()

        response, _ = await _blocked_messages_reply({"model": "claude-haiku-5-5"}, fastapi_response, ())

        assert fastapi_response.headers["x-litellm-call-id"] == _BLOCK_CALL_ID
        assert isinstance(response, Mapping)
        assert response["content"] == [{"type": "text", "text": _BLOCK_TEXT}]


class TestProxyExceptionAnthropicEnvelope:
    @pytest.mark.asyncio
    async def test_anthropic_response_maps_proxy_exception_to_anthropic_envelope(self):
        """LIT-6468: a 400 ProxyException from request validation must surface as
        Anthropic's documented {"type": "error", "error": {...}} envelope with the
        original status and message, not the OpenAI {"error": {...}} envelope."""
        import litellm.proxy.anthropic_endpoints.endpoints as ep
        import litellm.proxy.proxy_server as proxy_server
        from litellm.proxy._types import ProxyErrorTypes, ProxyException

        exc = ProxyException(
            message="Invalid type for 'metadata': expected an object, but got a string instead.",
            type=ProxyErrorTypes.bad_request_error,
            param="metadata",
            code=400,
        )
        request = MagicMock()
        request.headers = {"x-request-id": "req_test_6468"}

        with (
            patch.object(ep, "read_request_body", new=AsyncMock(return_value={})),
            patch.object(
                ep.ProxyBaseLLMRequestProcessing,
                "base_process_llm_request",
                new=AsyncMock(side_effect=exc),
            ),
            patch.object(proxy_server, "proxy_logging_obj") as mock_logging,
        ):
            mock_logging.post_call_failure_hook = AsyncMock()
            response = await ep.anthropic_response(
                fastapi_response=MagicMock(),
                request=request,
                user_api_key_dict=MagicMock(),
            )

        assert response.status_code == 400
        body = json.loads(response.body)
        assert body == {
            "type": "error",
            "error": {
                "type": "invalid_request_error",
                "message": "Invalid type for 'metadata': expected an object, but got a string instead.",
            },
            "request_id": "req_test_6468",
        }
        mock_logging.post_call_failure_hook.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_anthropic_response_maps_429_to_rate_limit_error(self):
        """The Anthropic error type follows the status code (429 -> rate_limit_error),
        and a code-less exception falls back to 500 api_error."""
        import litellm.proxy.anthropic_endpoints.endpoints as ep
        from litellm.proxy._types import ProxyException

        request = MagicMock()
        request.headers = {}

        response = ep._anthropic_error_json_response(
            ProxyException(message="Rate limit exceeded", type="rate_limit_error", param=None, code=429),
            request,
        )
        assert response.status_code == 429
        assert json.loads(response.body)["error"]["type"] == "rate_limit_error"

        fallback = ep._anthropic_error_json_response(
            ProxyException(message="boom", type="None", param=None, code=None),
            request,
        )
        assert fallback.status_code == 500
        assert json.loads(fallback.body)["error"]["type"] == "api_error"

    @staticmethod
    def _call_id_error_response(general_settings, provider_specific_fields=None):
        import litellm.proxy.anthropic_endpoints.endpoints as ep
        from litellm.proxy._types import ProxyException

        request = MagicMock()
        request.headers = {}
        exc = ProxyException(
            message="Rate limit exceeded",
            type="rate_limit_error",
            param=None,
            code=429,
            headers={"x-litellm-call-id": "call-8302"},
            provider_specific_fields=provider_specific_fields,
        )
        with patch("litellm.proxy.proxy_server.general_settings", general_settings):
            return ep._anthropic_error_json_response(exc, request)

    def test_anthropic_error_copies_the_call_id_into_the_error_when_opted_in(self):
        """With include_call_id_in_error_body on, error.litellm_call_id is byte-identical to
        the x-litellm-call-id header and lives inside the error object, which is what the
        Anthropic SDK keeps as e.body."""
        response = self._call_id_error_response({"include_call_id_in_error_body": True})

        assert response.headers["x-litellm-call-id"] == "call-8302"
        assert json.loads(response.body) == {
            "type": "error",
            "error": {
                "type": "rate_limit_error",
                "message": "Rate limit exceeded",
                "litellm_call_id": "call-8302",
            },
        }

    def test_anthropic_error_keeps_provider_specific_fields_next_to_the_call_id(self):
        response = self._call_id_error_response(
            {"include_call_id_in_error_body": True},
            provider_specific_fields={"guardrail": "keyword-block"},
        )

        assert json.loads(response.body)["error"] == {
            "type": "rate_limit_error",
            "message": "Rate limit exceeded",
            "provider_specific_fields": {"guardrail": "keyword-block"},
            "litellm_call_id": "call-8302",
        }

    def test_anthropic_error_leaves_the_envelope_alone_when_opted_out(self):
        response = self._call_id_error_response({})

        assert response.headers["x-litellm-call-id"] == "call-8302"
        assert json.loads(response.body) == {
            "type": "error",
            "error": {"type": "rate_limit_error", "message": "Rate limit exceeded"},
        }


class TestHttpExceptionDictDetail:
    @pytest.mark.asyncio
    async def test_anthropic_response_serializes_dict_detail_http_exception(self):
        """LIT-6466 + LIT-6468: a post_call guardrail's HTTPException(detail=<dict>)
        must surface as Anthropic's {"type": "error", "error": {...}} envelope with
        the guardrail's clean message plus provider_specific_fields, not the str()
        of the exception and not the OpenAI envelope."""
        from fastapi import HTTPException

        import litellm.proxy.anthropic_endpoints.endpoints as ep
        import litellm.proxy.proxy_server as proxy_server
        from litellm.proxy._types import UserAPIKeyAuth

        detail = {
            "error": "Content blocked: keyword 'kumquat' detected",
            "keyword": "kumquat",
            "guardrail": "keyword-block",
        }
        exc = HTTPException(status_code=400, detail=detail)
        request = MagicMock()
        request.headers = {}

        with (
            patch.object(
                ep, "read_request_body", new=AsyncMock(return_value={})
            ),  # test-quality-ok: endpoint reads the body via a module function; no injection seam
            patch.object(  # test-quality-ok: the guardrail raise happens deep inside this call; the test targets the endpoint's except block
                ep.ProxyBaseLLMRequestProcessing,
                "base_process_llm_request",
                new=AsyncMock(side_effect=exc),
            ),
            patch.object(
                proxy_server, "proxy_logging_obj"
            ) as mock_logging,  # test-quality-ok: module global imported at call time; no injection seam
        ):
            mock_logging.post_call_failure_hook = AsyncMock()
            response = await ep.anthropic_response(
                fastapi_response=MagicMock(),
                request=request,
                user_api_key_dict=UserAPIKeyAuth(),
            )

        assert response.status_code == 400
        body = json.loads(response.body)
        assert body["type"] == "error"
        assert body["error"]["type"] == "invalid_request_error"
        assert body["error"]["message"] == "Content blocked: keyword 'kumquat' detected"
        assert "{'error'" not in body["error"]["message"]
        assert body["error"]["provider_specific_fields"] == detail
        mock_logging.post_call_failure_hook.assert_awaited_once()


class TestFailureHookRequestData:
    @pytest.mark.asyncio
    async def test_failure_hook_gets_post_setup_data_with_logging_obj(self):
        """Request setup replaces the processor's data dict (adding the logging
        object the failure hook needs to lift token usage from); the exception
        handler must pass that replaced dict, not the raw request body dict."""
        import litellm.proxy.anthropic_endpoints.endpoints as ep
        import litellm.proxy.proxy_server as proxy_server
        from litellm.proxy._types import UserAPIKeyAuth

        captured = {}

        async def fake_process(self, **kwargs):
            self.data = {**self.data, "litellm_logging_obj": "logging-obj-sentinel"}
            captured["processor_data"] = self.data
            raise RuntimeError("provider timeout")

        request = MagicMock()
        request.headers = {}

        with (
            patch.object(ep, "read_request_body", new=AsyncMock(return_value={"model": "claude-sonnet"})),
            patch.object(ep.ProxyBaseLLMRequestProcessing, "base_process_llm_request", new=fake_process),
            patch.object(proxy_server, "proxy_logging_obj") as mock_logging,
        ):
            mock_logging.post_call_failure_hook = AsyncMock()
            response = await ep.anthropic_response(
                fastapi_response=MagicMock(),
                request=request,
                user_api_key_dict=UserAPIKeyAuth(),
            )

        assert response.status_code == 500
        assert json.loads(response.body)["error"]["message"] == "provider timeout"

        hook_request_data = mock_logging.post_call_failure_hook.await_args.kwargs["request_data"]
        assert hook_request_data is captured["processor_data"]
        assert hook_request_data["litellm_logging_obj"] == "logging-obj-sentinel"


class TestErrorLogCarriesCallId:
    """LIT-7836: the /v1/messages and /v1/messages/count_tokens error lines must carry
    the request's litellm_call_id, rendered in the message and as a structured field."""

    @pytest.fixture(autouse=True)
    def propagating_proxy_logger(self):
        verbose_proxy_logger.propagate = True
        try:
            yield
        finally:
            verbose_proxy_logger.propagate = False

    @staticmethod
    def _error_record(caplog: pytest.LogCaptureFixture) -> logging.LogRecord:
        return next(r for r in caplog.records if "Exception occured" in r.getMessage())

    @pytest.mark.asyncio
    async def test_messages_failure_log_carries_call_id(self, caplog: pytest.LogCaptureFixture):
        import litellm.proxy.anthropic_endpoints.endpoints as ep
        import litellm.proxy.proxy_server as proxy_server
        from litellm.proxy._types import UserAPIKeyAuth

        call_id = "messages-call-7836"

        async def fake_process(self, **kwargs):
            self.data = {**self.data, "litellm_call_id": call_id}
            raise RuntimeError("provider timeout")

        request = MagicMock()
        request.headers = {}

        with (
            patch.object(
                ep, "read_request_body", new=AsyncMock(return_value={"model": "claude-sonnet"})
            ),  # test-quality-ok: endpoint reads the body via a module function; no injection seam
            patch.object(
                ep.ProxyBaseLLMRequestProcessing, "base_process_llm_request", new=fake_process
            ),  # test-quality-ok: the provider failure happens inside this call; the test targets the endpoint's except block
            patch.object(
                proxy_server, "proxy_logging_obj"
            ) as mock_logging,  # test-quality-ok: module global imported at call time; no injection seam
            caplog.at_level(logging.ERROR, logger="LiteLLM Proxy"),
        ):
            mock_logging.post_call_failure_hook = AsyncMock()
            response = await ep.anthropic_response(
                fastapi_response=MagicMock(),
                request=request,
                user_api_key_dict=UserAPIKeyAuth(),
            )

        assert response.status_code == 500
        record = self._error_record(caplog)
        assert record.litellm_call_id == call_id
        assert call_id in record.getMessage()

    @pytest.mark.asyncio
    async def test_messages_already_shaped_failure_answers_with_the_call_id(self):
        import litellm.proxy.anthropic_endpoints.endpoints as ep
        import litellm.proxy.proxy_server as proxy_server
        from litellm.proxy._types import ProxyErrorTypes, ProxyException, UserAPIKeyAuth

        call_id = "messages-call-7836-shaped"

        async def fake_process(self, **kwargs):
            self.data = {**self.data, "litellm_call_id": call_id}
            raise ProxyException(message="budget exceeded", type=ProxyErrorTypes.budget_exceeded, param="key", code=402)

        request = MagicMock()
        request.headers = {}

        with (
            patch.object(
                ep, "read_request_body", new=AsyncMock(return_value={"model": "claude-sonnet"})
            ),  # test-quality-ok: endpoint reads the body via a module function; no injection seam
            patch.object(
                ep.ProxyBaseLLMRequestProcessing, "base_process_llm_request", new=fake_process
            ),  # test-quality-ok: the proxy shaped failure happens inside this call; the test targets the endpoint's except block
            patch.object(
                proxy_server, "proxy_logging_obj"
            ) as mock_logging,  # test-quality-ok: module global imported at call time; no injection seam
        ):
            mock_logging.post_call_failure_hook = AsyncMock()
            response = await ep.anthropic_response(
                fastapi_response=MagicMock(),
                request=request,
                user_api_key_dict=UserAPIKeyAuth(),
            )

        assert response.status_code == 402
        assert response.headers["x-litellm-call-id"] == call_id

    @pytest.mark.asyncio
    async def test_count_tokens_failure_log_carries_callers_call_id(self, caplog: pytest.LogCaptureFixture):
        import litellm.proxy.anthropic_endpoints.endpoints as ep
        import litellm.proxy.proxy_server as proxy_server
        from litellm.proxy._types import UserAPIKeyAuth

        call_id = "count-tokens-call-7836"
        request = MagicMock()
        request.headers = {"x-litellm-call-id": call_id}

        with (
            patch.object(  # test-quality-ok: endpoint reads the body via a module function; no injection seam
                ep,
                "read_request_body",
                new=AsyncMock(return_value={"model": "claude-sonnet", "messages": [{"role": "user", "content": "hi"}]}),
            ),
            patch.object(
                proxy_server, "count_request_tokens", new=AsyncMock(side_effect=RuntimeError("tokenizer down"))
            ),  # test-quality-ok: module global imported at call time; the test targets the endpoint's except block
            caplog.at_level(logging.ERROR, logger="LiteLLM Proxy"),
        ):
            response = await ep.count_tokens(request=request, user_api_key_dict=UserAPIKeyAuth())

        assert response.status_code == 500
        record = self._error_record(caplog)
        assert record.litellm_call_id == call_id
        assert call_id in record.getMessage()


def _claude_router() -> Router:
    return Router(
        model_list=[
            {
                "model_name": "claude-sonnet",
                "litellm_params": {"model": "anthropic/claude-sonnet-5-5", "api_key": "sk-test"},
            }
        ]
    )


_MALFORMED_TOOL_ENTRY: Final = [{"name": "get_weather", "input_schema": "object"}]
_SEVEN_TOKENS: Final = TokenCountResponse(
    total_tokens=7, request_model="claude-sonnet", model_used="claude-sonnet", tokenizer_type="test"
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model", "tools"),
    [
        ("claude-sonnet", 5),
        ("claude-sonnet", ""),
        ("claude-sonnet", ["get_weather"]),
        ("claude-sonnet", _MALFORMED_TOOL_ENTRY),
        ("team-public-name", 5),
    ],
    ids=["int", "empty_string", "list_of_strings", "input_schema_string", "unnamed_model_int"],
)
async def test_count_tokens_answers_400_for_malformed_tools_without_counting(model: str, tools: object) -> None:
    import litellm.proxy.anthropic_endpoints.endpoints as ep
    import litellm.proxy.proxy_server as proxy_server
    from litellm.proxy._types import UserAPIKeyAuth

    request: Final = MagicMock()
    request.headers = {}
    counter: Final = AsyncMock()
    body: Final = {"model": model, "messages": [{"role": "user", "content": "hi"}], "tools": tools}

    with (
        patch.object(ep, "read_request_body", new=AsyncMock(return_value=body)),  # test-quality-ok: endpoint reads the body via a module function; no injection seam
        patch.object(proxy_server, "count_request_tokens", new=counter),  # test-quality-ok: module global imported at call time; the test asserts it is never reached
        patch.object(proxy_server, "llm_router", new=_claude_router()),  # test-quality-ok: module global imported at call time; no injection seam on this route
    ):
        response: Final = await ep.count_tokens(request=request, user_api_key_dict=UserAPIKeyAuth())

    assert response.status_code == 400
    assert json.loads(response.body)["error"]["type"] == "invalid_request_error"
    counter.assert_not_awaited()


@pytest.mark.asyncio
async def test_count_tokens_counts_malformed_tool_entries_for_a_model_the_router_cannot_name() -> None:
    import litellm.proxy.anthropic_endpoints.endpoints as ep
    import litellm.proxy.proxy_server as proxy_server
    from litellm.proxy._types import UserAPIKeyAuth

    request: Final = MagicMock()
    request.headers = {}
    counter: Final = AsyncMock(return_value=_SEVEN_TOKENS)
    body: Final = {"model": "team-public-name", "messages": [{"role": "user", "content": "hi"}], "tools": _MALFORMED_TOOL_ENTRY}

    with (
        patch.object(ep, "read_request_body", new=AsyncMock(return_value=body)),  # test-quality-ok: endpoint reads the body via a module function; no injection seam
        patch.object(proxy_server, "count_request_tokens", new=counter),  # test-quality-ok: module global imported at call time; the test asserts what it receives
        patch.object(proxy_server, "llm_router", new=_claude_router()),  # test-quality-ok: module global imported at call time; no injection seam on this route
    ):
        response: Final = await ep.count_tokens(request=request, user_api_key_dict=UserAPIKeyAuth())

    assert response == {"input_tokens": 7}
    assert counter.await_args.kwargs["request"].tools == _MALFORMED_TOOL_ENTRY


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("dropping", "headers"),
    [
        ({"drop_params": True}, {}),
        ({"additional_drop_params": ["tools"]}, {}),
        ({}, {"user-agent": "claude-cli/2.1.0 (external, cli)"}),
    ],
    ids=["drop_params", "additional_drop_params", "claude_code_user_agent"],
)
async def test_count_tokens_drops_malformed_tools_when_drop_params_is_set(
    dropping: dict[str, object], headers: dict[str, str]
) -> None:
    import litellm.proxy.anthropic_endpoints.endpoints as ep
    import litellm.proxy.proxy_server as proxy_server
    from litellm.proxy._types import UserAPIKeyAuth

    request: Final = MagicMock()
    request.headers = headers
    counter: Final = AsyncMock(return_value=_SEVEN_TOKENS)
    messages: Final = [{"role": "user", "content": "hi"}]
    body: Final = {"model": "claude-sonnet", "messages": messages, "tools": 5, **dropping}

    with (
        patch.object(ep, "read_request_body", new=AsyncMock(return_value=body)),  # test-quality-ok: endpoint reads the body via a module function; no injection seam
        patch.object(proxy_server, "count_request_tokens", new=counter),  # test-quality-ok: module global imported at call time; the test asserts what it receives
    ):
        response: Final = await ep.count_tokens(request=request, user_api_key_dict=UserAPIKeyAuth())

    assert response == {"input_tokens": 7}
    counted: Final = counter.await_args.kwargs["request"]
    assert counted.tools is None
    assert counted.messages == messages


@pytest.mark.asyncio
async def test_count_tokens_drops_malformed_tools_the_counted_deployment_drops_even_with_fallbacks() -> None:
    import litellm.proxy.anthropic_endpoints.endpoints as ep
    import litellm.proxy.proxy_server as proxy_server
    from litellm import Router
    from litellm.proxy._types import UserAPIKeyAuth

    router: Final = Router(
        model_list=[
            {
                "model_name": "claude-sonnet",
                "litellm_params": {
                    "model": "anthropic/claude-sonnet-5-5",
                    "api_key": "sk-test",
                    "additional_drop_params": ["tools"],
                },
            },
            {"model_name": "backup", "litellm_params": {"model": "anthropic/claude-haiku-5-5", "api_key": "sk-test"}},
        ],
        fallbacks=[{"claude-sonnet": ["backup"]}],
    )
    request: Final = MagicMock()
    request.headers = {}
    counter: Final = AsyncMock(return_value=_SEVEN_TOKENS)
    body: Final = {"model": "claude-sonnet", "messages": [{"role": "user", "content": "hi"}], "tools": 5}

    with (
        patch.object(ep, "read_request_body", new=AsyncMock(return_value=body)),  # test-quality-ok: endpoint reads the body via a module function; no injection seam
        patch.object(proxy_server, "count_request_tokens", new=counter),  # test-quality-ok: module global imported at call time; the test asserts what it receives
        patch.object(proxy_server, "llm_router", new=router),  # test-quality-ok: module global imported at call time; no injection seam on this route
    ):
        response: Final = await ep.count_tokens(request=request, user_api_key_dict=UserAPIKeyAuth())

    assert response == {"input_tokens": 7}
    assert counter.await_args.kwargs["request"].tools is None


def _count_on_claude(
    monkeypatch: pytest.MonkeyPatch, peer: Callable[[httpx.Request], httpx.Response], headers: Mapping[str, str]
) -> httpx.Response:
    from fastapi import FastAPI

    import litellm.proxy.proxy_server as proxy_server
    from litellm.llms.anthropic.count_tokens import handler as anthropic_count_tokens_handler
    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.anthropic_endpoints.endpoints import router
    from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

    claude: Final = {"model": "anthropic/claude-zeta-9", "api_key": "test-key", "api_base": "https://count-tokens.test"}
    monkeypatch.setattr(
        proxy_server,
        "llm_router",
        Router(
            model_list=[
                {
                    "model_name": "count-log-claude",
                    "litellm_params": {**claude, "order": 2},
                    "model_info": {"id": "count-log-fallback-deployment"},
                },
                {
                    "model_name": "count-log-claude",
                    "litellm_params": {**claude, "order": 1},
                    "model_info": {"id": "count-log-routed-deployment"},
                },
            ]
        ),
    )
    monkeypatch.setattr(
        anthropic_count_tokens_handler,
        "get_async_httpx_client",
        lambda *args, **kwargs: AsyncHTTPHandler(transport=httpx.MockTransport(peer)),
    )
    app: Final = FastAPI()
    app.include_router(router)
    app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth()
    return TestClient(app).post(
        "/v1/messages/count_tokens",
        json={"model": "count-log-claude", "messages": [{"role": "user", "content": "Hello"}]},
        headers=dict(headers),
    )


def test_count_tokens_logs_one_line_with_the_callers_call_id_model_and_count(
    monkeypatch: pytest.MonkeyPatch, token_count_log_lines: Callable[[], tuple[dict[str, object], ...]]
) -> None:
    response: Final = _count_on_claude(
        monkeypatch,
        lambda request: httpx.Response(200, json={"input_tokens": 77}, request=request),
        {"x-litellm-call-id": "count-log-msg-claude"},
    )

    assert response.json() == {"input_tokens": 77, "litellm_count_source": "provider"}, response.text
    assert token_count_log_lines() == (
        {
            "litellm_call_id": "count-log-msg-claude",
            "request_model": "count-log-claude",
            "model_id": "count-log-routed-deployment",
            "model_used": "claude-zeta-9",
            "input_tokens": 77,
            "count_source": "provider",
            "status_code": 200,
        },
    )


def test_count_tokens_provider_rejection_logs_its_status_and_answers_the_same_error(
    monkeypatch: pytest.MonkeyPatch, token_count_log_lines: Callable[[], tuple[dict[str, object], ...]]
) -> None:
    rejection: Final = {
        "type": "error",
        "error": {"type": "invalid_request_error", "message": "messages.0.content.0: invalid base64 data"},
    }

    response: Final = _count_on_claude(
        monkeypatch,
        lambda request: httpx.Response(400, json=rejection, request=request),
        {"x-litellm-call-id": "count-log-msg-error"},
    )

    assert response.status_code == 400, response.text
    assert response.json()["error"] == rejection["error"]
    assert token_count_log_lines() == (
        {
            "litellm_call_id": "count-log-msg-error",
            "request_model": "count-log-claude",
            "model_id": "count-log-routed-deployment",
            "model_used": None,
            "input_tokens": None,
            "count_source": None,
            "status_code": 400,
        },
    )


def _count_on_local_model(monkeypatch: pytest.MonkeyPatch, body: Mapping[str, object]) -> httpx.Response:
    from fastapi import FastAPI

    import litellm.proxy.proxy_server as proxy_server
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.anthropic_endpoints.endpoints import router
    from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

    monkeypatch.setattr(
        proxy_server,
        "llm_router",
        Router(
            model_list=[
                {
                    "model_name": "count-log-local",
                    "litellm_params": {"model": "hosted_vllm/count-log-local", "api_base": "https://local.test/v1"},
                    "model_info": {"id": "count-log-local-deployment"},
                }
            ]
        ),
    )
    app: Final = FastAPI()
    app.include_router(router)
    app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth()
    return TestClient(app).post(
        "/v1/messages/count_tokens", json=dict(body), headers={"x-litellm-call-id": "count-log-local"}
    )


def test_count_tokens_unsupported_content_logs_the_400_it_answers(
    monkeypatch: pytest.MonkeyPatch, token_count_log_lines: Callable[[], tuple[dict[str, object], ...]]
) -> None:
    response: Final = _count_on_local_model(
        monkeypatch,
        {"model": "count-log-local", "messages": [{"role": "user", "content": [{"type": "future_block"}]}]},
    )

    assert (response.status_code, response.json()["error"]["type"]) == (400, "invalid_request_error"), response.text
    assert token_count_log_lines() == (
        {
            "litellm_call_id": "count-log-local",
            "request_model": "count-log-local",
            "model_id": "count-log-local-deployment",
            "model_used": None,
            "input_tokens": None,
            "count_source": None,
            "status_code": 400,
        },
    )


@pytest.mark.parametrize(
    ("body", "request_model"),
    [
        ({"messages": [{"role": "user", "content": "Hello"}]}, None),
        ({"model": "count-log-local", "messages": []}, "count-log-local"),
    ],
)
def test_count_tokens_rejected_before_counting_logs_the_400_it_answers(
    monkeypatch: pytest.MonkeyPatch,
    token_count_log_lines: Callable[[], tuple[dict[str, object], ...]],
    body: Mapping[str, object],
    request_model: str | None,
) -> None:
    response: Final = _count_on_local_model(monkeypatch, body)

    assert response.status_code == 400, response.text
    assert token_count_log_lines() == (
        {
            "litellm_call_id": "count-log-local",
            "request_model": request_model,
            "model_id": None,
            "model_used": None,
            "input_tokens": None,
            "count_source": None,
            "status_code": 400,
        },
    )


class TestEventLoggingBatchEndpoint:
    """Test the stubbed event logging batch endpoint"""

    def test_event_logging_batch_endpoint_exists(self):
        """Test that the event_logging_batch endpoint exists and returns 200"""
        from fastapi import FastAPI

        from litellm.proxy.anthropic_endpoints.endpoints import router

        app = FastAPI()
        app.include_router(router)

        client = TestClient(app)
        response = client.post("/api/event_logging/batch", json={"events": []})

        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestStripTotalTokens(unittest.TestCase):
    """Cover ``_strip_total_tokens_from_anthropic_response``.

    The Anthropic /v1/messages spec does not define ``usage.total_tokens``.
    LiteLLM injects it internally; the helper must remove it from the wire
    response so the non-streaming path matches the streaming SSE shape and
    direct Anthropic API responses.
    """

    def test_strips_total_tokens_when_present(self):
        from litellm.proxy.anthropic_endpoints.endpoints import (
            _strip_total_tokens_from_anthropic_response,
        )

        response = {
            "id": "msg_123",
            "usage": {
                "input_tokens": 100,
                "output_tokens": 50,
                "total_tokens": 150,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 0,
            },
        }
        _strip_total_tokens_from_anthropic_response(response)
        assert "total_tokens" not in response["usage"]
        assert response["usage"]["input_tokens"] == 100
        assert response["usage"]["output_tokens"] == 50
        assert response["usage"]["cache_read_input_tokens"] == 0

    def test_no_op_when_total_tokens_absent(self):
        from litellm.proxy.anthropic_endpoints.endpoints import (
            _strip_total_tokens_from_anthropic_response,
        )

        response = {"usage": {"input_tokens": 100, "output_tokens": 50}}
        _strip_total_tokens_from_anthropic_response(response)
        assert response["usage"] == {"input_tokens": 100, "output_tokens": 50}

    def test_no_op_when_usage_missing(self):
        from litellm.proxy.anthropic_endpoints.endpoints import (
            _strip_total_tokens_from_anthropic_response,
        )

        response = {"id": "msg_123"}
        _strip_total_tokens_from_anthropic_response(response)
        assert response == {"id": "msg_123"}

    def test_no_op_on_non_dict_response(self):
        from litellm.proxy.anthropic_endpoints.endpoints import (
            _strip_total_tokens_from_anthropic_response,
        )

        # Streaming responses (StreamingResponse, async iterators) are not dicts.
        # The helper must not raise or attempt to mutate them.
        for value in (None, "stream", 42, [{"usage": {"total_tokens": 1}}]):
            _strip_total_tokens_from_anthropic_response(value)  # no raise

    def test_strips_total_tokens_on_pydantic_model_with_dict_usage(self):
        """Greptile P1 on #30382: helper must not silently no-op when the
        response is a Pydantic-shaped object whose `usage` attribute is a
        plain dict (the common case for objects wrapping raw upstream JSON).
        """
        from types import SimpleNamespace

        from litellm.proxy.anthropic_endpoints.endpoints import (
            _strip_total_tokens_from_anthropic_response,
        )

        # SimpleNamespace mimics the .usage attribute access pattern; the
        # helper's contract: if .usage is dict-shaped, strip total_tokens.
        response = SimpleNamespace(usage={"input_tokens": 100, "output_tokens": 50, "total_tokens": 150})
        _strip_total_tokens_from_anthropic_response(response)
        assert "total_tokens" not in response.usage
        assert response.usage == {"input_tokens": 100, "output_tokens": 50}


class TestStripTotalTokensFeatureFlag(unittest.TestCase):
    """The strip is gated behind `litellm.strip_anthropic_total_tokens`.

    Default off (backward compat). Greptile P1 on #30382 required a
    user-controlled flag so existing clients reading the LiteLLM-shaped
    `usage.total_tokens` continue to work after this PR lands.
    """

    def test_flag_defaults_off(self):
        import litellm

        assert litellm.strip_anthropic_total_tokens is False
