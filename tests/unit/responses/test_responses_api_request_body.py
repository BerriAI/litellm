"""
Test that litellm.responses() / litellm.aresponses() send the expected request body
over the wire and surface provider errors correctly. Expected JSON bodies are stored
in expected_responses_api_request/.
"""

import asyncio
import copy
import json
import logging
import math
from collections.abc import Awaitable, Callable, Mapping
from pathlib import Path
from importlib import import_module
from typing import Final
from unittest.mock import AsyncMock, patch

import httpx
import pytest
import respx
from pydantic import TypeAdapter

import litellm
from litellm._logging import verbose_logger
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.types.llms.openai import ResponsesAPIResponse
from tests.unit.llms.bedrock.slow_upstream import (
    STREAM_TIMEOUT_SECONDS,
    UPSTREAM_FIRST_BYTE_SECONDS,
    slow_upstream_sync_client,
)


def _expected_dir() -> Path:
    """Path to expected_responses_api_request folder (sibling of tests/unit/responses)."""
    return Path(__file__).resolve().parent.parent / "expected_responses_api_request"


def _load_expected_body(filename: str) -> dict:
    expected_path = _expected_dir() / filename
    assert expected_path.exists(), f"Expected file not found: {expected_path}"
    with open(expected_path) as f:
        return json.load(f)


def _minimal_responses_api_payload(response_id: str, model: str) -> dict:
    return {
        "id": response_id,
        "object": "response",
        "created_at": 1734366691,
        "status": "completed",
        "model": model,
        "output": [
            {
                "type": "message",
                "id": "msg_1",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Done.", "annotations": []}],
            }
        ],
        "parallel_tool_calls": True,
        "usage": {
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15,
            "output_tokens_details": {"reasoning_tokens": 0},
        },
        "error": None,
        "incomplete_details": None,
        "instructions": None,
        "metadata": None,
        "temperature": None,
        "tool_choice": "auto",
        "tools": [],
        "top_p": None,
        "max_output_tokens": None,
        "previous_response_id": None,
        "reasoning": None,
        "truncation": None,
        "user": None,
    }


class MockResponse:
    def __init__(self, json_data, status_code=200):
        self._json_data = json_data
        self.status_code = status_code
        self.text = json.dumps(json_data)
        self.headers = httpx.Headers({})

    def json(self):
        return self._json_data


def _assert_request_body_matches(request_body: dict, expected_body: dict) -> None:
    for key, expected_value in expected_body.items():
        assert key in request_body, f"Missing key in request body: {key}"
        assert request_body[key] == expected_value, (
            f"Mismatch for key {key}: got {request_body[key]!r}, expected {expected_value!r}"
        )


@pytest.mark.asyncio
async def test_aresponses_context_management_and_shell_request_body_matches_expected():
    """
    Call litellm.aresponses() with context_management and shell tool;
    assert the httpx POST request body matches the expected JSON.
    """
    expected_body = _load_expected_body("context_management_and_shell.json")

    with patch(
        "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
        new_callable=AsyncMock,
    ) as mock_post:
        mock_post.return_value = MockResponse(_minimal_responses_api_payload("resp_ctx_shell_test", "gpt-4o"), 200)

        await litellm.aresponses(
            model="openai/gpt-4o",
            input=expected_body["input"],
            context_management=expected_body["context_management"],
            tools=expected_body["tools"],
            tool_choice=expected_body["tool_choice"],
            max_output_tokens=expected_body["max_output_tokens"],
        )

        mock_post.assert_called_once()
        _assert_request_body_matches(mock_post.call_args.kwargs["json"], expected_body)


@pytest.mark.asyncio
async def test_aresponses_azure_shell_tool_request_body_matches_expected():
    """
    Call litellm.aresponses() on the Azure route with the shell tool;
    assert the httpx POST request body carries the shell tool verbatim.
    """
    expected_body = _load_expected_body("azure_shell_tool.json")

    with patch(
        "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
        new_callable=AsyncMock,
    ) as mock_post:
        mock_post.return_value = MockResponse(
            _minimal_responses_api_payload("resp_azure_shell_test", "gpt-5-mini"), 200
        )

        await litellm.aresponses(
            model="azure/gpt-5-mini",
            api_base="https://fake-resource.openai.azure.com",
            api_key="fake-api-key",
            api_version="2025-03-01-preview",
            input=expected_body["input"],
            tools=expected_body["tools"],
            tool_choice=expected_body["tool_choice"],
            max_output_tokens=expected_body["max_output_tokens"],
        )

        mock_post.assert_called_once()
        _assert_request_body_matches(mock_post.call_args.kwargs["json"], expected_body)


@pytest.mark.asyncio
async def test_aresponses_azure_shell_tool_400_maps_to_bad_request_error():
    """
    Azure rejects the shell tool for unsupported deployments with a 400;
    litellm must surface that as litellm.BadRequestError carrying the provider message.
    """
    error_body = {
        "error": {
            "message": "Tool of type 'shell' is not supported with this model.",
            "type": "invalid_request_error",
            "param": "tools",
            "code": None,
        }
    }

    def _raise_azure_400(*args, **kwargs):
        response = httpx.Response(
            status_code=400,
            json=error_body,
            request=httpx.Request(
                "POST",
                kwargs.get(
                    "url",
                    "https://fake-resource.openai.azure.com/openai/responses",
                ),
            ),
        )
        response.raise_for_status()

    with patch(
        "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
        new_callable=AsyncMock,
    ) as mock_post:
        mock_post.side_effect = _raise_azure_400

        with pytest.raises(litellm.BadRequestError) as excinfo:
            await litellm.aresponses(
                model="azure/gpt-5-mini",
                api_base="https://fake-resource.openai.azure.com",
                api_key="fake-api-key",
                api_version="2025-03-01-preview",
                input="List files in /mnt/data and run python --version.",
                tools=[{"type": "shell", "environment": {"type": "container_auto"}}],
                tool_choice="auto",
                max_output_tokens=256,
            )

    assert excinfo.value.status_code == 400
    assert "shell" in str(excinfo.value).lower()
    assert "not supported" in str(excinfo.value).lower()


@pytest.mark.asyncio
async def test_aresponses_drops_stream_options():
    """The Responses API rejects include_usage, so include_usage-only stream_options must never reach the wire."""
    with patch(
        "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
        new_callable=AsyncMock,
    ) as mock_post:
        mock_post.return_value = MockResponse(
            _minimal_responses_api_payload("resp_stream_options_test", "gpt-5.5"), 200
        )

        await litellm.aresponses(
            model="openai/gpt-5.5",
            api_key="fake-api-key",
            input="hi",
            stream_options={"include_usage": True},
        )

        mock_post.assert_called_once()
        post_kwargs = mock_post.call_args.kwargs
        request_body = post_kwargs["json"] if "json" in post_kwargs else json.loads(post_kwargs["data"])
        assert "stream_options" not in request_body


@pytest.mark.asyncio
async def test_aresponses_forwards_non_enum_reasoning_effort(
    monkeypatch: pytest.MonkeyPatch, respx_mock: respx.MockRouter
):
    monkeypatch.setenv("OPENAI_API_KEY", "fake-api-key")
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    litellm.in_memory_llm_clients_cache.flush_cache()
    upstream: Final = respx_mock.post("https://api.openai.com/v1/responses").mock(
        return_value=httpx.Response(200, json=_minimal_responses_api_payload("resp_effort_int", "gpt-5.4"))
    )

    response: Final = await litellm.aresponses(model="openai/gpt-5.4", input="hi", reasoning_effort=5)

    assert upstream.call_count == 1
    request_body: Final = json.loads(upstream.calls[0].request.read())
    assert request_body["reasoning"] == {"effort": 5}
    assert response.output[0].content[0].text == "Done."


@pytest.mark.asyncio
async def test_acompletion_with_tools_forwards_non_enum_reasoning_effort_over_the_bridge(
    monkeypatch: pytest.MonkeyPatch, respx_mock: respx.MockRouter
):
    monkeypatch.setenv("OPENAI_API_KEY", "fake-api-key")
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    litellm.in_memory_llm_clients_cache.flush_cache()
    upstream: Final = respx_mock.post("https://api.openai.com/v1/responses").mock(
        return_value=httpx.Response(200, json=_minimal_responses_api_payload("resp_bridge_int", "gpt-5.4"))
    )

    response: Final = await litellm.acompletion(
        model="openai/gpt-5.4",
        messages=[{"role": "user", "content": "What is the weather in Paris?"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
                },
            }
        ],
        reasoning_effort=5,
    )

    assert upstream.call_count == 1
    request_body: Final = json.loads(upstream.calls[0].request.read())
    assert request_body["reasoning"] == {"effort": 5}
    assert response.id == "resp_bridge_int"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model, api_base",
    [
        ("hosted_vllm/my-kimi", "https://vllm.example.test/v1"),
        ("azure_ai/my-kimi", "https://kimi.services.ai.azure.com/openai/v1"),
    ],
)
async def test_aresponses_sends_the_declared_reasoning_effort_level_rounded_up(
    monkeypatch: pytest.MonkeyPatch,
    respx_mock: respx.MockRouter,
    caplog: pytest.LogCaptureFixture,
    model: str,
    api_base: str,
):
    verbose_logger.addHandler(caplog.handler)
    verbose_logger.propagate = False
    verbose_logger.setLevel(logging.INFO)
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    litellm.in_memory_llm_clients_cache.flush_cache()
    upstream: Final = respx_mock.post(url__regex=rf"{api_base}/responses.*").mock(
        return_value=httpx.Response(200, json=_minimal_responses_api_payload("resp_declared_effort", "my-kimi"))
    )

    response: Final = await litellm.aresponses(
        model=model,
        api_base=api_base,
        api_key="fake-api-key",
        input="hi",
        reasoning={"effort": "minimal", "summary": "auto"},
        model_info={"id": "kimi-1", "reasoning_effort_levels": ["low", "high", "max", "none"]},
        litellm_metadata={"model_group": "kimi-k3"},
    )

    assert upstream.call_count == 1
    request_body: Final = json.loads(upstream.calls[0].request.read())
    assert request_body["reasoning"] == {"effort": "low", "summary": "auto"}
    assert response.output[0].content[0].text == "Done."
    assert [
        (
            getattr(record, "reasoning_effort_deployment_id"),
            getattr(record, "reasoning_effort_model_group"),
            getattr(record, "reasoning_effort_sent"),
        )
        for record in caplog.records
        if hasattr(record, "reasoning_effort_sent")
    ] == [("kimi-1", "kimi-k3", "low")]


def _effort_levels_disabled_by(entry: dict) -> tuple[str, ...]:
    return tuple(
        level for level in ("none", "minimal", "low") if entry.get(f"supports_{level}_reasoning_effort") is False
    )


def _gpt_5_row_disabling_a_level(provider: str) -> tuple[str, str]:
    return next(
        (key.removeprefix(f"{provider}/"), _effort_levels_disabled_by(entry)[0])
        for key, entry in litellm.model_cost.items()
        if entry.get("litellm_provider") == provider and "gpt-5" in key and _effort_levels_disabled_by(entry)
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "provider, api_base",
    [("openai", "https://gpt5-gateway.example.test/v1"), ("azure", "https://gpt5.openai.azure.com")],
)
async def test_aresponses_sends_a_declared_level_the_model_map_disables(
    local_model_cost_map: None,
    monkeypatch: pytest.MonkeyPatch,
    respx_mock: respx.MockRouter,
    provider: str,
    api_base: str,
):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "drop_params", False)
    litellm.in_memory_llm_clients_cache.flush_cache()
    model, level = _gpt_5_row_disabling_a_level(provider)
    upstream: Final = respx_mock.post(url__regex=rf"{api_base}/.*responses.*").mock(
        return_value=httpx.Response(200, json=_minimal_responses_api_payload("resp_declared_disabled", model))
    )
    request: Final = {
        "model": f"{provider}/{model}",
        "api_base": api_base,
        "api_version": "2025-04-01-preview",
        "api_key": "fake-api-key",
        "input": "hi",
        "reasoning": {"effort": level},
    }

    with pytest.raises(litellm.UnsupportedParamsError):
        await litellm.aresponses(**request, model_info={"id": "undeclared"})
    await litellm.aresponses(**request, model_info={"id": "declared", "reasoning_effort_levels": [level, "high"]})

    assert upstream.call_count == 1
    assert json.loads(upstream.calls[0].request.read())["reasoning"] == {"effort": level}


@pytest.mark.asyncio
async def test_aresponses_forwards_prompt_managed_reasoning_effort(
    monkeypatch: pytest.MonkeyPatch, respx_mock: respx.MockRouter
):
    from litellm.responses.main import _AsyncPromptManagementOutcome

    monkeypatch.setenv("OPENAI_API_KEY", "fake-api-key")
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    litellm.in_memory_llm_clients_cache.flush_cache()
    upstream: Final = respx_mock.post("https://api.openai.com/v1/responses").mock(
        return_value=httpx.Response(200, json=_minimal_responses_api_payload("resp_prompt_effort", "gpt-5.4"))
    )

    response: Final = await litellm.aresponses(
        model="openai/gpt-5.4",
        input="hi",
        _async_prompt_merged_params=_AsyncPromptManagementOutcome(
            merged_optional_params={"reasoning_effort": 5}, deployment_model_info=None
        ),
    )

    assert upstream.call_count == 1
    request_body: Final = json.loads(upstream.calls[0].request.read())
    assert request_body["reasoning"] == {"effort": 5}
    assert "reasoning_effort" not in request_body
    assert response.output[0].content[0].text == "Done."


@pytest.mark.asyncio
async def test_aresponses_forwards_prompt_managed_reasoning_dict(
    monkeypatch: pytest.MonkeyPatch, respx_mock: respx.MockRouter
):
    from litellm.responses.main import _AsyncPromptManagementOutcome

    monkeypatch.setenv("OPENAI_API_KEY", "fake-api-key")
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    litellm.in_memory_llm_clients_cache.flush_cache()
    upstream: Final = respx_mock.post("https://api.openai.com/v1/responses").mock(
        return_value=httpx.Response(200, json=_minimal_responses_api_payload("resp_prompt_reasoning", "gpt-5.4"))
    )

    response: Final = await litellm.aresponses(
        model="openai/gpt-5.4",
        input="hi",
        _async_prompt_merged_params=_AsyncPromptManagementOutcome(
            merged_optional_params={"reasoning": {"effort": "high", "summary": "detailed"}}, deployment_model_info=None
        ),
    )

    assert upstream.call_count == 1
    request_body: Final = json.loads(upstream.calls[0].request.read())
    assert request_body["reasoning"] == {"effort": "high", "summary": "detailed"}
    assert response.output[0].content[0].text == "Done."


@pytest.mark.asyncio
async def test_aresponses_keeps_include_obfuscation_in_stream_options():
    """include_obfuscation is a valid Responses API stream option and must survive the include_usage strip."""
    with patch(
        "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
        new_callable=AsyncMock,
    ) as mock_post:
        mock_post.return_value = MockResponse(
            _minimal_responses_api_payload("resp_stream_options_obfuscation", "gpt-5.5"), 200
        )

        await litellm.aresponses(
            model="openai/gpt-5.5",
            api_key="fake-api-key",
            input="hi",
            stream_options={"include_usage": True, "include_obfuscation": False},
        )

        mock_post.assert_called_once()
        post_kwargs = mock_post.call_args.kwargs
        request_body = post_kwargs["json"] if "json" in post_kwargs else json.loads(post_kwargs["data"])
        assert request_body["stream_options"] == {"include_obfuscation": False}


@pytest.mark.asyncio
@pytest.mark.parametrize("drop_params", [True, "true"])
async def test_aresponses_request_level_drop_params_drops_bedrock_mantle_service_tier(
    monkeypatch,
    drop_params,
):
    """
    Request-level drop_params=True (as the proxy injects for agentic CLIs) must
    reach the provider config so bedrock_mantle strips the unsupported
    service_tier before the request hits the wire.
    """
    monkeypatch.setattr(litellm, "drop_params", False)

    with patch(
        "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
        new_callable=AsyncMock,
    ) as mock_post:
        mock_post.return_value = MockResponse(
            _minimal_responses_api_payload("resp_mantle_tier_test", "openai.gpt-5.5"),
            200,
        )

        await litellm.aresponses(
            model="bedrock_mantle/openai.gpt-5.5",
            api_key="fake-bearer-token",
            aws_region_name="us-east-1",
            input="hi",
            service_tier="priority",
            drop_params=drop_params,
        )

        mock_post.assert_called_once()
        post_kwargs = mock_post.call_args.kwargs
        request_body = post_kwargs["json"] if "json" in post_kwargs else json.loads(post_kwargs["data"])
        assert "service_tier" not in request_body


@pytest.mark.asyncio
async def test_aresponses_bedrock_mantle_service_tier_raises_without_drop_params(
    monkeypatch,
):
    """
    Without drop_params, an unsupported service_tier must fail fast with an
    error that names drop_params instead of sending a request Mantle rejects.
    """
    monkeypatch.setattr(litellm, "drop_params", False)

    with patch(
        "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
        new_callable=AsyncMock,
    ) as mock_post:
        with pytest.raises(litellm.BadRequestError) as excinfo:
            await litellm.aresponses(
                model="bedrock_mantle/openai.gpt-5.5",
                api_key="fake-bearer-token",
                aws_region_name="us-east-1",
                input="hi",
                service_tier="priority",
            )

        mock_post.assert_not_called()
        assert "drop_params" in str(excinfo.value)
        assert "priority" in str(excinfo.value)


async def _aresponses_and_get_request_headers(**request_kwargs) -> dict:
    with patch(
        "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
        new_callable=AsyncMock,
    ) as mock_post:
        mock_post.return_value = MockResponse(_minimal_responses_api_payload("resp_headers_test", "gpt-4o"), 200)

        await litellm.aresponses(
            model="openai/gpt-4o",
            api_key="fake-api-key",
            input="hi",
            **request_kwargs,
        )

        mock_post.assert_called_once()
        return dict(mock_post.call_args.kwargs["headers"])


@pytest.mark.asyncio
async def test_aresponses_forwards_client_headers_kwarg_to_provider():
    """
    The proxy passes client headers it forwards (`forward_client_headers_to_llm_api`)
    as a `headers` kwarg; those must reach the provider request.
    """
    request_headers = await _aresponses_and_get_request_headers(headers={"x-my-new-header": "hello-from-client"})

    assert request_headers["x-my-new-header"] == "hello-from-client"


@pytest.mark.asyncio
async def test_aresponses_merges_client_headers_with_extra_headers():
    """
    A `headers` kwarg and an explicit `extra_headers` are merged, with
    `extra_headers` winning on conflicts.
    """
    request_headers = await _aresponses_and_get_request_headers(
        headers={"x-my-new-header": "hello-from-client", "x-shared": "from-client"},
        extra_headers={"x-explicit": "from-caller", "x-shared": "from-caller"},
    )

    assert request_headers["x-my-new-header"] == "hello-from-client"
    assert request_headers["x-explicit"] == "from-caller"
    assert request_headers["x-shared"] == "from-caller"


@pytest.mark.asyncio
async def test_aresponses_client_header_conflict_is_case_insensitive():
    """
    HTTP header names are case-insensitive, so a differently cased client header
    must not survive alongside the explicit `extra_headers` value.
    """
    request_headers = await _aresponses_and_get_request_headers(
        headers={"X-Shared": "from-client"},
        extra_headers={"x-shared": "from-caller"},
    )

    assert [name for name in request_headers if name.lower() == "x-shared"] == ["x-shared"]
    assert request_headers["x-shared"] == "from-caller"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model", "custom_llm_provider"),
    [
        ("openai/responses/gpt-5.6", None),
        ("responses/gpt-5.6", "openai"),
    ],
)
async def test_aresponses_strips_responses_routing_prefix_from_openai_model(model, custom_llm_provider):
    """
    `responses/` is LiteLLM routing sugar, never part of the provider model id.
    Deployments configured as openai/responses/<model> reach this path directly via
    /v1/responses and via the /v1/messages adapter (which passes responses/<model>
    with custom_llm_provider="openai"), so both shapes must hit OpenAI as <model>.
    """
    injected_client = AsyncHTTPHandler()
    mock_post = AsyncMock(return_value=MockResponse(_minimal_responses_api_payload("resp_prefix_test", "gpt-5.6"), 200))
    injected_client.post = mock_post

    await litellm.aresponses(
        model=model,
        custom_llm_provider=custom_llm_provider,
        input="ping",
        api_key="sk-test",
        client=injected_client,
    )

    mock_post.assert_called_once()
    assert mock_post.call_args.kwargs["url"].endswith("/responses")
    assert mock_post.call_args.kwargs["json"]["model"] == "gpt-5.6"


@pytest.mark.asyncio
async def test_aresponses_websocket_strips_responses_routing_prefix_from_openai_model():
    from unittest.mock import MagicMock

    from litellm.responses.main import _aresponses_websocket

    with patch.object(
        import_module("litellm.responses.main").base_llm_http_handler,
        "async_responses_websocket",
        new_callable=AsyncMock,
    ) as mock_ws:
        await _aresponses_websocket(
            model="openai/responses/gpt-5.6",
            websocket=MagicMock(),
            api_key="sk-test",
            litellm_logging_obj=MagicMock(),
        )

        mock_ws.assert_awaited_once()
        assert mock_ws.call_args.kwargs["model"] == "gpt-5.6"
        assert mock_ws.call_args.kwargs["custom_llm_provider"] == "openai"


@pytest.mark.asyncio
async def test_aresponses_websocket_keeps_routing_hints_out_of_the_relay_kwargs():  # test-quality-ok: the relay kwargs are the only place a dropped key is observable; the provider socket behind them is the boundary
    from unittest.mock import MagicMock

    from litellm.responses.main import _aresponses_websocket

    with patch.object(
        import_module("litellm.responses.main").base_llm_http_handler,
        "async_responses_websocket",
        new_callable=AsyncMock,
    ) as mock_ws:
        await _aresponses_websocket(
            model="openai/gpt-5.6",
            websocket=MagicMock(),
            api_key="sk-test",
            litellm_logging_obj=MagicMock(),
            input=[{"type": "message", "role": "user", "content": "hi"}],
            previous_response_id="resp_prev",
        )

        mock_ws.assert_awaited_once()
        assert "input" not in mock_ws.call_args.kwargs
        assert "previous_response_id" not in mock_ws.call_args.kwargs


_STRIPPED_WS_INPUT = [{"role": "user", "content": "hi"}]
_ORIGINAL_WS_INPUT = [
    {"type": "reasoning", "id": "rs_1", "encrypted_content": "blob-from-a-removed-deployment", "summary": []},
    *_STRIPPED_WS_INPUT,
]


@pytest.mark.asyncio
@pytest.mark.parametrize("nested", [False, True])
async def test_aresponses_websocket_forwards_the_routed_input_in_the_first_frame(
    nested: bool,
):  # test-quality-ok: the first frame handed to the relay is the only place the routed input is observable before the provider socket
    from unittest.mock import MagicMock

    from litellm.responses.main import _aresponses_websocket

    body = {"model": "gpt-5.6", "input": _ORIGINAL_WS_INPUT, "store": False}
    first_message = json.dumps(
        {"type": "response.create", "response": body} if nested else {"type": "response.create", **body}
    )

    with patch.object(
        import_module("litellm.responses.main").base_llm_http_handler,
        "async_responses_websocket",
        new_callable=AsyncMock,
    ) as mock_ws:
        await _aresponses_websocket(
            model="openai/gpt-5.6",
            websocket=MagicMock(),
            api_key="sk-test",
            litellm_logging_obj=MagicMock(),
            input=list(_STRIPPED_WS_INPUT),
            first_message=first_message,
        )

    forwarded = json.loads(mock_ws.call_args.kwargs["first_message"])
    container = forwarded["response"] if nested else forwarded
    assert container["input"] == _STRIPPED_WS_INPUT
    assert container["store"] is False
    assert container["model"] == "gpt-5.6"
    assert forwarded["type"] == "response.create"


@pytest.mark.asyncio
async def test_aresponses_websocket_forwards_the_first_frame_verbatim_when_routing_left_the_input_alone():  # test-quality-ok: the relay kwargs are the boundary; byte-identical passthrough is only observable there
    from unittest.mock import MagicMock

    from litellm.responses.main import _aresponses_websocket

    first_message = '{"type": "response.create", "model": "gpt-5.6",  "input": [{"role": "user", "content": "hi"}]}'

    with patch.object(
        import_module("litellm.responses.main").base_llm_http_handler,
        "async_responses_websocket",
        new_callable=AsyncMock,
    ) as mock_ws:
        await _aresponses_websocket(
            model="openai/gpt-5.6",
            websocket=MagicMock(),
            api_key="sk-test",
            litellm_logging_obj=MagicMock(),
            input=list(_STRIPPED_WS_INPUT),
            first_message=first_message,
        )

    assert mock_ws.call_args.kwargs["first_message"] == first_message


_INJECTION_POINT_INPUT = [{"role": "system", "content": "You are terse."}, {"role": "user", "content": "hi"}]
_SYSTEM_POINT = {"location": "message", "role": "system"}
_USER_POINT = {"location": "message", "role": "user"}
_SYSTEM_INJECTION_POINT = [_SYSTEM_POINT]
_ANTHROPIC_MESSAGES_PAYLOAD = {
    "id": "msg_1",
    "type": "message",
    "role": "assistant",
    "model": "claude-sonnet-4-5",
    "content": [{"type": "text", "text": "Done."}],
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 10, "output_tokens": 5},
}


def _sent_body(mock_post) -> dict:
    kwargs = mock_post.call_args.kwargs
    return kwargs["json"] if "json" in kwargs else json.loads(kwargs["data"])


@pytest.mark.asyncio
async def test_aresponses_injection_point_marks_input_text_on_gpt_5_6():
    with patch(
        "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
        new_callable=AsyncMock,
    ) as mock_post:
        mock_post.return_value = MockResponse(_minimal_responses_api_payload("resp_pcb_async", "gpt-5.6"), 200)

        await litellm.aresponses(
            model="openai/gpt-5.6",
            api_key="fake-api-key",
            input=copy.deepcopy(_INJECTION_POINT_INPUT),
            cache_control_injection_points=copy.deepcopy(_SYSTEM_INJECTION_POINT),
        )

        body = _sent_body(mock_post)
        assert body["input"][0]["content"][0] == {
            "type": "input_text",
            "text": "You are terse.",
            "prompt_cache_breakpoint": {"mode": "explicit"},
        }
        assert body["input"][1] == {"role": "user", "content": "hi"}
        assert body["prompt_cache_options"] == {"mode": "implicit"}


def test_responses_injection_point_marks_input_text_on_gpt_5_6():
    with patch("litellm.llms.custom_httpx.http_handler.HTTPHandler.post") as mock_post:
        mock_post.return_value = MockResponse(_minimal_responses_api_payload("resp_pcb_sync", "gpt-5.6"), 200)

        litellm.responses(
            model="openai/gpt-5.6",
            api_key="fake-api-key",
            input=copy.deepcopy(_INJECTION_POINT_INPUT),
            cache_control_injection_points=copy.deepcopy(_SYSTEM_INJECTION_POINT),
        )

        body = _sent_body(mock_post)
        assert body["input"][0]["content"][0] == {
            "type": "input_text",
            "text": "You are terse.",
            "prompt_cache_breakpoint": {"mode": "explicit"},
        }
        assert body["input"][1] == {"role": "user", "content": "hi"}
        assert body["prompt_cache_options"] == {"mode": "implicit"}


@pytest.mark.asyncio
async def test_aresponses_injection_point_sends_nothing_extra_below_gpt_5_6():
    with patch(
        "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
        new_callable=AsyncMock,
    ) as mock_post:
        mock_post.return_value = MockResponse(_minimal_responses_api_payload("resp_pcb_old", "gpt-4.1"), 200)

        await litellm.aresponses(
            model="openai/gpt-4.1",
            api_key="fake-api-key",
            input=copy.deepcopy(_INJECTION_POINT_INPUT),
            cache_control_injection_points=copy.deepcopy(_SYSTEM_INJECTION_POINT),
        )

        body = _sent_body(mock_post)
        assert body["input"] == _INJECTION_POINT_INPUT
        assert "prompt_cache_options" not in body
        assert "cache_control" not in json.dumps(body)


@pytest.fixture
def _no_openai_api_base_override(monkeypatch):
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    monkeypatch.setattr(litellm, "api_base", None)


_CUSTOM_API_BASE = "http://127.0.0.1:9/v1"


async def _aresponses_body_with_system_point(**request_kwargs) -> dict:
    with patch(
        "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
        new_callable=AsyncMock,
    ) as mock_post:
        mock_post.return_value = MockResponse(_minimal_responses_api_payload("resp_pcb_gate", "gpt-5.6"), 200)
        await litellm.aresponses(
            api_key="fake-api-key",
            input=copy.deepcopy(_INJECTION_POINT_INPUT),
            cache_control_injection_points=copy.deepcopy(_SYSTEM_INJECTION_POINT),
            **request_kwargs,
        )
        return _sent_body(mock_post)


@pytest.mark.asyncio
@pytest.mark.usefixtures("_no_openai_api_base_override")
async def test_aresponses_litellm_proxy_target_sends_no_openai_markers():
    body = await _aresponses_body_with_system_point(model="litellm_proxy/gpt-5.6", api_base=_CUSTOM_API_BASE)
    assert body["input"] == _INJECTION_POINT_INPUT
    assert "prompt_cache_options" not in body


@pytest.mark.asyncio
@pytest.mark.usefixtures("_no_openai_api_base_override")
async def test_aresponses_custom_api_base_sends_no_openai_markers():
    body = await _aresponses_body_with_system_point(model="gpt-5.6", api_base=_CUSTOM_API_BASE)
    assert body["input"] == _INJECTION_POINT_INPUT
    assert "prompt_cache_options" not in body


@pytest.mark.asyncio
@pytest.mark.usefixtures("_no_openai_api_base_override")
async def test_aresponses_custom_api_base_opts_in_through_prompt_cache_options():
    body = await _aresponses_body_with_system_point(
        model="gpt-5.6", api_base=_CUSTOM_API_BASE, prompt_cache_options={"mode": "explicit"}
    )
    assert body["input"][0]["content"][0] == {
        "type": "input_text",
        "text": "You are terse.",
        "prompt_cache_breakpoint": {"mode": "explicit"},
    }
    assert body["prompt_cache_options"] == {"mode": "explicit"}


@pytest.mark.asyncio
@pytest.mark.usefixtures("_no_openai_api_base_override")
async def test_aresponses_regional_openai_api_base_marks_input_text():
    body = await _aresponses_body_with_system_point(model="gpt-5.6", api_base="https://eu.api.openai.com/v1")
    assert body["input"][0]["content"][0]["prompt_cache_breakpoint"] == {"mode": "explicit"}
    assert body["prompt_cache_options"] == {"mode": "implicit"}


@pytest.mark.usefixtures("_no_openai_api_base_override")
def test_responses_custom_base_url_sends_no_openai_markers():
    with patch("litellm.llms.custom_httpx.http_handler.HTTPHandler.post") as mock_post:
        mock_post.return_value = MockResponse(_minimal_responses_api_payload("resp_pcb_gate_base_url", "gpt-5.6"), 200)

        litellm.responses(
            model="gpt-5.6",
            api_key="fake-api-key",
            base_url=_CUSTOM_API_BASE,
            input=copy.deepcopy(_INJECTION_POINT_INPUT),
            cache_control_injection_points=copy.deepcopy(_SYSTEM_INJECTION_POINT),
        )

        body = _sent_body(mock_post)
        assert body["input"] == _INJECTION_POINT_INPUT
        assert "prompt_cache_options" not in body


@pytest.mark.usefixtures("_no_openai_api_base_override")
def test_responses_custom_api_base_sends_no_openai_markers():
    with patch("litellm.llms.custom_httpx.http_handler.HTTPHandler.post") as mock_post:
        mock_post.return_value = MockResponse(_minimal_responses_api_payload("resp_pcb_gate_sync", "gpt-5.6"), 200)

        litellm.responses(
            model="gpt-5.6",
            api_key="fake-api-key",
            api_base=_CUSTOM_API_BASE,
            input=copy.deepcopy(_INJECTION_POINT_INPUT),
            cache_control_injection_points=copy.deepcopy(_SYSTEM_INJECTION_POINT),
        )

        body = _sent_body(mock_post)
        assert body["input"] == _INJECTION_POINT_INPUT
        assert "prompt_cache_options" not in body


@pytest.mark.asyncio
async def test_injection_points_still_reach_a_native_responses_provider():
    """Providers that serve Responses natively never reach the chat-completions bridge,
    so this layer is their only chance to inject and must keep doing so."""
    injected_client = AsyncHTTPHandler()
    mock_post = AsyncMock(return_value=MockResponse(_minimal_responses_api_payload("resp_native", "gpt-5.6"), 200))
    injected_client.post = mock_post

    await litellm.aresponses(
        model="openai/gpt-5.6",
        api_key="fake-api-key",
        input=copy.deepcopy(_INJECTION_POINT_INPUT),
        cache_control_injection_points=copy.deepcopy(_SYSTEM_INJECTION_POINT),
        client=injected_client,
    )

    body = _sent_body(mock_post)
    assert body["input"][0]["content"][0]["prompt_cache_breakpoint"] == {"mode": "explicit"}
    assert "cache_control_injection_points" not in body


@pytest.mark.asyncio
async def test_injection_points_reach_a_provider_prefixed_native_responses_model():
    """Bedrock Mantle resolves its Responses config from the price map by bare model name,
    so predicting the bridge with the ``bedrock_mantle/``-prefixed name read as bridged and
    deferred the system point to a chat-completions pass this native path never runs."""
    injected_client = AsyncHTTPHandler()
    mock_post = AsyncMock(
        return_value=MockResponse(_minimal_responses_api_payload("resp_mantle", "openai.gpt-5.6-sol"), 200)
    )
    injected_client.post = mock_post

    await litellm.aresponses(
        model="bedrock_mantle/openai.gpt-5.6-sol",
        api_key="fake-bearer-token",
        aws_region_name="us-east-1",
        input=copy.deepcopy(_INJECTION_POINT_INPUT),
        cache_control_injection_points=copy.deepcopy(_SYSTEM_INJECTION_POINT),
        client=injected_client,
    )

    assert mock_post.call_args.kwargs["url"].endswith("/openai/v1/responses")
    body = _sent_body(mock_post)
    assert body["input"][0]["content"][0]["prompt_cache_breakpoint"] == {"mode": "explicit"}
    assert body["prompt_cache_options"] == {"mode": "implicit"}
    assert "cache_control_injection_points" not in body


@pytest.mark.asyncio
async def test_injection_points_reach_a_foundry_deployment_of_an_openai_model(monkeypatch):
    """The router hands this layer the provider it resolved with the deployment's api_base, but the
    hook reads the provider from the request kwargs, which never carry it, and resolving
    ``azure_ai/gpt-6-astra`` by name alone reads the ambient ``AZURE_AI_API_BASE``, so next to an
    Azure OpenAI one the Foundry deployment got Anthropic marks the Responses transform then stripped."""
    monkeypatch.setenv("AZURE_AI_API_BASE", "https://other-deployment.openai.azure.com")
    injected_client = AsyncHTTPHandler()
    mock_post = AsyncMock(return_value=MockResponse(_minimal_responses_api_payload("resp_foundry", "gpt-6-astra"), 200))
    injected_client.post = mock_post

    await litellm.aresponses(
        model="azure_ai/gpt-6-astra",
        custom_llm_provider="azure_ai",
        api_key="fake-api-key",
        api_base="https://foundry.services.ai.azure.com",
        input=copy.deepcopy(_INJECTION_POINT_INPUT),
        cache_control_injection_points=copy.deepcopy(_SYSTEM_INJECTION_POINT),
        client=injected_client,
    )

    body = _sent_body(mock_post)
    assert body["input"][0]["content"][0]["prompt_cache_breakpoint"] == {"mode": "explicit"}
    assert body["prompt_cache_options"] == {"mode": "implicit"}
    assert "cache_control_injection_points" not in body


async def _bridged_body(mock_post, *, points, input, instructions="You are a documentation assistant."):
    injected_client = AsyncHTTPHandler()
    injected_client.post = mock_post

    await litellm.aresponses(
        model="anthropic/claude-sonnet-4-5",
        api_key="fake-api-key",
        instructions=instructions,
        input=copy.deepcopy(input),
        cache_control_injection_points=copy.deepcopy(points),
        client=injected_client,
    )
    return _sent_body(mock_post)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content",
    [
        pytest.param("hi", id="string-content"),
        pytest.param([{"type": "input_text", "text": "hi there friend"}], id="list-content"),
    ],
)
@pytest.mark.parametrize(
    "points",
    [
        pytest.param([_SYSTEM_POINT], id="system-only"),
        pytest.param([_USER_POINT, _SYSTEM_POINT], id="mixed-user-and-system"),
    ],
)
async def test_instructions_are_marked_when_the_bridge_builds_the_system_message(points, content):
    """The system prompt lives in `instructions`, which is not a message until the bridge
    builds one, so the point targeting it matches nothing at the Responses layer.

    Carrying it forward is what marks it at all. Carrying it *stamped* is what keeps a
    second point that did match from stranding it: without the stamp the next pass reads
    litellm's own marks as client breakpoints and stands the whole configuration down.
    """
    mock_post = AsyncMock(return_value=MockResponse(_ANTHROPIC_MESSAGES_PAYLOAD, 200))
    body = await _bridged_body(mock_post, points=points, input=[{"role": "user", "content": content}])

    assert body["system"][0]["cache_control"] == {"type": "ephemeral"}


@pytest.mark.asyncio
@pytest.mark.parametrize("instructions", [None, "You are a documentation assistant."])
async def test_positional_points_address_the_input_item_the_caller_indexed(instructions):
    """`index` counts the caller's `input` items, and the Responses layer is where that
    list still is, so a matched positional point must be spent there and never re-resolved
    against the bridge's list, where the system message shifts every ordinal by one."""
    mock_post = AsyncMock(return_value=MockResponse(_ANTHROPIC_MESSAGES_PAYLOAD, 200))
    body = await _bridged_body(
        mock_post,
        points=[{"location": "message", "index": 0}],
        input=[{"role": "user", "content": [{"type": "input_text", "text": "hi there friend"}]}],
        instructions=instructions,
    )

    assert body["messages"][0]["content"][0]["cache_control"] == {"type": "ephemeral"}
    if instructions:
        assert "cache_control" not in json.dumps(body["system"])


@pytest.mark.asyncio
async def test_out_of_bounds_positional_points_are_not_revived_by_a_longer_list():
    """An ordinal addresses the list in front of the pass that reads it.

    Carrying one forward would re-resolve it against the bridge's longer list, where an
    index that named nothing in the caller's `input` can land on a real message -- the
    system prompt included. Positional points are resolved where they were written or not
    at all.
    """
    mock_post = AsyncMock(return_value=MockResponse(_ANTHROPIC_MESSAGES_PAYLOAD, 200))
    body = await _bridged_body(
        mock_post,
        points=[{"location": "message", "index": 1}],
        input=[{"role": "user", "content": [{"type": "input_text", "text": "only item"}]}],
    )

    assert "cache_control" not in json.dumps(body["system"])
    assert "cache_control" not in json.dumps(body["messages"])


def _four_user_turns() -> list:
    return [
        item
        for i in range(4)
        for item in (
            {"role": "user", "content": [{"type": "input_text", "text": f"msg{i}"}]},
            {"role": "assistant", "content": [{"type": "output_text", "text": f"reply{i}", "annotations": []}]},
        )
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "points,instructions,system_marked,marked_messages",
    [
        pytest.param([_SYSTEM_POINT, _USER_POINT], "You are terse.", True, [0, 2, 4], id="earlier-point-wins"),
        pytest.param([_USER_POINT, _SYSTEM_POINT], "You are terse.", False, [0, 2, 4, 6], id="reversed-order-reverses"),
        pytest.param([_USER_POINT, _SYSTEM_POINT], None, False, [0, 2, 4, 6], id="target-never-built-costs-nothing"),
    ],
)
async def test_config_order_decides_who_wins_the_shared_breakpoint_budget(
    points, instructions, system_marked, marked_messages
):
    """Injection points are honoured in config order, earlier ones winning scarce slots.

    A role-targeted point is placed a pass later than a positional one, so the four
    breakpoints it competes for are shared across both passes. Every role point being
    settled in the pass that holds the final list -- rather than the earlier pass holding
    a slot for one it cannot place -- is what keeps that competition ordered in both
    directions, and what stops a point whose target is never built from costing anything.
    """
    mock_post = AsyncMock(return_value=MockResponse(_ANTHROPIC_MESSAGES_PAYLOAD, 200))
    body = await _bridged_body(mock_post, points=points, input=_four_user_turns(), instructions=instructions)

    assert ("cache_control" in json.dumps(body.get("system", []))) is system_marked
    assert [i for i, msg in enumerate(body["messages"]) if "cache_control" in json.dumps(msg)] == marked_messages


@pytest.mark.asyncio
async def test_a_native_responses_provider_places_every_point_itself():
    """A provider serving Responses natively gets no second pass.

    This layer is the last one that can place anything, so handing a point forward here
    drops it -- and an unmatchable point must not cost a matching one its slot either.
    The request has to be known to be bridged before anything is deferred.
    """
    input_items = _four_user_turns()

    async def _marked_indices(points):
        injected_client = AsyncHTTPHandler()
        mock_post = AsyncMock(return_value=MockResponse(_minimal_responses_api_payload("resp_native", "gpt-5.6"), 200))
        injected_client.post = mock_post
        await litellm.aresponses(
            model="openai/gpt-5.6",
            api_key="fake-api-key",
            input=copy.deepcopy(input_items),
            cache_control_injection_points=copy.deepcopy(points),
            client=injected_client,
        )
        body = _sent_body(mock_post)
        return [i for i, item in enumerate(body["input"]) if "prompt_cache_breakpoint" in json.dumps(item)]

    user_only = await _marked_indices([_USER_POINT])
    # The system point can never match here: nothing turns `instructions` into a message
    # on the native path, so it must not cost the user point a slot.
    with_unmatchable_system = await _marked_indices([_SYSTEM_POINT, _USER_POINT])

    assert user_only == [0, 2, 4, 6]
    assert with_unmatchable_system == user_only


_CLAUDE_COMPACTION_ITEM: Final = {
    "type": "compaction",
    "id": "cmp_claude",
    "encrypted_content": json.dumps({"type": "compaction", "content": "The user's cat is named Biscuit."}),
}
_OPENAI_COMPACTION_ITEM: Final = {"type": "compaction", "id": "cmp_openai", "encrypted_content": "gAAAAABopaque"}
_HISTORY: Final = (
    {"role": "system", "content": "Answer briefly."},
    {"role": "user", "content": "My cat is named Biscuit."},
    {"role": "assistant", "content": "Noted."},
)
_NEXT_TURN: Final = {"role": "user", "content": "What is my cat's name?"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("call", "upstream_url"),
    [
        pytest.param(litellm.aresponses, "https://api.openai.com/v1/responses", id="create"),
        pytest.param(litellm.acompact_responses, "https://api.openai.com/v1/responses/compact", id="compact"),
    ],
)
@pytest.mark.parametrize(
    ("compaction_item", "sent_input"),
    [
        (
            _CLAUDE_COMPACTION_ITEM,
            [
                _HISTORY[0],
                {"role": "user", "content": "The user's cat is named Biscuit.", "type": "message"},
                _NEXT_TURN,
            ],
        ),
        (_OPENAI_COMPACTION_ITEM, [*_HISTORY, _OPENAI_COMPACTION_ITEM, _NEXT_TURN]),
    ],
)
async def test_aresponses_sends_a_claude_compaction_item_to_openai_as_its_summary(
    monkeypatch: pytest.MonkeyPatch,
    respx_mock: respx.MockRouter,
    call: Callable[..., Awaitable[object]],
    upstream_url: str,
    compaction_item: dict[str, str],
    sent_input: list[dict[str, str]],
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    litellm.in_memory_llm_clients_cache.flush_cache()
    upstream: Final = respx_mock.post(upstream_url).mock(
        return_value=httpx.Response(200, json=_minimal_responses_api_payload("resp_compacted", "gpt-5.6"))
    )

    await call(model="openai/gpt-5.6", api_key="fake-api-key", input=[*_HISTORY, compaction_item, _NEXT_TURN])

    assert json.loads(upstream.calls[0].request.read())["input"] == sent_input


def test_non_streamed_response_ignores_a_stream_timeout_it_never_uses() -> None:
    answer: Final = json.dumps(_minimal_responses_api_payload("resp_unstreamed", "gpt-5.6")).encode()
    client: Final = HTTPHandler(
        client=httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, content=answer)))
    )

    response: Final = litellm.responses(
        model="openai/gpt-5.6", input="hi", stream_timeout="unset", api_key="fake-key", client=client
    )

    assert response.output[0].content[0].text == "Done."


def test_faked_stream_accepts_a_text_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(
        litellm.model_cost,
        "slow-faked-responses-model",
        {"litellm_provider": "openai", "mode": "responses", "supports_native_streaming": False},
    )
    answer: Final = json.dumps(_minimal_responses_api_payload("resp_faked", "slow-faked-responses-model")).encode()

    response: Final = litellm.responses(
        model="openai/slow-faked-responses-model",
        input="hi",
        stream=True,
        timeout=str(UPSTREAM_FIRST_BYTE_SECONDS * 2),
        stream_timeout=STREAM_TIMEOUT_SECONDS,
        api_key="fake-key",
        client=slow_upstream_sync_client(answer),
    )

    assert [event.response.output[0].content[0].text for event in response if event.type == "response.completed"] == [
        "Done."
    ]


async def _invoke_responses(sync_mode: bool, model: str, pricing: Mapping[str, float]) -> ResponsesAPIResponse:
    response: Final = (
        litellm.responses(model=model, input="hi", **pricing)
        if sync_mode
        else await litellm.aresponses(model=model, input="hi", **pricing)
    )
    assert isinstance(response, ResponsesAPIResponse)
    return response


def _response_cost(response: ResponsesAPIResponse) -> float:
    hidden_params: Final[dict[str, float]] = getattr(response, "_hidden_params")
    return hidden_params["response_cost"]


@pytest.mark.asyncio
@pytest.mark.parametrize("sync_mode", (True, False))
@pytest.mark.parametrize("model", ("openai/gpt-5.4-nano", "openai/custom-priced-model-not-in-cost-map"))
async def test_responses_bills_per_call_custom_pricing_from_usage(
    monkeypatch: pytest.MonkeyPatch, respx_mock: respx.MockRouter, sync_mode: bool, model: str
):
    monkeypatch.setenv("OPENAI_API_KEY", "fake-api-key")
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    model_cost: Final[dict[str, object]] = getattr(litellm, "model_cost")
    monkeypatch.setattr(litellm, "model_cost", dict(model_cost))
    litellm.in_memory_llm_clients_cache.flush_cache()
    respx_mock.post("https://api.openai.com/v1/responses").mock(
        return_value=httpx.Response(200, json=_minimal_responses_api_payload("resp_custom_price", model.split("/")[1]))
    )
    pricing: Final = {"input_cost_per_token": 1e-5, "output_cost_per_token": 2e-5}

    response: Final = await _invoke_responses(sync_mode, model, pricing)

    assert response.usage is not None
    expected: Final = (
        response.usage.input_tokens * pricing["input_cost_per_token"]
        + response.usage.output_tokens * pricing["output_cost_per_token"]
    )
    assert math.isclose(_response_cost(response), expected)


@pytest.mark.asyncio
async def test_concurrent_responses_calls_keep_their_own_per_call_prices(
    monkeypatch: pytest.MonkeyPatch, respx_mock: respx.MockRouter
):
    monkeypatch.setenv("OPENAI_API_KEY", "fake-api-key")
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    litellm.in_memory_llm_clients_cache.flush_cache()
    model: Final = "openai/gpt-5.4-nano"
    model_cost: Final[dict[str, object]] = getattr(litellm, "model_cost")
    cost_map_row_before: Final = copy.deepcopy(model_cost.get(model))
    respx_mock.post("https://api.openai.com/v1/responses").mock(
        return_value=httpx.Response(200, json=_minimal_responses_api_payload("resp_concurrent", "gpt-5.4-nano"))
    )
    rates: Final = ((1e-5, 2e-5), (3e-5, 4e-5))

    responses: Final = await asyncio.gather(
        *(
            _invoke_responses(False, model, {"input_cost_per_token": in_rate, "output_cost_per_token": out_rate})
            for in_rate, out_rate in rates
        )
    )

    for response, (in_rate, out_rate) in zip(responses, rates, strict=True):
        assert response.usage is not None
        assert math.isclose(
            _response_cost(response), response.usage.input_tokens * in_rate + response.usage.output_tokens * out_rate
        )
    assert model_cost.get(model) == cost_map_row_before


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    ("api_base", "explicit_header", "expected_session"),
    [
        ("https://opencode.ai/inference/go/openai/v1", None, "conversation-123"),
        ("https://opencode.ai/inference/go/openai/v1", "caller-override", "caller-override"),
        ("https://other.example/v1", None, None),
    ],
)
@pytest.mark.asyncio
async def test_responses_opencode_session_header(
    asynchronous: bool, stream: bool, api_base: str, explicit_header: str | None, expected_session: str | None
) -> None:
    received: Final[asyncio.Queue[httpx.Request]] = asyncio.Queue()
    payload: Final = _minimal_responses_api_payload("resp_test", "gpt-6-luna")

    def respond(request: httpx.Request) -> httpx.Response:
        received.put_nowait(request)
        return (
            httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                text="".join(
                    f"event: {kind}\ndata: {json.dumps({'type': kind, 'response': payload})}\n\n"
                    for kind in ("response.created", "response.completed")
                ),
            )
            if stream
            else httpx.Response(200, json=payload)
        )

    args: Final = {
        "model": "openai/gpt-6-luna",
        "input": "hello",
        "api_base": api_base,
        "api_key": "test-key",
        "litellm_session_id": "conversation-123",
        "stream": stream,
        "extra_headers": {"X-OpenCode-Session": explicit_header} if explicit_header else {"x-existing": "unchanged"},
    }
    if asynchronous:
        async_client: Final = AsyncHTTPHandler(transport=httpx.MockTransport(respond))
        try:
            response: Final = await litellm.aresponses(**args, client=async_client)
            if stream:
                events: Final = tuple([event.type async for event in response])
                assert "response.completed" in events
            else:
                assert response.output[0].content[0].text == "Done."
        finally:
            await async_client.close()
    else:
        sync_client: Final = HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(respond)))
        try:
            result: Final = litellm.responses(**args, client=sync_client)
            if stream:
                assert "response.completed" in tuple(event.type for event in result)
            else:
                assert result.output[0].content[0].text == "Done."
        finally:
            sync_client.close()
    sent: Final = received.get_nowait()
    assert str(sent.url) == f"{api_base}/responses"
    assert sent.headers.get("x-opencode-session") == expected_session
    assert sent.headers["authorization"] == "Bearer test-key"
    assert json.loads(sent.content) == {"model": "gpt-6-luna", "input": "hello", "stream": stream}
    assert received.empty()


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("source", ["OPENAI_BASE_URL", "OPENAI_API_BASE", "litellm.api_base"])
@pytest.mark.parametrize("explicit_base", [None, "https://other.example/v1"])
@pytest.mark.asyncio
async def test_responses_opencode_session_uses_effective_destination(
    monkeypatch: pytest.MonkeyPatch,
    respx_mock: respx.MockRouter,
    asynchronous: bool,
    source: str,
    explicit_base: str | None,
) -> None:
    default_base: Final = "https://opencode.ai/inference/go/openai/v1"
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    monkeypatch.setattr(litellm, "api_base", default_base if source == "litellm.api_base" else None)
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    if source != "litellm.api_base":
        monkeypatch.setenv(source, default_base)
    endpoint: Final = respx_mock.post(f"{explicit_base or default_base}/responses").respond(
        json=_minimal_responses_api_payload("resp_test", "gpt-6-luna")
    )
    args: Final = {
        "model": "openai/gpt-6-luna",
        "input": "hello",
        "api_base": explicit_base,
        "api_key": "test-key",
        "litellm_session_id": "conversation-123",
    }
    response: Final = await litellm.aresponses(**args) if asynchronous else litellm.responses(**args)
    assert response.output[0].content[0].text == "Done."
    assert endpoint.call_count == 1
    sent: Final = endpoint.calls[0].request
    assert sent.headers.get("x-opencode-session") == (None if explicit_base else "conversation-123")
    assert sent.headers["authorization"] == "Bearer test-key"
    assert json.loads(sent.content) == {"model": "gpt-6-luna", "input": "hello"}


_JSON_OBJECT: Final = TypeAdapter(dict[str, object])


@pytest.mark.asyncio
async def test_router_responses_deployment_pricing_keeps_inherited_cache_rates(
    monkeypatch: pytest.MonkeyPatch, respx_mock: respx.MockRouter
):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    model_cost: Final[dict[str, object]] = getattr(litellm, "model_cost")
    monkeypatch.setattr(litellm, "model_cost", dict(model_cost))
    litellm.in_memory_llm_clients_cache.flush_cache()
    rates: Final = {"input_cost_per_token": 1e-5, "output_cost_per_token": 2e-5}
    router: Final = litellm.Router(
        model_list=[
            {
                "model_name": "nano",
                "litellm_params": {"model": "openai/gpt-5.4-nano", "api_key": "fake-api-key", **rates},
                "model_info": {"id": "nano-deployment"},
            }
        ]
    )
    payload: Final[dict[str, object]] = _minimal_responses_api_payload("resp_router_cache", "gpt-5.4-nano")
    usage: Final = _JSON_OBJECT.validate_python(payload["usage"])
    cached_tokens: Final = 8
    respx_mock.post("https://api.openai.com/v1/responses").mock(
        return_value=httpx.Response(
            200,
            json={**payload, "usage": {**usage, "input_tokens_details": {"cached_tokens": cached_tokens}}},
        )
    )

    response: Final = await router.aresponses(model="nano", input="hi")

    assert isinstance(response, ResponsesAPIResponse)
    assert response.usage is not None
    uncached_input: Final = response.usage.input_tokens - cached_tokens
    full_price: Final = (
        response.usage.input_tokens * rates["input_cost_per_token"]
        + response.usage.output_tokens * rates["output_cost_per_token"]
    )
    floor: Final = (
        uncached_input * rates["input_cost_per_token"] + response.usage.output_tokens * rates["output_cost_per_token"]
    )
    assert floor < _response_cost(response) < full_price
