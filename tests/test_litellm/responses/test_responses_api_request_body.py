"""
Test that litellm.responses() / litellm.aresponses() send the expected request body
over the wire and surface provider errors correctly. Expected JSON bodies are stored
in expected_responses_api_request/.
"""

import copy
import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
import pytest

import litellm
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler


def _expected_dir() -> Path:
    """Path to expected_responses_api_request folder (sibling of test_litellm/responses)."""
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
async def test_aresponses_request_level_drop_params_drops_bedrock_mantle_service_tier(
    monkeypatch,
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
            drop_params=True,
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

    with patch(
        "litellm.responses.main.base_llm_http_handler.async_responses_websocket",
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
        assert body["prompt_cache_options"] == {"mode": "explicit"}


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
        assert body["prompt_cache_options"] == {"mode": "explicit"}


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
    assert body["prompt_cache_options"] == {"mode": "explicit"}


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
