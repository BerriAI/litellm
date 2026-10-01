"""Tests for Vertex AI partner count_tokens system/tools forwarding.

Regression coverage for issue #44051.
"""

from unittest.mock import AsyncMock, Mock

import pytest

from litellm.llms.vertex_ai.common_utils import VertexAITokenCounter


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("system", "tools"),
    [
        ("You are a careful assistant.", None),
        (
            None,
            [
                {
                    "name": "noop",
                    "description": "Do nothing",
                    "input_schema": {"type": "object", "properties": {}},
                }
            ],
        ),
        (
            "You are a careful assistant.",
            [
                {
                    "name": "noop",
                    "description": "Do nothing",
                    "input_schema": {"type": "object", "properties": {}},
                }
            ],
        ),
        (None, None),
    ],
)
async def test_vertex_partner_count_tokens_forwards_system_and_tools(
    monkeypatch,
    system,
    tools,
) -> None:
    from litellm.llms.vertex_ai.vertex_ai_partner_models.count_tokens import handler

    token_counter_class = handler.VertexAIPartnerModelsTokenCounter
    model = "claude-3-5-sonnet-v2"
    messages = [{"role": "user", "content": "Hello"}]
    response = Mock(status_code=200, text="")
    response.json.return_value = {"input_tokens": 37}
    http_client = Mock()
    http_client.post = AsyncMock(return_value=response)

    monkeypatch.setattr(
        token_counter_class,
        "_ensure_access_token_async",
        AsyncMock(return_value=("test-token", "test-project")),
    )
    monkeypatch.setattr(
        token_counter_class,
        "_build_count_tokens_endpoint",
        Mock(return_value="https://vertex.example/count_tokens"),
    )
    monkeypatch.setattr(handler, "get_async_httpx_client", lambda **_: http_client)

    result = await VertexAITokenCounter().count_tokens(
        model_to_use=model,
        messages=messages,
        contents=None,
        deployment={"litellm_params": {"vertex_location": "us-east5"}},
        request_model=model,
        system=system,
        tools=tools,
    )

    assert result is not None
    request_json = http_client.post.await_args.kwargs["json"]
    if system is not None:
        assert request_json.get("system") == system
    if tools is not None:
        assert request_json.get("tools") == tools
    else:
        assert "tools" not in request_json
    if system is None:
        assert "system" not in request_json
    if system is None and tools is None:
        assert request_json == {"model": model, "messages": messages}
