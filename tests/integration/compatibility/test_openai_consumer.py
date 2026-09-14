import json
import uuid
from importlib.metadata import version
from typing import Final

import httpx
import pytest
from openai import AsyncOpenAI, OpenAI

from integration._support.client import Gateway
from integration._support.wire import Reply, Request, wire_server


@pytest.mark.covers("other.compatibility.openai.retained_client_parses_tools_and_usage")
async def test_retained_openai_clients_parse_real_proxy_tool_and_usage_responses(gateway: Gateway) -> None:
    assert version("openai") == "2.33.0", "Retain this consumer version independently before upgrading the candidate lock"
    def provider(request: Request) -> Reply:
        assert request.method == "POST" and request.target == "/v1/chat/completions"
        body: Final = json.loads(request.body)
        assert body["tools"][0]["function"]["name"] == "add"
        return Reply(body=json.dumps({"id": "chatcmpl-" + uuid.uuid4().hex, "object": "chat.completion", "created": 1, "model": "gpt-4o-mini", "choices": [{"index": 0, "message": {"role": "assistant", "content": None, "tool_calls": [{"id": "synthetic-call", "type": "function", "function": {"name": "add", "arguments": '{"a":3,"b":5}'}}]}, "finish_reason": "tool_calls"}], "usage": {"prompt_tokens": 11, "completion_tokens": 4, "total_tokens": 15}}).encode())

    with wire_server(provider) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=wire.url + "/v1")
        key: Final = scenario.key(models=[model])
        parameters: Final = {"model": model, "messages": [{"role": "user", "content": "synthetic tool request"}], "tools": [{"type": "function", "function": {"name": "add", "parameters": {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}, "required": ["a", "b"]}}}], "extra_body": {"cache": {"no-cache": True}}}
        with OpenAI(api_key=key, base_url=str(gateway.client.base_url).rstrip("/") + "/v1", max_retries=0, http_client=httpx.Client(timeout=10, trust_env=False)) as sync:
            first: Final = sync.chat.completions.create(**parameters)
        async with AsyncOpenAI(api_key=key, base_url=str(gateway.client.base_url).rstrip("/") + "/v1", max_retries=0, http_client=httpx.AsyncClient(timeout=10, trust_env=False)) as asynchronous:
            second: Final = await asynchronous.chat.completions.create(**parameters)
        assert first.id != second.id
        for response in (first, second):
            assert response.object == "chat.completion" and response.choices[0].finish_reason == "tool_calls"
            call: Final = response.choices[0].message.tool_calls[0]
            assert call.id == "synthetic-call" and call.function.name == "add"
            assert json.loads(call.function.arguments) == {"a": 3, "b": 5}
            assert response.usage.prompt_tokens == 11 and response.usage.completion_tokens == 4 and response.usage.total_tokens == 15
        assert len(wire.drain()) == 2
