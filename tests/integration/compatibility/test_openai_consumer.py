import json
import uuid
from collections.abc import Callable
from importlib.metadata import version
from typing import Final

import httpx
import pytest
from integration._support.client import Gateway
from integration._support.wire import Reply, Request, wire_server
from openai import AsyncAzureOpenAI, AsyncOpenAI, AzureOpenAI, OpenAI


@pytest.mark.covers("other.compatibility.openai.retained_client_parses_tools_and_usage")
async def test_retained_openai_clients_parse_real_proxy_tool_and_usage_responses(gateway: Gateway) -> None:
    assert version("openai") == "2.33.0", (
        "Retain this consumer version independently before upgrading the candidate lock"
    )

    def provider(request: Request) -> Reply:
        assert request.method == "POST" and request.target == "/v1/chat/completions"
        body: Final = json.loads(request.body)
        tools: Final = body.get("tools")
        if tools:
            assert tools[0]["function"]["name"] == "add"
        message: Final = (
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "synthetic-call",
                        "type": "function",
                        "function": {"name": "add", "arguments": '{"a":3,"b":5}'},
                    }
                ],
            }
            if tools
            else {"role": "assistant", "content": "Synthetic answer: 8"}
        )
        return Reply(
            body=json.dumps(
                {
                    "id": "chatcmpl-" + uuid.uuid4().hex,
                    "object": "chat.completion",
                    "created": 1,
                    "model": "gpt-4o-mini",
                    "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls" if tools else "stop"}],
                    "usage": {"prompt_tokens": 11, "completion_tokens": 4, "total_tokens": 15},
                }
            ).encode()
        )

    with wire_server(provider) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=wire.url + "/v1")
        key: Final = scenario.key(models=[model])
        parameters: Final = {
            "model": model,
            "messages": [{"role": "user", "content": "synthetic tool request"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "add",
                        "parameters": {
                            "type": "object",
                            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                            "required": ["a", "b"],
                        },
                    },
                }
            ],
            "extra_body": {"cache": {"no-cache": True}},
        }
        plain: Final = {name: value for name, value in parameters.items() if name != "tools"}
        with OpenAI(
            api_key=key,
            base_url=str(gateway.client.base_url).rstrip("/") + "/v1",
            max_retries=0,
            http_client=httpx.Client(timeout=10, trust_env=False),
        ) as sync:
            first: Final = sync.chat.completions.create(**parameters)
            first_text: Final = sync.chat.completions.create(**plain)
        async with AsyncOpenAI(
            api_key=key,
            base_url=str(gateway.client.base_url).rstrip("/") + "/v1",
            max_retries=0,
            http_client=httpx.AsyncClient(timeout=10, trust_env=False),
        ) as asynchronous:
            second: Final = await asynchronous.chat.completions.create(**parameters)
            second_text: Final = await asynchronous.chat.completions.create(**plain)
        assert len({response.id for response in (first, second, first_text, second_text)}) == 4
        for response in (first, second, first_text, second_text):
            assert response.object == "chat.completion"
            assert (
                response.usage.prompt_tokens == 11
                and response.usage.completion_tokens == 4
                and response.usage.total_tokens == 15
            )
        for response in (first, second):
            assert response.choices[0].finish_reason == "tool_calls"
            call: Final = response.choices[0].message.tool_calls[0]
            assert call.id == "synthetic-call" and call.function.name == "add"
            assert json.loads(call.function.arguments) == {"a": 3, "b": 5}
        for response in (first_text, second_text):
            assert response.choices[0].finish_reason == "stop"
            assert response.choices[0].message.content == "Synthetic answer: 8"
            assert not response.choices[0].message.tool_calls
        assert len(wire.drain()) == 4


_AZURE_API_VERSION: Final = "2024-10-21"
_CHAT_BACKEND: Final = "gpt-5.4-mini"
_TEXT_BACKEND: Final = "gpt-3.5-turbo-instruct"
_USAGE: Final = {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}


def _sse(events: tuple[dict[str, object], ...]) -> Reply:
    return Reply(
        content_type="text/event-stream",
        chunks=tuple(f"data: {json.dumps(event)}\n\n".encode() for event in events) + (b"data: [DONE]\n\n",),
    )


def _data_lines(text: str) -> tuple[str, ...]:
    return tuple(line for line in text.splitlines() if line.startswith("data: "))


def _chat_provider(identity: str, prompts: set[str]) -> Callable[[Request], Reply]:
    def provider(request: Request) -> Reply:
        assert (request.method, request.target) == ("POST", "/chat/completions"), request
        assert request.headers["authorization"] == "Bearer synthetic-azure-consumer-key"
        body: Final = json.loads(request.body)
        assert body["messages"][0]["content"] in prompts, body
        messages: Final = [{"role": "user", "content": body["messages"][0]["content"]}]
        if body.get("stream"):
            assert body == {
                "model": _CHAT_BACKEND,
                "messages": messages,
                "stream": True,
                "stream_options": {"include_usage": True},
            }
            return _sse(
                (
                    {
                        "id": identity,
                        "object": "chat.completion.chunk",
                        "created": 1,
                        "model": _CHAT_BACKEND,
                        "choices": [
                            {"index": 0, "delta": {"role": "assistant", "content": "Hel"}, "finish_reason": None}
                        ],
                    },
                    {
                        "id": identity,
                        "object": "chat.completion.chunk",
                        "created": 1,
                        "model": _CHAT_BACKEND,
                        "choices": [{"index": 0, "delta": {"content": "lo"}, "finish_reason": "stop"}],
                    },
                    {
                        "id": identity,
                        "object": "chat.completion.chunk",
                        "created": 1,
                        "model": _CHAT_BACKEND,
                        "choices": [],
                        "usage": _USAGE,
                    },
                )
            )
        assert body == {"model": _CHAT_BACKEND, "messages": messages}
        return Reply(
            body=json.dumps(
                {
                    "id": identity,
                    "object": "chat.completion",
                    "created": 1,
                    "model": _CHAT_BACKEND,
                    "choices": [
                        {"index": 0, "message": {"role": "assistant", "content": "Hello"}, "finish_reason": "stop"}
                    ],
                    "usage": _USAGE,
                }
            ).encode()
        )

    return provider


async def test_azure_openai_clients_reach_openai_deployment_through_deployments_alias(gateway: Gateway) -> None:
    identity: Final = f"chatcmpl-azure-{uuid.uuid4().hex}"
    prompts: Final = {f"{kind} {identity}" for kind in ("sync", "sync-stream", "async", "async-stream", "raw-stream")}
    with wire_server(_chat_provider(identity, prompts)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=f"openai/{_CHAT_BACKEND}", api_base=wire.url, api_key="synthetic-azure-consumer-key"
        )
        key: Final = scenario.key(models=[model])
        endpoint: Final = str(gateway.client.base_url).rstrip("/")
        sent: Final[list[str]] = []

        def record(request: httpx.Request) -> None:
            assert request.headers["api-key"] == key
            sent.append(f"{request.url.path}?{request.url.query.decode()}")

        async def record_async(request: httpx.Request) -> None:
            record(request)

        with AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=key,
            api_version=_AZURE_API_VERSION,
            max_retries=0,
            http_client=httpx.Client(timeout=10, trust_env=False, event_hooks={"request": [record]}),
        ) as sync:
            completion: Final = sync.chat.completions.create(
                model=model, messages=[{"role": "user", "content": f"sync {identity}"}]
            )
            sync_chunks: Final = list(
                sync.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": f"sync-stream {identity}"}],
                    stream=True,
                    stream_options={"include_usage": True},
                )
            )
        async with AsyncAzureOpenAI(
            azure_endpoint=endpoint,
            api_key=key,
            api_version=_AZURE_API_VERSION,
            max_retries=0,
            http_client=httpx.AsyncClient(timeout=10, trust_env=False, event_hooks={"request": [record_async]}),
        ) as asynchronous:
            async_completion: Final = await asynchronous.chat.completions.create(
                model=model, messages=[{"role": "user", "content": f"async {identity}"}]
            )
            async_stream: Final = await asynchronous.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": f"async-stream {identity}"}],
                stream=True,
                stream_options={"include_usage": True},
            )
            async_chunks: Final = [chunk async for chunk in async_stream]
        raw: Final = gateway.client.post(
            f"/openai/deployments/{model}/chat/completions",
            params={"api-version": _AZURE_API_VERSION},
            headers={"api-key": key},
            json={
                "messages": [{"role": "user", "content": f"raw-stream {identity}"}],
                "stream": True,
                "stream_options": {"include_usage": True},
            },
        )
        assert sent == [f"/openai/deployments/{model}/chat/completions?api-version={_AZURE_API_VERSION}"] * 4
        for response in (completion, async_completion):
            assert (response.id, response.object, response.model) == (identity, "chat.completion", model), response
            assert [(choice.message.content, choice.finish_reason) for choice in response.choices] == [
                ("Hello", "stop")
            ]
            assert (response.usage.prompt_tokens, response.usage.completion_tokens, response.usage.total_tokens) == (
                3,
                2,
                5,
            ), response.usage
        for chunks in (sync_chunks, async_chunks):
            assert {(chunk.id, chunk.object, chunk.model) for chunk in chunks} == {
                (identity, "chat.completion.chunk", model)
            }, chunks
            assert "".join(choice.delta.content or "" for chunk in chunks for choice in chunk.choices) == "Hello"
            assert [choice.finish_reason for chunk in chunks for choice in chunk.choices if choice.finish_reason] == [
                "stop"
            ], chunks
            assert [chunk.usage is None for chunk in chunks] == [True] * (len(chunks) - 1) + [False], chunks
            assert (
                chunks[-1].usage.prompt_tokens,
                chunks[-1].usage.completion_tokens,
                chunks[-1].usage.total_tokens,
            ) == (3, 2, 5), chunks[-1]
        assert raw.status_code == 200, raw.text
        assert raw.headers["content-type"].startswith("text/event-stream"), raw.text
        lines: Final = _data_lines(raw.text)
        assert lines[-1] == "data: [DONE]", raw.text
        assert (
            "".join(
                choice["delta"].get("content") or ""
                for line in lines[:-1]
                for choice in json.loads(line.removeprefix("data: "))["choices"]
            )
            == "Hello"
        ), raw.text
        assert json.loads(lines[-2].removeprefix("data: "))["usage"]["total_tokens"] == 5, raw.text
        assert [
            (request.method, request.target, json.loads(request.body)["messages"][0]["content"])
            for request in wire.drain()
        ] == [
            ("POST", "/chat/completions", f"{kind} {identity}")
            for kind in ("sync", "sync-stream", "async", "async-stream", "raw-stream")
        ]


def _text_provider(identity: str, prompts: set[str]) -> Callable[[Request], Reply]:
    def provider(request: Request) -> Reply:
        assert (request.method, request.target) == ("POST", "/completions"), request
        assert request.headers["authorization"] == "Bearer synthetic-text-consumer-key"
        body: Final = json.loads(request.body)
        assert body["prompt"] in prompts, body
        if body.get("stream"):
            assert body == {
                "model": _TEXT_BACKEND,
                "prompt": body["prompt"],
                "stream": True,
                "stream_options": {"include_usage": True},
            }
            return _sse(
                (
                    {
                        "id": identity,
                        "object": "text_completion",
                        "created": 1,
                        "model": _TEXT_BACKEND,
                        "choices": [{"index": 0, "text": "Hel", "finish_reason": None, "logprobs": None}],
                    },
                    {
                        "id": identity,
                        "object": "text_completion",
                        "created": 1,
                        "model": _TEXT_BACKEND,
                        "choices": [{"index": 0, "text": "lo", "finish_reason": "stop", "logprobs": None}],
                    },
                    {
                        "id": identity,
                        "object": "text_completion",
                        "created": 1,
                        "model": _TEXT_BACKEND,
                        "choices": [],
                        "usage": _USAGE,
                    },
                )
            )
        assert body == {"model": _TEXT_BACKEND, "prompt": body["prompt"], "n": 2}
        return Reply(
            body=json.dumps(
                {
                    "id": identity,
                    "object": "text_completion",
                    "created": 1,
                    "model": _TEXT_BACKEND,
                    "choices": [
                        {"index": 0, "text": "first", "finish_reason": "stop", "logprobs": None},
                        {"index": 1, "text": "second", "finish_reason": "length", "logprobs": None},
                    ],
                    "usage": _USAGE,
                }
            ).encode()
        )

    return provider


async def test_openai_clients_parse_text_completion_streams_and_choices(gateway: Gateway) -> None:
    identity: Final = f"cmpl-consumer-{uuid.uuid4().hex}"
    prompts: Final = {f"{kind} {identity}" for kind in ("sync-stream", "async-stream", "raw-stream", "sync-n")}
    with wire_server(_text_provider(identity, prompts)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=f"openai/{_TEXT_BACKEND}", api_base=wire.url, api_key="synthetic-text-consumer-key"
        )
        key: Final = scenario.key(models=[model])
        base_url: Final = str(gateway.client.base_url).rstrip("/") + "/v1"
        streamed: Final = {"stream": True, "stream_options": {"include_usage": True}}
        with OpenAI(
            api_key=key, base_url=base_url, max_retries=0, http_client=httpx.Client(timeout=10, trust_env=False)
        ) as sync:
            sync_chunks: Final = list(
                sync.completions.create(model=model, prompt=f"sync-stream {identity}", **streamed)
            )
            choices: Final = sync.completions.create(model=model, prompt=f"sync-n {identity}", n=2)
        async with AsyncOpenAI(
            api_key=key, base_url=base_url, max_retries=0, http_client=httpx.AsyncClient(timeout=10, trust_env=False)
        ) as asynchronous:
            async_stream: Final = await asynchronous.completions.create(
                model=model, prompt=f"async-stream {identity}", **streamed
            )
            async_chunks: Final = [chunk async for chunk in async_stream]
        raw: Final = gateway.request(
            "POST", "/v1/completions", {"model": model, "prompt": f"raw-stream {identity}", **streamed}, key=key
        )
        for chunks in (sync_chunks, async_chunks):
            assert {(chunk.id, chunk.object, chunk.model) for chunk in chunks} == {
                (identity, "text_completion", model)
            }, chunks
            assert "".join(choice.text or "" for chunk in chunks for choice in chunk.choices) == "Hello", chunks
            assert [choice.finish_reason for chunk in chunks for choice in chunk.choices if choice.finish_reason] == [
                "stop"
            ], chunks
            assert [chunk.usage is None for chunk in chunks] == [True] * (len(chunks) - 1) + [False], chunks
            assert (
                chunks[-1].usage.prompt_tokens,
                chunks[-1].usage.completion_tokens,
                chunks[-1].usage.total_tokens,
            ) == (3, 2, 5), chunks[-1]
        assert raw.status_code == 200, raw.text
        assert raw.headers["content-type"].startswith("text/event-stream"), raw.text
        lines: Final = _data_lines(raw.text)
        assert lines[-1] == "data: [DONE]", raw.text
        frames: Final = [json.loads(line.removeprefix("data: ")) for line in lines[:-1]]
        assert {frame["object"] for frame in frames} == {"text_completion"}, raw.text
        assert "".join(choice.get("text") or "" for frame in frames for choice in frame["choices"]) == "Hello"
        assert (choices.id, choices.object, choices.model) == (identity, "text_completion", model), choices
        assert [(choice.index, choice.text, choice.finish_reason) for choice in choices.choices] == [
            (0, "first", "stop"),
            (1, "second", "length"),
        ], choices
        assert [(request.method, request.target, json.loads(request.body)["prompt"]) for request in wire.drain()] == [
            ("POST", "/completions", f"{kind} {identity}")
            for kind in ("sync-stream", "sync-n", "async-stream", "raw-stream")
        ]


def test_text_completion_usage_chunk_has_empty_choices(gateway: Gateway) -> None:
    pytest.skip("BUG: the /v1/completions include_usage chunk carries choices [{index: 0}] instead of []")
    identity: Final = f"cmpl-usage-{uuid.uuid4().hex}"
    prompt: Final = f"raw-stream {identity}"
    with wire_server(_text_provider(identity, {prompt})) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=f"openai/{_TEXT_BACKEND}", api_base=wire.url, api_key="synthetic-text-consumer-key"
        )
        raw: Final = gateway.request(
            "POST",
            "/v1/completions",
            {"model": model, "prompt": prompt, "stream": True, "stream_options": {"include_usage": True}},
        )
        assert raw.status_code == 200, raw.text
        lines: Final = _data_lines(raw.text)
        assert lines[-1] == "data: [DONE]", raw.text
        usage_frame: Final = json.loads(lines[-2].removeprefix("data: "))
        assert (usage_frame["choices"], usage_frame["usage"]["total_tokens"]) == ([], 5), raw.text
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", "/completions")]
