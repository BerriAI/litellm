import asyncio
from collections.abc import Awaitable, Callable
from typing import Final

import httpx
import openai
from integration._support.client import Gateway, Scenario, object_value
from integration._support.wire import Reply, Request, wire_server
from integration.providers._api_base_trailing_slash_support import (
    KEY,
    VECTOR,
    assert_control_answers,
    assert_one_request,
    body_of,
    marker_of,
    new_marker,
    of_marker,
    provider_peer,
    reply_text,
    result_url,
)
from pydantic import JsonValue

_PER_KIND: Final = 6


def _async_openai(gateway: Gateway) -> openai.AsyncOpenAI:
    return openai.AsyncOpenAI(base_url=str(gateway.client.base_url) + "/v1", api_key=gateway.key, max_retries=0)


def _search_tool(gateway: Gateway, scenario: Scenario, api_base: str) -> str:
    name: Final = f"slash-tool-{new_marker()}"
    created: Final = gateway.post(
        "/search_tools",
        {
            "search_tool": {
                "search_tool_name": name,
                "litellm_params": {"search_provider": "tavily", "api_key": KEY, "api_base": api_base},
            }
        },
    )
    scenario.cleanups.callback(gateway.request, "DELETE", f"/search_tools/{created['search_tool_id']}")
    return name


async def _post_json(gateway: Gateway, path: str, body: dict[str, JsonValue]) -> dict[str, JsonValue]:
    async with httpx.AsyncClient(base_url=str(gateway.client.base_url), timeout=60, trust_env=False) as client:
        response: Final = await client.post(path, json=body, headers={"Authorization": f"Bearer {gateway.key}"})
    return body_of(response)


def test_mixed_burst_on_slash_bases_lands_each_call_once_on_its_single_slash_url(gateway: Gateway) -> None:
    with wire_server(provider_peer) as wire, gateway.scenario() as scenario:
        deepseek: Final = scenario.model(model="deepseek/deepseek-chat", api_base=f"{wire.url}/ds/", api_key=KEY)
        ollama: Final = scenario.model(model="ollama_chat/llama3.1", api_base=f"{wire.url}/ol/", api_key=KEY)
        jina: Final = scenario.model(model="jina_ai/jina-embeddings-v3", api_base=f"{wire.url}/jn/v1/", api_key=KEY)
        tool: Final = _search_tool(gateway, scenario, f"{wire.url}/tv/")

        async def chat(model: str, marker: str) -> bool:
            client: Final = _async_openai(gateway)
            try:
                completion: Final = await client.chat.completions.create(
                    model=model, messages=[{"role": "user", "content": marker}]
                )
            finally:
                await client.close()
            return completion.choices[0].message.content == reply_text(marker)

        async def streamed(model: str, marker: str) -> bool:
            client: Final = _async_openai(gateway)
            try:
                stream: Final = await client.chat.completions.create(
                    model=model, messages=[{"role": "user", "content": marker}], stream=True
                )
                parts: Final = [chunk.choices[0].delta.content or "" async for chunk in stream if chunk.choices]
            finally:
                await client.close()
            return "".join(parts) == reply_text(marker)

        async def embed(marker: str) -> bool:
            payload: Final = await _post_json(gateway, "/v1/embeddings", {"model": jina, "input": marker})
            return payload["data"] == [{"object": "embedding", "index": 0, "embedding": list(VECTOR)}]

        async def search(marker: str) -> bool:
            payload: Final = await _post_json(gateway, f"/v1/search/{tool}", {"query": marker})
            results: Final = payload["results"]
            assert isinstance(results, list)
            return [object_value(result)["url"] for result in results] == [result_url(marker)]

        kinds: Final[tuple[tuple[str, Callable[[str], Awaitable[bool]]], ...]] = (
            ("/ds/chat/completions", lambda marker: chat(deepseek, marker)),
            ("/ds/chat/completions", lambda marker: streamed(deepseek, marker)),
            ("/ol/api/chat", lambda marker: chat(ollama, marker)),
            ("/ol/api/chat", lambda marker: streamed(ollama, marker)),
            ("/jn/v1/embeddings", embed),
            ("/tv/search", search),
        )
        calls: Final = tuple((target, call, new_marker()) for target, call in kinds for _ in range(_PER_KIND))

        async def burst() -> list[bool]:
            return await asyncio.gather(*(call(marker) for _, call, marker in calls))

        assert asyncio.run(burst()) == [True] * len(calls)
        received: Final = wire.drain()
    assert len(received) == len(calls), sorted(request.target for request in received)
    for target, _, marker in calls:
        assert_one_request(of_marker(received, marker), target, marker, keyed=not target.startswith("/ol/"))


def test_upstream_dropping_doomed_connections_mid_burst_fails_only_those_calls(gateway: Gateway) -> None:
    markers: Final = tuple(new_marker() for _ in range(12))
    doomed: Final = frozenset(markers[::2])

    def respond(request: Request) -> Reply:
        return Reply(drop_connection=True) if marker_of(request) in doomed else provider_peer(request)

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model="deepseek/deepseek-chat", api_base=f"{wire.url}/ds/", api_key=KEY)

        async def call(marker: str) -> tuple[int, str]:
            async with httpx.AsyncClient(base_url=str(gateway.client.base_url), timeout=60, trust_env=False) as client:
                response: Final = await client.post(
                    "/v1/chat/completions",
                    json={"model": model, "messages": [{"role": "user", "content": marker}]},
                    headers={"Authorization": f"Bearer {gateway.key}"},
                )
            return response.status_code, response.text

        async def burst() -> list[tuple[int, str]]:
            return await asyncio.gather(*(call(marker) for marker in markers))

        answers: Final = dict(zip(markers, asyncio.run(burst()), strict=True))
        received: Final = wire.drain()
        assert_control_answers(gateway, scenario)
    for marker in doomed:
        _assert_dropped(answers[marker], of_marker(received, marker))
    for marker in frozenset(markers) - doomed:
        assert answers[marker][0] == 200 and reply_text(marker) in answers[marker][1], answers[marker]
        assert_one_request(of_marker(received, marker), "/ds/chat/completions", marker)


def _assert_dropped(answer: tuple[int, str], attempts: tuple[Request, ...]) -> None:
    assert answer[0] >= 500 and "error" in answer[1], answer
    assert attempts and {request.target for request in attempts} == {"/ds/chat/completions"}, attempts
