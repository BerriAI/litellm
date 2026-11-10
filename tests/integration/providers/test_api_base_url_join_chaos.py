import asyncio
from collections.abc import Awaitable, Callable
from itertools import chain, repeat
from typing import Final, TypeAlias

import anthropic
import httpx
import openai
from integration._support.client import Gateway, Scenario, eventually
from integration._support.database import read_rows
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
    reply_text,
)
from integration.providers._api_base_url_join_support import join_peer
from pydantic import JsonValue

_PER_KIND: Final = 3
_HAIKU: Final = "anthropic.claude-haiku-5-5"
_GENERATE: Final = "/gm/v1beta/models/gemini-3.8-flash:generateContent"
_STREAM_GENERATE: Final = "/gm/v1beta/models/gemini-3.8-flash:streamGenerateContent?alt=sse"

Answer: TypeAlias = tuple[str | None, str | None]
Call: TypeAlias = Callable[[str], Awaitable[Answer]]


def _deploy(scenario: Scenario, model: str, api_base: str) -> str:
    if model.startswith("bedrock/"):
        return scenario.model(model=model, api_base=api_base, api_key=KEY, aws_region_name="us-east-1")
    return scenario.model(model=model, api_base=api_base, api_key=KEY)


def _spend_api_bases(identities: tuple[str, ...]) -> dict[str, JsonValue]:
    rows: Final = eventually(
        lambda: read_rows(
            "SELECT request_id, api_base FROM \"LiteLLM_SpendLogs\" WHERE request_id = ANY(string_to_array(%s, ','))",
            (",".join(identities),),
        ),
        lambda found: len(found) == len(identities),
        seconds=70,
    )
    return {str(row["request_id"]): row["api_base"] for row in rows}


def test_mixed_burst_on_slash_bases_lands_each_call_once_on_its_single_slash_url(gateway: Gateway) -> None:
    with wire_server(join_peer) as wire, gateway.scenario() as scenario:
        claude: Final = _deploy(scenario, "anthropic/claude-haiku-5-5", f"{wire.url}/an/")
        gemini: Final = _deploy(scenario, "gemini/gemini-3.8-flash", f"{wire.url}/gm/v1beta/")
        mistral: Final = _deploy(scenario, "mistral/mistral-small-latest", f"{wire.url}/mi/v1/")
        converse: Final = _deploy(scenario, f"bedrock/converse/{_HAIKU}", f"{wire.url}/bc/")
        databricks: Final = _deploy(
            scenario, "databricks/databricks-meta-llama-3-3-70b-instruct", f"{wire.url}/db/serving-endpoints/"
        )
        embed_model: Final = _deploy(scenario, "mistral/mistral-embed", f"{wire.url}/mi/v1/")

        def openai_client() -> openai.AsyncOpenAI:
            return openai.AsyncOpenAI(base_url=str(gateway.client.base_url) + "/v1", api_key=gateway.key, max_retries=0)

        def chat(model: str) -> Call:
            async def call(marker: str) -> Answer:
                async with openai_client() as client:
                    completion: Final = await client.chat.completions.create(
                        model=model, messages=[{"role": "user", "content": marker}]
                    )
                return completion.choices[0].message.content, completion.id

            return call

        def streamed(model: str) -> Call:
            async def call(marker: str) -> Answer:
                async with openai_client() as client:
                    stream: Final = await client.chat.completions.create(
                        model=model, messages=[{"role": "user", "content": marker}], stream=True
                    )
                    parts: Final = [chunk.choices[0].delta.content or "" async for chunk in stream if chunk.choices]
                return "".join(parts), None

            return call

        async def messages(marker: str) -> Answer:
            async with anthropic.AsyncAnthropic(
                base_url=str(gateway.client.base_url), api_key=gateway.key, max_retries=0
            ) as client:
                message: Final = await client.messages.create(
                    model=claude, max_tokens=16, messages=[{"role": "user", "content": marker}]
                )
            return "".join(block.text for block in message.content if block.type == "text"), None

        async def embed(marker: str) -> Answer:
            async with httpx.AsyncClient(base_url=str(gateway.client.base_url), timeout=60, trust_env=False) as client:
                response: Final = await client.post(
                    "/v1/embeddings",
                    json={"model": embed_model, "input": marker},
                    headers={"Authorization": f"Bearer {gateway.key}"},
                )
            data: Final = body_of(response)["data"]
            vector_matches: Final = data == [{"object": "embedding", "index": 0, "embedding": list(VECTOR)}]
            return (reply_text(marker) if vector_matches else str(data)), None

        kinds: Final[tuple[tuple[str, Call], ...]] = (
            ("/an/v1/messages", chat(claude)),
            ("/an/v1/messages", streamed(claude)),
            ("/an/v1/messages", messages),
            (_GENERATE, chat(gemini)),
            (_STREAM_GENERATE, streamed(gemini)),
            ("/mi/v1/chat/completions", chat(mistral)),
            ("/mi/v1/chat/completions", streamed(mistral)),
            (f"/bc/model/{_HAIKU}/converse", chat(converse)),
            (f"/bc/model/{_HAIKU}/converse-stream", streamed(converse)),
            ("/db/serving-endpoints/chat/completions", chat(databricks)),
            ("/mi/v1/embeddings", embed),
        )
        repeated_kinds: Final = tuple(chain.from_iterable(repeat(kind, _PER_KIND) for kind in kinds))
        calls: Final = tuple((target, call, new_marker()) for target, call in repeated_kinds)

        async def burst() -> list[Answer]:
            return await asyncio.gather(*(call(marker) for _, call, marker in calls))

        answers: Final = asyncio.run(burst())
        assert [text for text, _ in answers] == [reply_text(marker) for *_, marker in calls]
        received: Final = wire.drain()
        assert len(received) == len(calls), sorted(request.target for request in received)
        for target, _, marker in calls:
            assert_one_request(of_marker(received, marker), target, marker)
        logged: Final = {
            identity: f"{wire.url}{target}"
            for (target, _, _), (_, identity) in zip(calls, answers, strict=True)
            if identity is not None
        }
        assert _spend_api_bases(tuple(logged)) == logged


def test_upstream_dropping_doomed_connections_mid_burst_fails_only_those_calls(gateway: Gateway) -> None:
    markers: Final = tuple(new_marker() for _ in range(16))
    doomed: Final = frozenset(markers[::2])

    def respond(request: Request) -> Reply:
        return Reply(drop_connection=True) if marker_of(request) in doomed else join_peer(request)

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        claude: Final = _deploy(scenario, "anthropic/claude-haiku-5-5", f"{wire.url}/an/")
        gemini: Final = _deploy(scenario, "gemini/gemini-3.8-flash", f"{wire.url}/gm/v1beta/")
        routed: Final = {
            marker: (claude, "/an/v1/messages") if index % 4 < 2 else (gemini, _GENERATE)
            for index, marker in enumerate(markers)
        }

        async def call(marker: str) -> tuple[int, str]:
            async with httpx.AsyncClient(base_url=str(gateway.client.base_url), timeout=60, trust_env=False) as client:
                response: Final = await client.post(
                    "/v1/chat/completions",
                    json={"model": routed[marker][0], "messages": [{"role": "user", "content": marker}]},
                    headers={"Authorization": f"Bearer {gateway.key}"},
                )
            return response.status_code, response.text

        async def burst() -> list[tuple[int, str]]:
            return await asyncio.gather(*(call(marker) for marker in markers))

        answers: Final = dict(zip(markers, asyncio.run(burst()), strict=True))
        received: Final = wire.drain()
        assert_control_answers(gateway, scenario)
    for marker in doomed:
        assert answers[marker][0] >= 500 and "error" in answers[marker][1], answers[marker]
        assert {request.target for request in of_marker(received, marker)} == {routed[marker][1]}, marker
    for marker in frozenset(markers) - doomed:
        assert answers[marker][0] == 200 and reply_text(marker) in answers[marker][1], answers[marker]
        assert_one_request(of_marker(received, marker), routed[marker][1], marker)
