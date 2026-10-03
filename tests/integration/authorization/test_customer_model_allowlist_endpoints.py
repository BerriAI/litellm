import asyncio
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Final

import anthropic
import httpx
import openai
import pytest
from pydantic import JsonValue, TypeAdapter

from tests.integration._support.client import Gateway, Scenario, object_value
from tests.integration._support.customer_model_allowlist_wire import customer_model_allowlist_reply
from tests.integration._support.wire import Wire, wire_server

_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_RESPONSE_TEXT: Final = "scripted response"


@dataclass(frozen=True, slots=True)
class EndpointScenario:
    key: str
    customer: str
    m1: str
    m2: str
    wire: Wire


def _customer(scenario: Scenario, model: str) -> str:
    identity: Final = f"integration-customer-{uuid.uuid4().hex}"
    scenario.gateway.post("/customer/new", {"user_id": identity, "models": [model]})
    scenario.cleanups.callback(scenario.gateway.post, "/customer/delete", {"user_ids": [identity]})
    return identity


@contextmanager
def _endpoint_scenario(gateway: Gateway, *, embedding: bool = False) -> Iterator[EndpointScenario]:
    with wire_server(customer_model_allowlist_reply) as wire, gateway.scenario() as scenario:
        model: Final = "openai/text-embedding-3-small" if embedding else "openai/gpt-4o-mini"
        m1: Final = scenario.model(model=model, api_base=f"{wire.url}/v1")
        m2: Final = scenario.model(model=model, api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[m1, m2])
        customer: Final = _customer(scenario, m1)
        yield EndpointScenario(key=key, customer=customer, m1=m1, m2=m2, wire=wire)


def _assert_denial(status: int, text: str, model: str) -> None:
    assert status == 403, text
    error: Final = object_value(object_value(_JSON_OBJECT.validate_json(text))["error"])
    message: Final = str(error["message"])
    assert model in message, text
    assert "allowed models for this customer" in message, text
    assert error["param"] == "model", text
    assert error["code"] == "403", text


def _assert_upstream(wire: Wire, marker: str, expected: int) -> None:
    requests: Final = wire.drain()
    matching: Final = tuple(request for request in requests if marker.encode() in request.body)
    assert len(matching) == expected, matching


def _openai_base_url(gateway: Gateway) -> str:
    return f"{str(gateway.client.base_url).rstrip('/')}/v1"


def test_chat_completions_non_stream_is_enforced_with_openai_sync_sdk(gateway: Gateway) -> None:
    with (
        _endpoint_scenario(gateway) as scenario,
        openai.OpenAI(
            api_key=scenario.key,
            base_url=_openai_base_url(gateway),
            http_client=httpx.Client(timeout=15, trust_env=False),
        ) as client,
    ):
        denied_marker: Final = uuid.uuid4().hex
        with pytest.raises(openai.PermissionDeniedError) as denied:
            client.chat.completions.create(
                model=scenario.m2,
                messages=[{"role": "user", "content": denied_marker}],
                user=scenario.customer,
                max_tokens=4,
            )
        _assert_denial(denied.value.response.status_code, denied.value.response.text, scenario.m2)
        _assert_upstream(scenario.wire, denied_marker, 0)

        allowed_marker: Final = uuid.uuid4().hex
        allowed = client.chat.completions.create(
            model=scenario.m1,
            messages=[{"role": "user", "content": allowed_marker}],
            user=scenario.customer,
            max_tokens=4,
        )
        assert allowed.choices[0].message.content == _RESPONSE_TEXT, allowed.model_dump_json()
        _assert_upstream(scenario.wire, allowed_marker, 1)


def test_chat_completions_non_stream_is_enforced_with_openai_async_sdk(gateway: Gateway) -> None:
    with _endpoint_scenario(gateway) as scenario:

        async def exercise() -> None:
            async with openai.AsyncOpenAI(
                api_key=scenario.key,
                base_url=_openai_base_url(gateway),
                http_client=httpx.AsyncClient(timeout=15, trust_env=False),
            ) as client:
                denied_marker: Final = uuid.uuid4().hex
                with pytest.raises(openai.PermissionDeniedError) as denied:
                    await client.chat.completions.create(
                        model=scenario.m2,
                        messages=[{"role": "user", "content": denied_marker}],
                        user=scenario.customer,
                        max_tokens=4,
                    )
                _assert_denial(denied.value.response.status_code, denied.value.response.text, scenario.m2)
                _assert_upstream(scenario.wire, denied_marker, 0)

                allowed_marker: Final = uuid.uuid4().hex
                allowed = await client.chat.completions.create(
                    model=scenario.m1,
                    messages=[{"role": "user", "content": allowed_marker}],
                    user=scenario.customer,
                    max_tokens=4,
                )
                assert allowed.choices[0].message.content == _RESPONSE_TEXT, allowed.model_dump_json()
                _assert_upstream(scenario.wire, allowed_marker, 1)

        asyncio.run(exercise())


def test_chat_completions_stream_is_enforced_with_openai_sync_and_async_sdks(gateway: Gateway) -> None:
    with _endpoint_scenario(gateway) as scenario:
        with openai.OpenAI(
            api_key=scenario.key,
            base_url=_openai_base_url(gateway),
            http_client=httpx.Client(timeout=15, trust_env=False),
        ) as client:
            denied_marker: Final = uuid.uuid4().hex
            with pytest.raises(openai.PermissionDeniedError) as denied:
                tuple(
                    client.chat.completions.create(
                        model=scenario.m2,
                        messages=[{"role": "user", "content": denied_marker}],
                        user=scenario.customer,
                        stream=True,
                    )
                )
            _assert_denial(denied.value.response.status_code, denied.value.response.text, scenario.m2)
            _assert_upstream(scenario.wire, denied_marker, 0)

            allowed_marker: Final = uuid.uuid4().hex
            chunks: Final = tuple(
                client.chat.completions.create(
                    model=scenario.m1,
                    messages=[{"role": "user", "content": allowed_marker}],
                    user=scenario.customer,
                    stream=True,
                )
            )
            assert "".join(chunk.choices[0].delta.content or "" for chunk in chunks) == _RESPONSE_TEXT, str(chunks)
            _assert_upstream(scenario.wire, allowed_marker, 1)

        async def exercise() -> None:
            async with openai.AsyncOpenAI(
                api_key=scenario.key,
                base_url=_openai_base_url(gateway),
                http_client=httpx.AsyncClient(timeout=15, trust_env=False),
            ) as client:
                denied_marker: Final = uuid.uuid4().hex

                async def create_and_drain_denied_stream() -> None:
                    stream: Final = await client.chat.completions.create(
                        model=scenario.m2,
                        messages=[{"role": "user", "content": denied_marker}],
                        user=scenario.customer,
                        stream=True,
                    )
                    tuple([chunk async for chunk in stream])

                with pytest.raises(openai.PermissionDeniedError) as denied:
                    await create_and_drain_denied_stream()
                _assert_denial(denied.value.response.status_code, denied.value.response.text, scenario.m2)
                _assert_upstream(scenario.wire, denied_marker, 0)

                allowed_marker: Final = uuid.uuid4().hex
                stream = await client.chat.completions.create(
                    model=scenario.m1,
                    messages=[{"role": "user", "content": allowed_marker}],
                    user=scenario.customer,
                    stream=True,
                )
                chunks: Final = tuple([chunk async for chunk in stream])
                assert "".join(chunk.choices[0].delta.content or "" for chunk in chunks) == _RESPONSE_TEXT, str(chunks)
                _assert_upstream(scenario.wire, allowed_marker, 1)

        asyncio.run(exercise())


def test_chat_completions_raw_httpx_is_enforced(gateway: Gateway) -> None:
    with _endpoint_scenario(gateway) as scenario:
        denied_marker: Final = uuid.uuid4().hex
        denied: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": scenario.m2,
                "messages": [{"role": "user", "content": denied_marker}],
                "user": scenario.customer,
            },
            key=scenario.key,
        )
        _assert_denial(denied.status_code, denied.text, scenario.m2)
        _assert_upstream(scenario.wire, denied_marker, 0)

        allowed_marker: Final = uuid.uuid4().hex
        allowed: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": scenario.m1,
                "messages": [{"role": "user", "content": allowed_marker}],
                "user": scenario.customer,
            },
            key=scenario.key,
        )
        assert allowed.status_code == 200, allowed.text
        _assert_upstream(scenario.wire, allowed_marker, 1)


def test_messages_non_stream_is_enforced_with_anthropic_sync_and_async_sdks(gateway: Gateway) -> None:
    with _endpoint_scenario(gateway) as scenario:
        with anthropic.Anthropic(
            api_key=scenario.key,
            base_url=str(gateway.client.base_url),
            http_client=httpx.Client(timeout=15, trust_env=False),
        ) as client:
            denied_marker: Final = uuid.uuid4().hex
            with pytest.raises(anthropic.PermissionDeniedError) as denied:
                client.messages.create(
                    model=scenario.m2,
                    max_tokens=8,
                    messages=[{"role": "user", "content": denied_marker}],
                    metadata={"user_id": scenario.customer},
                )
            _assert_denial(denied.value.response.status_code, denied.value.response.text, scenario.m2)
            _assert_upstream(scenario.wire, denied_marker, 0)

            allowed_marker: Final = uuid.uuid4().hex
            allowed = client.messages.create(
                model=scenario.m1,
                max_tokens=8,
                messages=[{"role": "user", "content": allowed_marker}],
                metadata={"user_id": scenario.customer},
            )
            assert allowed.content[0].text == _RESPONSE_TEXT, allowed.model_dump_json()
            _assert_upstream(scenario.wire, allowed_marker, 1)

        async def exercise() -> None:
            async with anthropic.AsyncAnthropic(
                api_key=scenario.key,
                base_url=str(gateway.client.base_url),
                http_client=httpx.AsyncClient(timeout=15, trust_env=False),
            ) as client:
                denied_marker: Final = uuid.uuid4().hex
                with pytest.raises(anthropic.PermissionDeniedError) as denied:
                    await client.messages.create(
                        model=scenario.m2,
                        max_tokens=8,
                        messages=[{"role": "user", "content": denied_marker}],
                        metadata={"user_id": scenario.customer},
                    )
                _assert_denial(denied.value.response.status_code, denied.value.response.text, scenario.m2)
                _assert_upstream(scenario.wire, denied_marker, 0)

                allowed_marker: Final = uuid.uuid4().hex
                allowed = await client.messages.create(
                    model=scenario.m1,
                    max_tokens=8,
                    messages=[{"role": "user", "content": allowed_marker}],
                    metadata={"user_id": scenario.customer},
                )
                assert allowed.content[0].text == _RESPONSE_TEXT, allowed.model_dump_json()
                _assert_upstream(scenario.wire, allowed_marker, 1)

        asyncio.run(exercise())


def test_messages_stream_is_enforced_with_anthropic_sync_and_async_sdks(gateway: Gateway) -> None:
    with _endpoint_scenario(gateway) as scenario:
        with anthropic.Anthropic(
            api_key=scenario.key,
            base_url=str(gateway.client.base_url),
            http_client=httpx.Client(timeout=15, trust_env=False),
        ) as client:
            denied_marker: Final = uuid.uuid4().hex
            with pytest.raises(anthropic.PermissionDeniedError) as denied:
                with client.messages.stream(
                    model=scenario.m2,
                    max_tokens=8,
                    messages=[{"role": "user", "content": denied_marker}],
                    metadata={"user_id": scenario.customer},
                ) as stream:
                    tuple(stream.text_stream)
            _assert_denial(denied.value.response.status_code, denied.value.response.text, scenario.m2)
            _assert_upstream(scenario.wire, denied_marker, 0)

            allowed_marker: Final = uuid.uuid4().hex
            with client.messages.stream(
                model=scenario.m1,
                max_tokens=8,
                messages=[{"role": "user", "content": allowed_marker}],
                metadata={"user_id": scenario.customer},
            ) as stream:
                text: Final = "".join(stream.text_stream)
            assert text == _RESPONSE_TEXT, text
            _assert_upstream(scenario.wire, allowed_marker, 1)

        async def exercise() -> None:
            async with anthropic.AsyncAnthropic(
                api_key=scenario.key,
                base_url=str(gateway.client.base_url),
                http_client=httpx.AsyncClient(timeout=15, trust_env=False),
            ) as client:
                denied_marker: Final = uuid.uuid4().hex
                with pytest.raises(anthropic.PermissionDeniedError) as denied:
                    async with client.messages.stream(
                        model=scenario.m2,
                        max_tokens=8,
                        messages=[{"role": "user", "content": denied_marker}],
                        metadata={"user_id": scenario.customer},
                    ) as stream:
                        tuple([text async for text in stream.text_stream])
                _assert_denial(denied.value.response.status_code, denied.value.response.text, scenario.m2)
                _assert_upstream(scenario.wire, denied_marker, 0)

                allowed_marker: Final = uuid.uuid4().hex
                async with client.messages.stream(
                    model=scenario.m1,
                    max_tokens=8,
                    messages=[{"role": "user", "content": allowed_marker}],
                    metadata={"user_id": scenario.customer},
                ) as stream:
                    text: Final = "".join([chunk async for chunk in stream.text_stream])
                assert text == _RESPONSE_TEXT, text
                _assert_upstream(scenario.wire, allowed_marker, 1)

        asyncio.run(exercise())


def test_responses_non_stream_is_enforced_with_openai_sync_and_async_sdks(gateway: Gateway) -> None:
    with _endpoint_scenario(gateway) as scenario:
        with openai.OpenAI(
            api_key=scenario.key,
            base_url=_openai_base_url(gateway),
            http_client=httpx.Client(timeout=15, trust_env=False),
        ) as client:
            denied_marker: Final = uuid.uuid4().hex
            with pytest.raises(openai.PermissionDeniedError) as denied:
                client.responses.create(
                    model=scenario.m2,
                    input=denied_marker,
                    user=scenario.customer,
                    max_output_tokens=4,
                )
            _assert_denial(denied.value.response.status_code, denied.value.response.text, scenario.m2)
            _assert_upstream(scenario.wire, denied_marker, 0)

            allowed_marker: Final = uuid.uuid4().hex
            allowed = client.responses.create(
                model=scenario.m1,
                input=allowed_marker,
                user=scenario.customer,
                max_output_tokens=4,
            )
            assert allowed.output_text == _RESPONSE_TEXT, allowed.model_dump_json()
            _assert_upstream(scenario.wire, allowed_marker, 1)

        async def exercise() -> None:
            async with openai.AsyncOpenAI(
                api_key=scenario.key,
                base_url=_openai_base_url(gateway),
                http_client=httpx.AsyncClient(timeout=15, trust_env=False),
            ) as client:
                denied_marker: Final = uuid.uuid4().hex
                with pytest.raises(openai.PermissionDeniedError) as denied:
                    await client.responses.create(
                        model=scenario.m2,
                        input=denied_marker,
                        user=scenario.customer,
                        max_output_tokens=4,
                    )
                _assert_denial(denied.value.response.status_code, denied.value.response.text, scenario.m2)
                _assert_upstream(scenario.wire, denied_marker, 0)

                allowed_marker: Final = uuid.uuid4().hex
                allowed = await client.responses.create(
                    model=scenario.m1,
                    input=allowed_marker,
                    user=scenario.customer,
                    max_output_tokens=4,
                )
                assert allowed.output_text == _RESPONSE_TEXT, allowed.model_dump_json()
                _assert_upstream(scenario.wire, allowed_marker, 1)

        asyncio.run(exercise())


def test_responses_stream_is_enforced_with_openai_sync_and_async_sdks(gateway: Gateway) -> None:
    with _endpoint_scenario(gateway) as scenario:
        with openai.OpenAI(
            api_key=scenario.key,
            base_url=_openai_base_url(gateway),
            http_client=httpx.Client(timeout=15, trust_env=False),
        ) as client:
            denied_marker: Final = uuid.uuid4().hex
            with pytest.raises(openai.PermissionDeniedError) as denied:
                tuple(
                    client.responses.create(
                        model=scenario.m2,
                        input=denied_marker,
                        user=scenario.customer,
                        stream=True,
                    )
                )
            _assert_denial(denied.value.response.status_code, denied.value.response.text, scenario.m2)
            _assert_upstream(scenario.wire, denied_marker, 0)

            allowed_marker: Final = uuid.uuid4().hex
            events: Final = tuple(
                client.responses.create(
                    model=scenario.m1,
                    input=allowed_marker,
                    user=scenario.customer,
                    stream=True,
                )
            )
            assert any(event.type == "response.output_text.delta" for event in events), str(events)
            _assert_upstream(scenario.wire, allowed_marker, 1)

        async def exercise() -> None:
            async with openai.AsyncOpenAI(
                api_key=scenario.key,
                base_url=_openai_base_url(gateway),
                http_client=httpx.AsyncClient(timeout=15, trust_env=False),
            ) as client:
                denied_marker: Final = uuid.uuid4().hex

                async def create_and_drain_denied_stream() -> None:
                    stream: Final = await client.responses.create(
                        model=scenario.m2,
                        input=denied_marker,
                        user=scenario.customer,
                        stream=True,
                    )
                    tuple([event async for event in stream])

                with pytest.raises(openai.PermissionDeniedError) as denied:
                    await create_and_drain_denied_stream()
                _assert_denial(denied.value.response.status_code, denied.value.response.text, scenario.m2)
                _assert_upstream(scenario.wire, denied_marker, 0)

                allowed_marker: Final = uuid.uuid4().hex
                stream = await client.responses.create(
                    model=scenario.m1,
                    input=allowed_marker,
                    user=scenario.customer,
                    stream=True,
                )
                events: Final = tuple([event async for event in stream])
                assert any(event.type == "response.output_text.delta" for event in events), str(events)
                _assert_upstream(scenario.wire, allowed_marker, 1)

        asyncio.run(exercise())


def test_embeddings_enforcement_is_checked_with_raw_httpx(gateway: Gateway) -> None:
    with _endpoint_scenario(gateway, embedding=True) as scenario:
        denied_marker: Final = uuid.uuid4().hex
        denied: Final = gateway.request(
            "POST",
            "/v1/embeddings",
            {"model": scenario.m2, "input": denied_marker, "user": scenario.customer},
            key=scenario.key,
        )
        _assert_denial(denied.status_code, denied.text, scenario.m2)
        _assert_upstream(scenario.wire, denied_marker, 0)

        allowed_marker: Final = uuid.uuid4().hex
        allowed: Final = gateway.request(
            "POST",
            "/v1/embeddings",
            {"model": scenario.m1, "input": allowed_marker, "user": scenario.customer},
            key=scenario.key,
        )
        assert allowed.status_code == 200, allowed.text
        assert allowed.json()["data"][0]["embedding"] == [0.125, 0.25], allowed.text
        _assert_upstream(scenario.wire, allowed_marker, 1)
