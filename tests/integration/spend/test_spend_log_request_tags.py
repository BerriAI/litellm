import asyncio
import http.client
import json
import uuid
from hashlib import sha256
from pathlib import Path
from typing import Final

import anthropic
import httpx
from collections.abc import Mapping, Sequence
from itertools import chain
import openai
import pytest
import yaml
from integration._support.client import Gateway, eventually
from integration._support.database import read_rows
from integration._support.process import owned_proxy, owned_proxy_process
from integration._support.wire import Reply, Request, wire_server
from integration.spend._request_tag_helpers import (
    GEMINI_MODEL,
    OPENAI_MODEL,
    provider_env,
    provider_reply,
    tags_by_id,
    tags_by_key,
    tags_of,
    write_config,
)

MODEL: Final = "claude-sonnet-4-5-20250929"
SENT_HEADERS: Final = {"user-agent": "claude-cli/2.0.0", "x-tenant-id": "tenant-a"}
EXPECTED_TAGS: Final = ["User-Agent: claude-cli", "User-Agent: claude-cli/2.0.0", "x-tenant-id: tenant-a"]


@pytest.mark.parametrize(
    "route", [pytest.param("/anthropic/v1/messages", id="passthrough"), pytest.param("/v1/messages", id="unified")]
)
def test_header_derived_spend_tags_are_recorded_on_anthropic_messages_routes(
    gateway: Gateway, tmp_path: Path, route: str
) -> None:
    with wire_server(provider_reply) as wire:
        config: Final = write_config(tmp_path, {"litellm_settings": {"extra_spend_tag_headers": ["x-tenant-id"]}})
        with (
            owned_proxy(gateway, tmp_path, provider_env(wire.url), config=config, workers=2) as candidate,
            candidate.scenario() as scenario,
        ):
            model: Final = scenario.model(
                model=f"anthropic/{MODEL}", api_base=wire.url, api_key="synthetic-anthropic-key"
            )
            key: Final = scenario.key()
            response: Final = candidate.request(
                "POST",
                route,
                {
                    "model": MODEL if route == "/anthropic/v1/messages" else model,
                    "max_tokens": 16,
                    "messages": [{"role": "user", "content": f"tag me {uuid.uuid4().hex}"}],
                },
                key=key,
                headers={**SENT_HEADERS, "anthropic-version": "2023-06-01"},
            )
            assert response.status_code == 200, response.text
            assert len(wire.drain()) == 1
            assert eventually(lambda: tags_by_key(key), lambda tags: len(tags) == 1, seconds=70) == [EXPECTED_TAGS]


def _base_url(candidate: Gateway) -> str:
    return str(candidate.client.base_url).rstrip("/")


UA_TAG: Final = "User-Agent: claude-cli/2.0.0"
UA_FAMILY_TAG: Final = "User-Agent: claude-cli"
TENANT_TAG: Final = "x-tenant-id: tenant-a"


def test_pass_through_anthropic_stream_records_header_tags(gateway: Gateway, tmp_path: Path) -> None:
    with wire_server(provider_reply) as wire:
        config: Final = write_config(tmp_path, {"litellm_settings": {"extra_spend_tag_headers": ["x-tenant-id"]}})
        with (
            owned_proxy(gateway, tmp_path, provider_env(wire.url), config=config, workers=2) as candidate,
            candidate.scenario() as scenario,
        ):
            key: Final = scenario.key()
            response: Final = candidate.request(
                "POST",
                "/anthropic/v1/messages",
                {
                    "model": MODEL,
                    "max_tokens": 16,
                    "messages": [{"role": "user", "content": f"tag me {uuid.uuid4().hex}"}],
                    "stream": True,
                },
                key=key,
                headers={**SENT_HEADERS, "anthropic-version": "2023-06-01"},
            )
            assert response.status_code == 200, response.text
            assert '"type":"message_start"' in response.text.replace(" ", ""), response.text
            assert len(wire.drain()) == 1
            assert eventually(lambda: tags_by_key(key), lambda tags: len(tags) == 1, seconds=70) == [EXPECTED_TAGS]


def test_pass_through_anthropic_sdk_records_header_tags(gateway: Gateway, tmp_path: Path) -> None:
    with wire_server(provider_reply) as wire:
        config: Final = write_config(tmp_path, {"litellm_settings": {"extra_spend_tag_headers": ["x-tenant-id"]}})
        with (
            owned_proxy(gateway, tmp_path, provider_env(wire.url), config=config, workers=2) as candidate,
            candidate.scenario() as scenario,
        ):
            key: Final = scenario.key()
            client: Final = anthropic.Anthropic(
                base_url=f"{_base_url(candidate)}/anthropic", auth_token=key, default_headers=SENT_HEADERS
            )
            message: Final = client.messages.create(
                model=MODEL, max_tokens=16, messages=[{"role": "user", "content": f"tag me {uuid.uuid4().hex}"}]
            )
            assert message.id.startswith("msg_")
            assert len(wire.drain()) == 1
            assert eventually(lambda: tags_by_key(key), lambda tags: len(tags) == 1, seconds=70) == [
                ["User-Agent: Anthropic", f"User-Agent: Anthropic/Python {anthropic.__version__}", TENANT_TAG]
            ]


def test_pass_through_anthropic_sdk_stream_records_header_tags(gateway: Gateway, tmp_path: Path) -> None:
    with wire_server(provider_reply) as wire:
        config: Final = write_config(tmp_path, {"litellm_settings": {"extra_spend_tag_headers": ["x-tenant-id"]}})
        with (
            owned_proxy(gateway, tmp_path, provider_env(wire.url), config=config, workers=2) as candidate,
            candidate.scenario() as scenario,
        ):
            key: Final = scenario.key()
            client: Final = anthropic.Anthropic(
                base_url=f"{_base_url(candidate)}/anthropic", auth_token=key, default_headers=SENT_HEADERS
            )
            with client.messages.stream(
                model=MODEL, max_tokens=16, messages=[{"role": "user", "content": f"tag me {uuid.uuid4().hex}"}]
            ) as stream:
                message: Final = stream.get_final_message()
            assert message.id.startswith("msg_")
            assert len(wire.drain()) == 1
            assert eventually(lambda: tags_by_key(key), lambda tags: len(tags) == 1, seconds=70) == [
                ["User-Agent: Anthropic", f"User-Agent: Anthropic/Python {anthropic.__version__}", TENANT_TAG]
            ]


@pytest.mark.parametrize("stream", [pytest.param(False, id="sync"), pytest.param(True, id="stream")])
def test_pass_through_openai_chat_records_header_tags(gateway: Gateway, tmp_path: Path, stream: bool) -> None:
    with wire_server(provider_reply) as wire:
        config: Final = write_config(tmp_path, {"litellm_settings": {"extra_spend_tag_headers": ["x-tenant-id"]}})
        with (
            owned_proxy(gateway, tmp_path, provider_env(wire.url), config=config, workers=2) as candidate,
            candidate.scenario() as scenario,
        ):
            key: Final = scenario.key()
            response: Final = candidate.request(
                "POST",
                "/openai/v1/chat/completions",
                {
                    "model": OPENAI_MODEL,
                    "messages": [{"role": "user", "content": f"tag me {uuid.uuid4().hex}"}],
                    **({"stream": True} if stream else {}),
                },
                key=key,
                headers=SENT_HEADERS,
            )
            assert response.status_code == 200, response.text
            assert "chatcmpl_" in response.text, response.text
            assert len(wire.drain()) == 1
            assert eventually(lambda: tags_by_key(key), lambda tags: len(tags) == 1, seconds=70) == [EXPECTED_TAGS]


@pytest.mark.parametrize("stream", [pytest.param(False, id="sync"), pytest.param(True, id="stream")])
def test_pass_through_openai_sdk_records_header_tags(gateway: Gateway, tmp_path: Path, stream: bool) -> None:
    with wire_server(provider_reply) as wire:
        config: Final = write_config(tmp_path, {"litellm_settings": {"extra_spend_tag_headers": ["x-tenant-id"]}})
        with (
            owned_proxy(gateway, tmp_path, provider_env(wire.url), config=config, workers=2) as candidate,
            candidate.scenario() as scenario,
        ):
            key: Final = scenario.key()
            client: Final = openai.OpenAI(
                api_key=key, base_url=f"{_base_url(candidate)}/openai/v1", default_headers=SENT_HEADERS
            )
            completion: Final = client.chat.completions.create(
                model=OPENAI_MODEL, messages=[{"role": "user", "content": f"tag me {uuid.uuid4().hex}"}], stream=stream
            )
            if stream:
                request_id: Final = next(chunk.id for chunk in completion)
                for _ in completion:
                    pass
            else:
                request_id = completion.id
            assert request_id.startswith("chatcmpl_")
            assert len(wire.drain()) == 1
            assert eventually(lambda: tags_by_key(key), lambda tags: len(tags) == 1, seconds=70) == [
                ["User-Agent: OpenAI", f"User-Agent: OpenAI/Python {openai.__version__}", TENANT_TAG]
            ]


def test_pass_through_gemini_records_header_tags(gateway: Gateway, tmp_path: Path) -> None:
    with wire_server(provider_reply) as wire:
        config: Final = write_config(tmp_path, {"litellm_settings": {"extra_spend_tag_headers": ["x-tenant-id"]}})
        with (
            owned_proxy(gateway, tmp_path, provider_env(wire.url), config=config, workers=2) as candidate,
            candidate.scenario() as scenario,
        ):
            key: Final = scenario.key()
            response: Final = candidate.request(
                "POST",
                f"/gemini/v1beta/models/{GEMINI_MODEL}:generateContent",
                {"contents": [{"parts": [{"text": f"tag me {uuid.uuid4().hex}"}]}]},
                key=key,
                headers={**SENT_HEADERS, "x-goog-api-key": key},
            )
            assert response.status_code == 200, response.text
            assert len(wire.drain()) == 1
            assert eventually(lambda: tags_by_key(key), lambda tags: len(tags) == 1, seconds=70) == [EXPECTED_TAGS]


def test_custom_pass_through_endpoint_records_header_tags(gateway: Gateway, tmp_path: Path) -> None:
    with wire_server(provider_reply) as wire:
        config: Final = write_config(
            tmp_path,
            {
                "litellm_settings": {"extra_spend_tag_headers": ["x-tenant-id"]},
                "general_settings": {
                    "pass_through_endpoints": [
                        {
                            "path": "/custom-anthropic",
                            "target": f"{wire.url}/v1/messages",
                            "auth": True,
                            "headers": {
                                "x-api-key": "synthetic-anthropic-key",
                                "anthropic-version": "2023-06-01",
                            },
                        }
                    ]
                },
            },
        )
        with (
            owned_proxy(gateway, tmp_path, provider_env(wire.url), config=config, workers=2) as candidate,
            candidate.scenario() as scenario,
        ):
            key: Final = scenario.key(allowed_passthrough_routes=["/custom-anthropic"])
            response: Final = candidate.request(
                "POST",
                "/custom-anthropic",
                {
                    "model": MODEL,
                    "max_tokens": 16,
                    "messages": [{"role": "user", "content": f"tag me {uuid.uuid4().hex}"}],
                },
                key=key,
                headers=SENT_HEADERS,
            )
            assert response.status_code == 200, response.text
            assert len(wire.drain()) == 1
            assert eventually(lambda: tags_by_key(key), lambda tags: len(tags) == 1, seconds=70) == [EXPECTED_TAGS]


def test_pass_through_openai_async_sdk_stream_records_header_tags(gateway: Gateway, tmp_path: Path) -> None:
    with wire_server(provider_reply) as wire:
        config: Final = write_config(tmp_path, {"litellm_settings": {"extra_spend_tag_headers": ["x-tenant-id"]}})
        with (
            owned_proxy(gateway, tmp_path, provider_env(wire.url), config=config, workers=2) as candidate,
            candidate.scenario() as scenario,
        ):
            key: Final = scenario.key()
            client: Final = openai.AsyncOpenAI(
                api_key=key, base_url=f"{_base_url(candidate)}/openai/v1", default_headers=SENT_HEADERS
            )

            async def call() -> str:
                completion: Final = await client.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=[{"role": "user", "content": f"tag me {uuid.uuid4().hex}"}],
                    stream=True,
                )
                first: Final = await completion.__anext__()
                async for _ in completion:
                    pass
                return first.id

            request_id: Final = asyncio.run(call())
            assert request_id.startswith("chatcmpl_")
            assert len(wire.drain()) == 1
            assert eventually(lambda: tags_by_key(key), lambda tags: len(tags) == 1, seconds=70) == [
                ["User-Agent: AsyncOpenAI", f"User-Agent: AsyncOpenAI/Python {openai.__version__}", TENANT_TAG]
            ]


def test_unified_chat_completions_records_header_tags(gateway: Gateway, tmp_path: Path) -> None:
    with wire_server(provider_reply) as wire:
        config: Final = write_config(tmp_path, {"litellm_settings": {"extra_spend_tag_headers": ["x-tenant-id"]}})
        with (
            owned_proxy(gateway, tmp_path, provider_env(wire.url), config=config, workers=2) as candidate,
            candidate.scenario() as scenario,
        ):
            model: Final = scenario.model(model=f"openai/{OPENAI_MODEL}", api_base=f"{wire.url}/v1")
            key: Final = scenario.key()
            response: Final = candidate.request(
                "POST",
                "/v1/chat/completions",
                {"model": model, "messages": [{"role": "user", "content": f"tag me {uuid.uuid4().hex}"}]},
                key=key,
                headers=SENT_HEADERS,
            )
            assert response.status_code == 200, response.text
            request_id: Final = response.json()["id"]
            assert len(wire.drain()) == 1
            assert eventually(lambda: tags_by_id(request_id), lambda tags: len(tags) == 1, seconds=70) == [
                EXPECTED_TAGS
            ]


@pytest.mark.parametrize("stream", [pytest.param(False, id="sync"), pytest.param(True, id="stream")])
def test_unified_responses_records_header_tags(gateway: Gateway, tmp_path: Path, stream: bool) -> None:
    with wire_server(provider_reply) as wire:
        config: Final = write_config(tmp_path, {"litellm_settings": {"extra_spend_tag_headers": ["x-tenant-id"]}})
        with (
            owned_proxy(gateway, tmp_path, provider_env(wire.url), config=config, workers=2) as candidate,
            candidate.scenario() as scenario,
        ):
            model: Final = scenario.model(model=f"openai/{OPENAI_MODEL}", api_base=f"{wire.url}/v1")
            key: Final = scenario.key()
            client: Final = openai.OpenAI(
                api_key=key, base_url=f"{_base_url(candidate)}/v1", default_headers=SENT_HEADERS
            )
            created: Final = client.responses.create(model=model, input=f"tag me {uuid.uuid4().hex}", stream=stream)
            if stream:
                frames: Final = list(created)
                request_id: Final = next(event.response.id for event in frames if event.type == "response.completed")
            else:
                request_id = created.id
            assert request_id.startswith("resp_")
            assert len(wire.drain()) == 1
            assert eventually(lambda: tags_by_key(key), lambda tags: len(tags) == 1, seconds=70) == [EXPECTED_TAGS]


def test_unified_cache_hit_twin_records_header_tags(gateway: Gateway, tmp_path: Path) -> None:
    with wire_server(provider_reply) as wire:
        config: Final = write_config(tmp_path, {"litellm_settings": {"extra_spend_tag_headers": ["x-tenant-id"]}})
        with (
            owned_proxy(gateway, tmp_path, provider_env(wire.url), config=config, workers=2) as candidate,
            candidate.scenario() as scenario,
        ):
            model: Final = scenario.model(model=f"openai/{OPENAI_MODEL}", api_base=f"{wire.url}/v1")
            key: Final = scenario.key()
            body: Final = {"model": model, "messages": [{"role": "user", "content": f"cache {uuid.uuid4().hex}"}]}
            first: Final = candidate.request("POST", "/v1/chat/completions", body, key=key, headers=SENT_HEADERS)
            assert first.status_code == 200, first.text
            second: Final = candidate.request("POST", "/v1/chat/completions", body, key=key, headers=SENT_HEADERS)
            assert second.status_code == 200, second.text
            assert len(wire.drain()) == 1, "identical second call should have hit the response cache"
            assert eventually(lambda: tags_by_key(key), lambda tags: len(tags) == 2, seconds=70) == [
                EXPECTED_TAGS,
                EXPECTED_TAGS,
            ]


def test_pass_through_tags_reach_generic_api_sink(gateway: Gateway, tmp_path: Path) -> None:
    def sink(request: Request) -> Reply:
        return Reply()

    with wire_server(provider_reply) as wire, wire_server(sink) as endpoint:
        config: Final = write_config(
            tmp_path,
            {
                "litellm_settings": {
                    "extra_spend_tag_headers": ["x-tenant-id"],
                    "callbacks": ["generic_api"],
                    "DEFAULT_FLUSH_INTERVAL_SECONDS": 1,
                }
            },
        )
        with (
            owned_proxy(
                gateway,
                tmp_path,
                {
                    **provider_env(wire.url),
                    "GENERIC_LOGGER_ENDPOINT": endpoint.url,
                    "GENERIC_LOGGER_HEADERS": "Authorization=Bearer synthetic-sink-secret",
                },
                config=config,
                workers=2,
            ) as candidate,
            candidate.scenario() as scenario,
        ):
            key: Final = scenario.key()
            response: Final = candidate.request(
                "POST",
                "/anthropic/v1/messages",
                {
                    "model": MODEL,
                    "max_tokens": 16,
                    "messages": [{"role": "user", "content": f"tag me {uuid.uuid4().hex}"}],
                },
                key=key,
                headers={**SENT_HEADERS, "anthropic-version": "2023-06-01"},
            )
            assert response.status_code == 200, response.text
            request_id: Final = response.json()["id"]
            assert len(wire.drain()) == 1
            batches: Final[list[Request]] = []  # mutable-ok: drain consumes the queue between polls

            def delivered() -> Sequence[Mapping]:
                batches.extend(endpoint.drain())
                events: Final = chain.from_iterable(json.loads(batch.body) for batch in batches)
                return [event for event in events if event.get("id") == request_id]

            events: Final = eventually(delivered, lambda values: len(values) == 1, seconds=30)
            assert events[0]["request_tags"] == EXPECTED_TAGS
            assert eventually(lambda: tags_by_id(request_id), lambda tags: len(tags) == 1, seconds=70) == [
                EXPECTED_TAGS
            ]


def test_pass_through_tags_accrue_daily_tag_spend(gateway: Gateway, tmp_path: Path) -> None:
    with wire_server(provider_reply) as wire:
        config: Final = write_config(tmp_path, {"litellm_settings": {"extra_spend_tag_headers": ["x-tenant-id"]}})
        with (
            owned_proxy(gateway, tmp_path, provider_env(wire.url), config=config, workers=2) as candidate,
            candidate.scenario() as scenario,
        ):
            key: Final = scenario.key()
            response: Final = candidate.request(
                "POST",
                "/anthropic/v1/messages",
                {
                    "model": MODEL,
                    "max_tokens": 16,
                    "messages": [{"role": "user", "content": f"tag me {uuid.uuid4().hex}"}],
                },
                key=key,
                headers={**SENT_HEADERS, "anthropic-version": "2023-06-01"},
            )
            assert response.status_code == 200, response.text
            request_id: Final = response.json()["id"]
            assert len(wire.drain()) == 1
            assert eventually(lambda: tags_by_id(request_id), lambda tags: len(tags) == 1, seconds=70) == [
                EXPECTED_TAGS
            ]
            rows: Final = eventually(
                lambda: read_rows(
                    'SELECT tag FROM "LiteLLM_DailyTagSpend" WHERE api_key=%s', (sha256(key.encode()).hexdigest(),)
                ),
                lambda values: len(values) == 3,
                seconds=70,
            )
            assert {row["tag"] for row in rows} == set(EXPECTED_TAGS)


@pytest.mark.parametrize(
    "route", [pytest.param("/anthropic/v1/messages", id="passthrough"), pytest.param("/v1/messages", id="unified")]
)
def test_header_tags_without_extra_spend_tag_headers_record_user_agent_only(
    gateway: Gateway, tmp_path: Path, route: str
) -> None:
    with wire_server(provider_reply) as wire:
        config: Final = write_config(tmp_path, {})
        with (
            owned_proxy(gateway, tmp_path, provider_env(wire.url), config=config, workers=2) as candidate,
            candidate.scenario() as scenario,
        ):
            model: Final = scenario.model(
                model=f"anthropic/{MODEL}", api_base=wire.url, api_key="synthetic-anthropic-key"
            )
            key: Final = scenario.key()
            response: Final = candidate.request(
                "POST",
                route,
                {
                    "model": MODEL if route == "/anthropic/v1/messages" else model,
                    "max_tokens": 16,
                    "messages": [{"role": "user", "content": f"tag me {uuid.uuid4().hex}"}],
                },
                key=key,
                headers={**SENT_HEADERS, "anthropic-version": "2023-06-01"},
            )
            assert response.status_code == 200, response.text
            assert len(wire.drain()) == 1
            expected: Final = [UA_FAMILY_TAG, UA_TAG]
            assert eventually(lambda: tags_by_id(response.json()["id"]), lambda tags: len(tags) == 1, seconds=70) == [
                expected
            ]


@pytest.mark.parametrize(
    "route", [pytest.param("/anthropic/v1/messages", id="passthrough"), pytest.param("/v1/messages", id="unified")]
)
def test_routes_without_headers_record_no_tags(gateway: Gateway, tmp_path: Path, route: str) -> None:
    with wire_server(provider_reply) as wire:
        config: Final = write_config(tmp_path, {"litellm_settings": {"extra_spend_tag_headers": ["x-tenant-id"]}})
        with (
            owned_proxy(gateway, tmp_path, provider_env(wire.url), config=config, workers=2) as candidate,
            candidate.scenario() as scenario,
        ):
            model: Final = scenario.model(
                model=f"anthropic/{MODEL}", api_base=wire.url, api_key="synthetic-anthropic-key"
            )
            key: Final = scenario.key()
            connection: Final = http.client.HTTPConnection("127.0.0.1", candidate.client.base_url.port)
            connection.request(
                "POST",
                route,
                body=json.dumps(
                    {
                        "model": MODEL if route == "/anthropic/v1/messages" else model,
                        "max_tokens": 16,
                        "messages": [{"role": "user", "content": f"tag me {uuid.uuid4().hex}"}],
                    }
                ),
                headers={
                    "authorization": f"Bearer {key}",
                    "content-type": "application/json",
                    "anthropic-version": "2023-06-01",
                },
            )
            raw: Final = connection.getresponse()
            payload: Final = raw.read()
            connection.close()
            assert raw.status == 200, payload
            request_id: Final = json.loads(payload)["id"]
            assert len(wire.drain()) == 1
            assert eventually(lambda: tags_by_id(request_id), lambda tags: len(tags) == 1, seconds=70) == [[]]


@pytest.mark.parametrize(
    "route", [pytest.param("/anthropic/v1/messages", id="passthrough"), pytest.param("/v1/messages", id="unified")]
)
def test_disabled_user_agent_keeps_only_extra_header_tags(gateway: Gateway, tmp_path: Path, route: str) -> None:
    with wire_server(provider_reply) as wire:
        config: Final = write_config(
            tmp_path,
            {
                "litellm_settings": {
                    "extra_spend_tag_headers": ["x-tenant-id"],
                    "disable_add_user_agent_to_request_tags": True,
                }
            },
        )
        with (
            owned_proxy(gateway, tmp_path, provider_env(wire.url), config=config, workers=2) as candidate,
            candidate.scenario() as scenario,
        ):
            model: Final = scenario.model(
                model=f"anthropic/{MODEL}", api_base=wire.url, api_key="synthetic-anthropic-key"
            )
            key: Final = scenario.key()
            response: Final = candidate.request(
                "POST",
                route,
                {
                    "model": MODEL if route == "/anthropic/v1/messages" else model,
                    "max_tokens": 16,
                    "messages": [{"role": "user", "content": f"tag me {uuid.uuid4().hex}"}],
                },
                key=key,
                headers={**SENT_HEADERS, "anthropic-version": "2023-06-01"},
            )
            assert response.status_code == 200, response.text
            assert len(wire.drain()) == 1
            assert eventually(lambda: tags_by_id(response.json()["id"]), lambda tags: len(tags) == 1, seconds=70) == [
                [TENANT_TAG]
            ]


@pytest.mark.parametrize(
    "route", [pytest.param("/anthropic/v1/messages", id="passthrough"), pytest.param("/v1/messages", id="unified")]
)
def test_unsent_configured_header_contributes_no_tag(gateway: Gateway, tmp_path: Path, route: str) -> None:
    with wire_server(provider_reply) as wire:
        config: Final = write_config(tmp_path, {"litellm_settings": {"extra_spend_tag_headers": ["x-never-sent"]}})
        with (
            owned_proxy(gateway, tmp_path, provider_env(wire.url), config=config, workers=2) as candidate,
            candidate.scenario() as scenario,
        ):
            model: Final = scenario.model(
                model=f"anthropic/{MODEL}", api_base=wire.url, api_key="synthetic-anthropic-key"
            )
            key: Final = scenario.key()
            response: Final = candidate.request(
                "POST",
                route,
                {
                    "model": MODEL if route == "/anthropic/v1/messages" else model,
                    "max_tokens": 16,
                    "messages": [{"role": "user", "content": f"tag me {uuid.uuid4().hex}"}],
                },
                key=key,
                headers={**SENT_HEADERS, "anthropic-version": "2023-06-01"},
            )
            assert response.status_code == 200, response.text
            assert len(wire.drain()) == 1
            assert eventually(lambda: tags_by_id(response.json()["id"]), lambda tags: len(tags) == 1, seconds=70) == [
                [UA_FAMILY_TAG, UA_TAG]
            ]


@pytest.mark.parametrize(
    "route", [pytest.param("/anthropic/v1/messages", id="passthrough"), pytest.param("/v1/messages", id="unified")]
)
def test_default_httpx_user_agent_is_recorded(gateway: Gateway, tmp_path: Path, route: str) -> None:
    with wire_server(provider_reply) as wire:
        config: Final = write_config(tmp_path, {"litellm_settings": {"extra_spend_tag_headers": ["x-tenant-id"]}})
        with (
            owned_proxy(gateway, tmp_path, provider_env(wire.url), config=config, workers=2) as candidate,
            candidate.scenario() as scenario,
        ):
            model: Final = scenario.model(
                model=f"anthropic/{MODEL}", api_base=wire.url, api_key="synthetic-anthropic-key"
            )
            key: Final = scenario.key()
            response: Final = candidate.request(
                "POST",
                route,
                {
                    "model": MODEL if route == "/anthropic/v1/messages" else model,
                    "max_tokens": 16,
                    "messages": [{"role": "user", "content": f"tag me {uuid.uuid4().hex}"}],
                },
                key=key,
                headers={"anthropic-version": "2023-06-01"},
            )
            assert response.status_code == 200, response.text
            assert len(wire.drain()) == 1
            expected: Final = ["User-Agent: python-httpx", f"User-Agent: python-httpx/{httpx.__version__}"]
            assert eventually(lambda: tags_by_id(response.json()["id"]), lambda tags: len(tags) == 1, seconds=70) == [
                expected
            ]


@pytest.mark.parametrize("route", [pytest.param("/anthropic/v1/messages", id="passthrough")])
def test_unauthenticated_pass_through_writes_untagged_spend_row(gateway: Gateway, tmp_path: Path, route: str) -> None:
    with wire_server(provider_reply) as wire:
        config: Final = write_config(tmp_path, {"litellm_settings": {"extra_spend_tag_headers": ["x-tenant-id"]}})
        with (
            owned_proxy(gateway, tmp_path, provider_env(wire.url), config=config, workers=2) as candidate,
            candidate.scenario() as scenario,
        ):
            anonymous_before: Final = len(
                read_rows("SELECT request_tags FROM \"LiteLLM_SpendLogs\" WHERE api_key IS NULL OR api_key=''", ())
            )
            response: Final = candidate.client.post(
                route,
                json={
                    "model": MODEL,
                    "max_tokens": 16,
                    "messages": [{"role": "user", "content": f"tag me {uuid.uuid4().hex}"}],
                },
                headers={**SENT_HEADERS, "anthropic-version": "2023-06-01"},
            )
            assert response.status_code == 401, response.text
            anonymous: Final = eventually(
                lambda: read_rows(
                    'SELECT request_tags FROM "LiteLLM_SpendLogs" WHERE api_key IS NULL OR api_key=\'\' ORDER BY "startTime" DESC',
                    (),
                ),
                lambda rows: len(rows) == anonymous_before + 1,
                seconds=70,
            )
            assert tags_of(anonymous[0]) == []
            key: Final = scenario.key()
            control: Final = candidate.request(
                "POST",
                route,
                {
                    "model": MODEL,
                    "max_tokens": 16,
                    "messages": [{"role": "user", "content": f"tag me {uuid.uuid4().hex}"}],
                },
                key=key,
                headers={**SENT_HEADERS, "anthropic-version": "2023-06-01"},
            )
            assert control.status_code == 200, control.text
            assert len(wire.drain()) == 1
            assert eventually(lambda: tags_by_id(control.json()["id"]), lambda tags: len(tags) == 1, seconds=70) == [
                EXPECTED_TAGS
            ]


@pytest.mark.parametrize(
    "route", [pytest.param("/anthropic/v1/messages", id="passthrough"), pytest.param("/v1/messages", id="unified")]
)
def test_upstream_failure_still_records_header_tags(gateway: Gateway, tmp_path: Path, route: str) -> None:
    with wire_server(provider_reply) as wire:
        config: Final = write_config(tmp_path, {"litellm_settings": {"extra_spend_tag_headers": ["x-tenant-id"]}})
        with (
            owned_proxy(gateway, tmp_path, provider_env(wire.url), config=config, workers=2) as candidate,
            candidate.scenario() as scenario,
        ):
            model: Final = scenario.model(
                model="anthropic/claude-nonexistent-model",
                api_base=wire.url,
                api_key="synthetic-anthropic-key",
            )
            key: Final = scenario.key()
            response: Final = candidate.request(
                "POST",
                route,
                {
                    "model": "claude-nonexistent-model" if route == "/anthropic/v1/messages" else model,
                    "max_tokens": 16,
                    "messages": [{"role": "user", "content": f"tag me {uuid.uuid4().hex}"}],
                },
                key=key,
                headers={**SENT_HEADERS, "anthropic-version": "2023-06-01"},
            )
            assert response.status_code == 400, response.text
            assert len(wire.drain()) == 1
            expected: Final = [] if route == "/anthropic/v1/messages" else EXPECTED_TAGS
            assert eventually(lambda: tags_by_key(key), lambda tags: len(tags) == 1, seconds=70) == [expected]


@pytest.mark.parametrize("extra", [pytest.param(None, id="null"), pytest.param([], id="empty")])
@pytest.mark.parametrize(
    "route", [pytest.param("/anthropic/v1/messages", id="passthrough"), pytest.param("/v1/messages", id="unified")]
)
def test_null_and_empty_extra_spend_tag_headers_record_user_agent_only(
    gateway: Gateway, tmp_path: Path, route: str, extra: object
) -> None:
    with wire_server(provider_reply) as wire:
        config: Final = write_config(tmp_path, {"litellm_settings": {"extra_spend_tag_headers": extra}})
        with (
            owned_proxy(gateway, tmp_path, provider_env(wire.url), config=config, workers=2) as candidate,
            candidate.scenario() as scenario,
        ):
            model: Final = scenario.model(
                model=f"anthropic/{MODEL}", api_base=wire.url, api_key="synthetic-anthropic-key"
            )
            key: Final = scenario.key()
            response: Final = candidate.request(
                "POST",
                route,
                {
                    "model": MODEL if route == "/anthropic/v1/messages" else model,
                    "max_tokens": 16,
                    "messages": [{"role": "user", "content": f"tag me {uuid.uuid4().hex}"}],
                },
                key=key,
                headers={**SENT_HEADERS, "anthropic-version": "2023-06-01"},
            )
            assert response.status_code == 200, response.text
            assert len(wire.drain()) == 1
            assert eventually(lambda: tags_by_id(response.json()["id"]), lambda tags: len(tags) == 1, seconds=70) == [
                [UA_FAMILY_TAG, UA_TAG]
            ]


def test_configured_header_case_differs_between_routes(gateway: Gateway, tmp_path: Path) -> None:
    with wire_server(provider_reply) as wire:
        config: Final = write_config(tmp_path, {"litellm_settings": {"extra_spend_tag_headers": ["X-Tenant-Id"]}})
        with (
            owned_proxy(gateway, tmp_path, provider_env(wire.url), config=config, workers=2) as candidate,
            candidate.scenario() as scenario,
        ):
            model: Final = scenario.model(
                model=f"anthropic/{MODEL}", api_base=wire.url, api_key="synthetic-anthropic-key"
            )
            key: Final = scenario.key()
            body: Final = {
                "model": MODEL,
                "max_tokens": 16,
                "messages": [{"role": "user", "content": f"tag me {uuid.uuid4().hex}"}],
            }
            passthrough: Final = candidate.request(
                "POST",
                "/anthropic/v1/messages",
                body,
                key=key,
                headers={**SENT_HEADERS, "anthropic-version": "2023-06-01"},
            )
            assert passthrough.status_code == 200, passthrough.text
            unified: Final = candidate.request(
                "POST",
                "/v1/messages",
                {**body, "model": model},
                key=key,
                headers={**SENT_HEADERS, "anthropic-version": "2023-06-01"},
            )
            assert unified.status_code == 200, unified.text
            assert len(wire.drain()) == 2
            rows: Final = eventually(lambda: tags_by_key(key), lambda tags: len(tags) == 2, seconds=70)
            passthrough_tags: Final = tags_by_id(passthrough.json()["id"])[0]
            unified_tags: Final = tags_by_id(unified.json()["id"])[0]
            assert passthrough_tags == [UA_FAMILY_TAG, UA_TAG, "X-Tenant-Id: tenant-a"], rows
            assert unified_tags == [UA_FAMILY_TAG, UA_TAG], rows


@pytest.mark.parametrize(
    "route", [pytest.param("/anthropic/v1/messages", id="passthrough"), pytest.param("/v1/messages", id="unified")]
)
def test_large_header_value_is_stored_verbatim(gateway: Gateway, tmp_path: Path, route: str) -> None:
    big: Final = "x" * 5000
    with wire_server(provider_reply) as wire:
        config: Final = write_config(tmp_path, {"litellm_settings": {"extra_spend_tag_headers": ["x-tenant-id"]}})
        with (
            owned_proxy(gateway, tmp_path, provider_env(wire.url), config=config, workers=2) as candidate,
            candidate.scenario() as scenario,
        ):
            model: Final = scenario.model(
                model=f"anthropic/{MODEL}", api_base=wire.url, api_key="synthetic-anthropic-key"
            )
            key: Final = scenario.key()
            response: Final = candidate.request(
                "POST",
                route,
                {
                    "model": MODEL if route == "/anthropic/v1/messages" else model,
                    "max_tokens": 16,
                    "messages": [{"role": "user", "content": f"tag me {uuid.uuid4().hex}"}],
                },
                key=key,
                headers={
                    "user-agent": "claude-cli/2.0.0",
                    "x-tenant-id": big,
                    "anthropic-version": "2023-06-01",
                },
            )
            assert response.status_code == 200, response.text
            assert len(wire.drain()) == 1
            assert eventually(lambda: tags_by_id(response.json()["id"]), lambda tags: len(tags) == 1, seconds=70) == [
                [UA_FAMILY_TAG, UA_TAG, f"x-tenant-id: {big}"]
            ]


def test_duplicate_header_values_follow_carrier_semantics(gateway: Gateway, tmp_path: Path) -> None:
    with wire_server(provider_reply) as wire:
        config: Final = write_config(tmp_path, {"litellm_settings": {"extra_spend_tag_headers": ["x-tenant-id"]}})
        with (
            owned_proxy(gateway, tmp_path, provider_env(wire.url), config=config, workers=2) as candidate,
            candidate.scenario() as scenario,
        ):
            model: Final = scenario.model(
                model=f"anthropic/{MODEL}", api_base=wire.url, api_key="synthetic-anthropic-key"
            )
            key: Final = scenario.key()
            body: Final = {
                "model": MODEL,
                "max_tokens": 16,
                "messages": [{"role": "user", "content": f"tag me {uuid.uuid4().hex}"}],
            }
            duplicated: Final = [
                ("Authorization", f"Bearer {key}"),
                ("user-agent", "claude-cli/2.0.0"),
                ("anthropic-version", "2023-06-01"),
                ("x-tenant-id", "t1"),
                ("x-tenant-id", "t2"),
            ]
            passthrough: Final = candidate.client.post("/anthropic/v1/messages", json=body, headers=duplicated)
            assert passthrough.status_code == 200, passthrough.text
            unified: Final = candidate.client.post("/v1/messages", json={**body, "model": model}, headers=duplicated)
            assert unified.status_code == 200, unified.text
            assert len(wire.drain()) == 2
            eventually(lambda: tags_by_key(key), lambda tags: len(tags) == 2, seconds=70)
            assert tags_by_id(passthrough.json()["id"])[0] == [UA_FAMILY_TAG, UA_TAG, "x-tenant-id: t1"]
            assert tags_by_id(unified.json()["id"])[0] == [UA_FAMILY_TAG, UA_TAG, "x-tenant-id: t2"]


@pytest.mark.parametrize(
    "route", [pytest.param("/anthropic/v1/messages", id="passthrough"), pytest.param("/v1/messages", id="unified")]
)
def test_x_litellm_tags_merges_with_header_tags(gateway: Gateway, tmp_path: Path, route: str) -> None:
    with wire_server(provider_reply) as wire:
        config: Final = write_config(tmp_path, {"litellm_settings": {"extra_spend_tag_headers": ["x-tenant-id"]}})
        with (
            owned_proxy(gateway, tmp_path, provider_env(wire.url), config=config, workers=2) as candidate,
            candidate.scenario() as scenario,
        ):
            model: Final = scenario.model(
                model=f"anthropic/{MODEL}", api_base=wire.url, api_key="synthetic-anthropic-key"
            )
            key: Final = scenario.key()
            response: Final = candidate.request(
                "POST",
                route,
                {
                    "model": MODEL if route == "/anthropic/v1/messages" else model,
                    "max_tokens": 16,
                    "messages": [{"role": "user", "content": f"tag me {uuid.uuid4().hex}"}],
                },
                key=key,
                headers={**SENT_HEADERS, "anthropic-version": "2023-06-01", "x-litellm-tags": "team-x"},
            )
            assert response.status_code == 200, response.text
            assert len(wire.drain()) == 1
            expected: Final = ["team-x", UA_FAMILY_TAG, UA_TAG, TENANT_TAG]
            assert eventually(lambda: tags_by_id(response.json()["id"]), lambda tags: len(tags) == 1, seconds=70) == [
                expected
            ]


@pytest.mark.parametrize(
    "route", [pytest.param("/anthropic/v1/messages", id="passthrough"), pytest.param("/v1/messages", id="unified")]
)
def test_repeated_requests_each_record_tags(gateway: Gateway, tmp_path: Path, route: str) -> None:
    with wire_server(provider_reply) as wire:
        config: Final = write_config(tmp_path, {"litellm_settings": {"extra_spend_tag_headers": ["x-tenant-id"]}})
        with (
            owned_proxy(gateway, tmp_path, provider_env(wire.url), config=config, workers=2) as candidate,
            candidate.scenario() as scenario,
        ):
            model: Final = scenario.model(
                model=f"anthropic/{MODEL}", api_base=wire.url, api_key="synthetic-anthropic-key"
            )
            key: Final = scenario.key()

            def repeat(_: int) -> httpx.Response:
                return candidate.request(
                    "POST",
                    route,
                    {
                        "model": MODEL if route == "/anthropic/v1/messages" else model,
                        "max_tokens": 16,
                        "messages": [{"role": "user", "content": f"repeat {uuid.uuid4().hex}"}],
                    },
                    key=key,
                    headers={**SENT_HEADERS, "anthropic-version": "2023-06-01"},
                )

            responses: Final = tuple(repeat(index) for index in range(3))
            assert all(response.status_code == 200 for response in responses), [
                (response.status_code, response.text) for response in responses
            ]
            ids: Final = tuple(response.json()["id"] for response in responses)
            assert len(wire.drain()) == 3
            for identity in ids:
                assert eventually(
                    lambda identity=identity: tags_by_id(identity), lambda tags: len(tags) == 1, seconds=70
                ) == [EXPECTED_TAGS]


def test_guardrail_mode_tag_decider_is_unchanged_on_pass_through(gateway: Gateway, tmp_path: Path) -> None:
    with wire_server(provider_reply) as wire:
        config: Final = write_config(
            tmp_path,
            {
                "litellm_settings": {"extra_spend_tag_headers": ["x-tenant-id"]},
                "general_settings": {
                    "pass_through_endpoints": [
                        {
                            "path": "/custom-anthropic",
                            "target": f"{wire.url}/v1/messages",
                            "auth": True,
                            "guardrails": ["tag-blocker"],
                            "headers": {
                                "x-api-key": "synthetic-anthropic-key",
                                "anthropic-version": "2023-06-01",
                            },
                        }
                    ]
                },
                "guardrails": [
                    {
                        "guardrail_name": "tag-blocker",
                        "litellm_params": {
                            "guardrail": "litellm_content_filter",
                            "blocked_words": [{"keyword": "bananablock", "action": "BLOCK"}],
                            "mode": {"tags": {UA_FAMILY_TAG: "pre_call"}, "default": "post_call"},
                            "default_on": True,
                        },
                    }
                ],
            },
        )
        with (
            owned_proxy(gateway, tmp_path, provider_env(wire.url), config=config, workers=2) as candidate,
            candidate.scenario() as scenario,
        ):
            model: Final = scenario.model(model=f"openai/{OPENAI_MODEL}", api_base=f"{wire.url}/v1")
            key: Final = scenario.key(allowed_passthrough_routes=["/custom-anthropic"])
            body: Final = {"model": MODEL, "max_tokens": 16, "messages": [{"role": "user", "content": "bananablock"}]}
            passthrough: Final = candidate.request("POST", "/custom-anthropic", body, key=key, headers=SENT_HEADERS)
            assert passthrough.status_code == 200, passthrough.text
            assert len(wire.drain()) == 1
            control: Final = candidate.request(
                "POST",
                "/v1/chat/completions",
                {"model": model, "messages": [{"role": "user", "content": "bananablock"}]},
                key=key,
                headers=SENT_HEADERS,
            )
            assert control.status_code != 200, control.text
            assert len(wire.drain()) == 0, "tag-matched guardrail should have blocked before the upstream"
            digest: Final = sha256(key.encode()).hexdigest()

            def blocked_rows() -> Sequence[Mapping]:
                return [
                    row
                    for row in read_rows(
                        'SELECT metadata FROM "LiteLLM_SpendLogs" WHERE api_key=%s',
                        (digest,),
                    )
                    if "Content blocked: keyword 'bananablock' detected" in json.dumps(row["metadata"])
                    and row["metadata"].get("status") == "failure"
                ]

            assert eventually(blocked_rows, lambda rows: len(rows) == 1, seconds=70), (
                "guardrail block was not recorded on the key's spend row"
            )
