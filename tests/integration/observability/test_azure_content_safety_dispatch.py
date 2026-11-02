import asyncio
import concurrent.futures
import json
import threading
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Final

import anthropic
import httpx
import openai
import pytest
from integration._support.client import Gateway, Scenario, eventually, gateway_from_environment, object_value
from integration._support.database import read_rows
from integration._support.wire import Wire
from integration.observability.azure_dispatch_support import (
    ATTACK_MARKER,
    AzureBehavior,
    MODERATION_MARKER,
    OVERSIZED_MARKER,
    PROVIDER_401_MARKER,
    azure_texts,
    dispatch_rig,
    provider_messages,
    provider_texts,
)
from pydantic import JsonValue

_ATTACK_MARKER: Final = ATTACK_MARKER
_MODERATION_MARKER: Final = MODERATION_MARKER


@pytest.fixture(scope="module")
def azure_rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[tuple[Gateway, Wire, Wire]]:
    directory: Final = tmp_path_factory.mktemp("azure-content-safety-dispatch")
    with gateway_from_environment() as gateway:
        with dispatch_rig(gateway, directory) as (owned, azure, provider, _):
            yield owned.gateway, azure, provider


@pytest.fixture
def key_update_rig(tmp_path: Path) -> Iterator[tuple[Gateway, Wire, Wire, AzureBehavior]]:
    behavior: Final = AzureBehavior(
        entered=threading.Event(),
        release=threading.Event(),
        barrier_marker="E3_BLOCK",
    )
    with gateway_from_environment() as gateway:
        with dispatch_rig(gateway, tmp_path, behavior=behavior) as (owned, azure, provider, _):
            yield owned.gateway, azure, provider, behavior


@pytest.fixture(autouse=True)
def _clear_wires(azure_rig: tuple[Gateway, Wire, Wire]) -> None:
    azure_rig[1].drain()
    azure_rig[2].drain()


def _model(scenario: Scenario, provider: Wire, model: str = "openai/gpt-4o-mini") -> str:
    return scenario.model(
        model=model,
        api_base=provider.url + "/v1",
        api_key="synthetic-provider-key",
    )


def _guardrail_entry(response: httpx.Response) -> dict[str, JsonValue]:
    request_id: Final = response.headers["x-litellm-call-id"]
    rows: Final = eventually(
        lambda: read_rows('SELECT metadata FROM "LiteLLM_SpendLogs" WHERE litellm_call_id=%s', (request_id,)),
        lambda values: len(values) == 1,
        seconds=70,
    )
    metadata_value: Final = rows[0]["metadata"]
    metadata: Final = object_value(json.loads(metadata_value) if isinstance(metadata_value, str) else metadata_value)
    entries: Final = metadata["guardrail_information"]
    assert isinstance(entries, list) and len(entries) == 1, f"{response.text}: {metadata}"
    return object_value(entries[0])


def _spend_metadata(request_id: str) -> dict[str, JsonValue]:
    rows: Final = eventually(
        lambda: read_rows('SELECT metadata FROM "LiteLLM_SpendLogs" WHERE litellm_call_id=%s', (request_id,)),
        lambda values: len(values) == 1,
        seconds=70,
    )
    metadata_value: Final = rows[0]["metadata"]
    return object_value(json.loads(metadata_value) if isinstance(metadata_value, str) else metadata_value)


def _chat_body(
    model: str,
    prompt: str,
    guardrails: tuple[str, ...],
    *,
    stream: bool = False,
    no_cache: bool = False,
) -> dict[str, JsonValue]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        **({"guardrails": list(guardrails)} if guardrails else {}),
        **({"stream": True} if stream else {}),
        **({"cache": {"no-cache": True}} if no_cache else {}),
    }


def _messages_body(model: str, prompt: str, guardrails: tuple[str, ...], *, stream: bool = False) -> dict[str, JsonValue]:
    return {
        "model": model,
        "max_tokens": 16,
        "messages": [{"role": "user", "content": prompt}],
        **({"guardrails": list(guardrails)} if guardrails else {}),
        **({"stream": True} if stream else {}),
    }


def _responses_body(model: str, value: JsonValue, guardrails: tuple[str, ...], *, stream: bool = False) -> dict[str, JsonValue]:
    return {
        "model": model,
        "input": value,
        **({"guardrails": list(guardrails)} if guardrails else {}),
        **({"stream": True} if stream else {}),
    }


def _stream_request(candidate: Gateway, path: str, body: dict[str, JsonValue]) -> tuple[int, str, dict[str, str]]:
    with candidate.client.stream(
        "POST",
        path,
        json=body,
        headers={"Authorization": f"Bearer {candidate.key}"},
    ) as response:
        response.read()
        return response.status_code, response.text, dict(response.headers)


def _chat_stream_delta_content(event: dict[str, JsonValue]) -> str:
    choices: Final = event["choices"]
    assert isinstance(choices, list), event
    return "".join(
        _chat_stream_choice_content(choice)
        for choice in choices
    )


def _chat_stream_choice_content(choice: JsonValue) -> str:
    delta: Final = object_value(object_value(choice)["delta"])
    content: Final = delta.get("content")
    return content if isinstance(content, str) else ""


def _chat_stream_content(response_text: str) -> tuple[str, bool]:
    events: Final = tuple(
        line.removeprefix("data: ")
        for line in response_text.splitlines()
        if line.startswith("data: ")
    )
    content_events: Final = tuple(event for event in events if event != "[DONE]")
    chunks: Final = tuple(object_value(json.loads(event)) for event in content_events)
    content: Final = "".join(_chat_stream_delta_content(chunk) for chunk in chunks)
    return content, bool(events) and events[-1] == "[DONE]"


def _openai_chat_sync(
    base_url: str, key: str, model: str, prompt: str, guardrails: tuple[str, ...]
) -> None:
    client: Final = openai.OpenAI(base_url=base_url, api_key=key, max_retries=0)
    with client:
        client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=8,
            extra_body={"guardrails": list(guardrails)},
        )


async def _openai_chat_async(
    base_url: str, key: str, model: str, prompt: str, guardrails: tuple[str, ...]
) -> None:
    client: Final = openai.AsyncOpenAI(base_url=base_url, api_key=key, max_retries=0)
    async with client:
        await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=8,
            extra_body={"guardrails": list(guardrails)},
        )


def _anthropic_messages_sync(base_url: str, key: str, model: str, prompt: str) -> None:
    client: Final = anthropic.Anthropic(base_url=base_url, api_key=key, max_retries=0)
    with client:
        client.messages.create(
            model=model,
            max_tokens=8,
            messages=[{"role": "user", "content": prompt}],
            extra_body={"guardrails": ["tuple-writer", "shield"]},
        )


async def _anthropic_messages_async(base_url: str, key: str, model: str, prompt: str) -> None:
    client: Final = anthropic.AsyncAnthropic(base_url=base_url, api_key=key, max_retries=0)
    async with client:
        await client.messages.create(
            model=model,
            max_tokens=8,
            messages=[{"role": "user", "content": prompt}],
            extra_body={"guardrails": ["tuple-writer", "shield"]},
        )


def _openai_responses_sync(base_url: str, key: str, model: str, prompt: str) -> None:
    client: Final = openai.OpenAI(base_url=base_url, api_key=key, max_retries=0)
    with client:
        client.responses.create(
            model=model,
            input=prompt,
            extra_body={"guardrails": ["shield"]},
        )


async def _openai_responses_async(base_url: str, key: str, model: str, prompt: str) -> None:
    client: Final = openai.AsyncOpenAI(base_url=base_url, api_key=key, max_retries=0)
    async with client:
        await client.responses.create(
            model=model,
            input=prompt,
            extra_body={"guardrails": ["shield"]},
        )


@pytest.mark.parametrize(
    ("guardrail_name", "marker"),
    [("shield", _ATTACK_MARKER), ("moderation", _MODERATION_MARKER)],
    ids=("H1-shield", "H1-moderation"),
)
def test_h1_tuple_attack_is_scanned_for_each_guardrail(
    azure_rig: tuple[Gateway, Wire, Wire], guardrail_name: str, marker: str
) -> None:
    candidate, azure, provider = azure_rig
    prompt: Final = f"synthetic prompt {marker} {uuid.uuid4().hex}"
    with candidate.scenario() as scenario:
        model: Final = _model(scenario, provider)
        response: Final = candidate.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "guardrails": ["tuple-writer", guardrail_name],
            },
        )
        assert response.status_code == 400, response.text
        assert _azure_texts(azure) == (prompt,), response.text
        assert provider.drain() == (), response.text


@pytest.mark.parametrize(
    ("guardrail_name", "marker"),
    [("shield", "benign shield tuple"), ("moderation", "benign moderation tuple")],
    ids=("H2-shield", "H2-moderation"),
)
def test_h2_tuple_benign_is_scanned_and_spent(
    azure_rig: tuple[Gateway, Wire, Wire], guardrail_name: str, marker: str
) -> None:
    candidate, azure, provider = azure_rig
    prompt: Final = f"{marker} {uuid.uuid4().hex}"
    with candidate.scenario() as scenario:
        model: Final = _model(scenario, provider)
        response: Final = candidate.request(
            "POST",
            "/v1/chat/completions",
            _chat_body(model, prompt, ("tuple-writer", guardrail_name)),
        )
        assert response.status_code == 200, response.text
        assert _azure_texts(azure) == (prompt,), response.text
        assert provider_texts(provider) == (prompt,), response.text
        assert _guardrail_entry(response)["guardrail_status"] == "success", response.text


@pytest.mark.parametrize(
    ("guardrail_name", "marker"),
    [("shield", _ATTACK_MARKER), ("moderation", _MODERATION_MARKER)],
    ids=("H8-shield-attack", "H8-moderation-attack"),
)
def test_h8_list_attack_control_is_scanned_for_each_guardrail(
    azure_rig: tuple[Gateway, Wire, Wire], guardrail_name: str, marker: str
) -> None:
    candidate, azure, provider = azure_rig
    prompt: Final = f"synthetic prompt {marker} {uuid.uuid4().hex}"
    with candidate.scenario() as scenario:
        model: Final = _model(scenario, provider)
        response: Final = candidate.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "guardrails": [guardrail_name],
            },
        )
        assert response.status_code == 400, response.text
        assert _azure_texts(azure) == (prompt,), response.text
        assert provider.drain() == (), response.text


@pytest.mark.parametrize(
    ("guardrail_name", "marker"),
    [("shield", "benign shield list"), ("moderation", "benign moderation list")],
    ids=("H8-shield", "H8-moderation"),
)
def test_h8_list_benign_is_scanned_and_served(
    azure_rig: tuple[Gateway, Wire, Wire], guardrail_name: str, marker: str
) -> None:
    candidate, azure, provider = azure_rig
    prompt: Final = f"{marker} {uuid.uuid4().hex}"
    with candidate.scenario() as scenario:
        model: Final = _model(scenario, provider)
        response: Final = candidate.request(
            "POST",
            "/v1/chat/completions",
            _chat_body(model, prompt, (guardrail_name,)),
        )
        assert response.status_code == 200, response.text
        assert _azure_texts(azure) == (prompt,), response.text
        assert provider_texts(provider) == (prompt,), response.text


@pytest.mark.parametrize(
    ("guardrail_name", "marker"),
    [("shield", _ATTACK_MARKER), ("moderation", _MODERATION_MARKER)],
    ids=("H3-shield", "H3-moderation"),
)
def test_h3_tuple_attack_chat_stream_is_blocked(
    azure_rig: tuple[Gateway, Wire, Wire], guardrail_name: str, marker: str
) -> None:
    candidate, azure, provider = azure_rig
    prompt: Final = f"stream {marker} {uuid.uuid4().hex}"
    with candidate.scenario() as scenario:
        model: Final = _model(scenario, provider)
        status, text, _ = _stream_request(
            candidate,
            "/v1/chat/completions",
            _chat_body(model, prompt, ("tuple-writer", guardrail_name), stream=True),
        )
        assert status == 400, text
        assert _azure_texts(azure) == (prompt,), text
        assert provider.drain() == (), text


@pytest.mark.parametrize(
    ("guardrail_name", "marker"),
    [("shield", "benign stream shield"), ("moderation", "benign stream moderation")],
    ids=("H4-shield", "H4-moderation"),
)
def test_h4_tuple_benign_chat_stream_reaches_provider_and_spend(
    azure_rig: tuple[Gateway, Wire, Wire], guardrail_name: str, marker: str
) -> None:
    candidate, azure, provider = azure_rig
    prompt: Final = f"{marker} {uuid.uuid4().hex}"
    with candidate.scenario() as scenario:
        model: Final = _model(scenario, provider)
        status, text, headers = _stream_request(
            candidate,
            "/v1/chat/completions",
            _chat_body(model, prompt, ("tuple-writer", guardrail_name), stream=True),
        )
        assert status == 200, text
        streamed_content, done = _chat_stream_content(text)
        assert streamed_content == "permitted response", text
        assert done, text
        assert _azure_texts(azure) == (prompt,), text
        assert provider_texts(provider) == (prompt,), text
        spend_metadata: Final = _spend_metadata(headers["x-litellm-call-id"])
        entries: Final = spend_metadata["guardrail_information"]
        assert isinstance(entries, list) and len(entries) == 1, text
        assert object_value(entries[0])["guardrail_status"] == "success", text


@pytest.mark.parametrize(
    ("guardrail_name", "marker"),
    [("shield", _ATTACK_MARKER), ("moderation", _MODERATION_MARKER)],
    ids=("H5-shield", "H5-moderation"),
)
def test_h5_tuple_attack_anthropic_messages_is_blocked(
    azure_rig: tuple[Gateway, Wire, Wire], guardrail_name: str, marker: str
) -> None:
    candidate, azure, provider = azure_rig
    prompt: Final = f"anthropic message {marker} {uuid.uuid4().hex}"
    with candidate.scenario() as scenario:
        model: Final = scenario.model(
            model="anthropic/claude-sonnet-4-5-20250929",
            api_base=provider.url,
            api_key="synthetic-provider-key",
        )
        response: Final = candidate.request(
            "POST",
            "/v1/messages",
            _messages_body(model, prompt, ("tuple-writer", guardrail_name)),
        )
        assert response.status_code == 400, response.text
        assert _azure_texts(azure) == (prompt,), response.text
        assert provider.drain() == (), response.text


@pytest.mark.parametrize(
    ("guardrail_name", "marker"),
    [("shield", _ATTACK_MARKER), ("moderation", _MODERATION_MARKER)],
    ids=("H6-shield", "H6-moderation"),
)
def test_h6_tuple_attack_anthropic_messages_stream_is_blocked(
    azure_rig: tuple[Gateway, Wire, Wire], guardrail_name: str, marker: str
) -> None:
    candidate, azure, provider = azure_rig
    prompt: Final = f"anthropic stream {marker} {uuid.uuid4().hex}"
    with candidate.scenario() as scenario:
        model: Final = scenario.model(
            model="anthropic/claude-sonnet-4-5-20250929",
            api_base=provider.url,
            api_key="synthetic-provider-key",
        )
        status, text, _ = _stream_request(
            candidate,
            "/v1/messages",
            _messages_body(model, prompt, ("tuple-writer", guardrail_name), stream=True),
        )
        assert status == 400, text
        assert _azure_texts(azure) == (prompt,), text
        assert provider.drain() == (), text


@pytest.mark.parametrize(
    ("guardrail_name", "marker"),
    [("shield", _ATTACK_MARKER), ("moderation", _MODERATION_MARKER)],
    ids=("H7-shield", "H7-moderation"),
)
def test_h7_list_attack_anthropic_messages_control(
    azure_rig: tuple[Gateway, Wire, Wire], guardrail_name: str, marker: str
) -> None:
    candidate, azure, provider = azure_rig
    prompt: Final = f"anthropic list {marker} {uuid.uuid4().hex}"
    with candidate.scenario() as scenario:
        model: Final = scenario.model(
            model="anthropic/claude-sonnet-4-5-20250929",
            api_base=provider.url,
            api_key="synthetic-provider-key",
        )
        response: Final = candidate.request(
            "POST",
            "/v1/messages",
            _messages_body(model, prompt, (guardrail_name,)),
        )
        assert response.status_code == 400, response.text
        assert _azure_texts(azure) == (prompt,), response.text
        assert provider.drain() == (), response.text


@pytest.mark.parametrize(
    ("guardrail_name", "marker"),
    [("shield", _ATTACK_MARKER), ("moderation", _MODERATION_MARKER)],
    ids=("H9-shield", "H9-moderation"),
)
def test_h9_responses_string_attack_is_scanned(
    azure_rig: tuple[Gateway, Wire, Wire], guardrail_name: str, marker: str
) -> None:
    candidate, azure, provider = azure_rig
    prompt: Final = f"synthetic prompt {marker} {uuid.uuid4().hex}"
    with candidate.scenario() as scenario:
        model: Final = _model(scenario, provider)
        response: Final = candidate.request(
            "POST",
            "/v1/responses",
            _responses_body(model, prompt, (guardrail_name,)),
        )
        assert response.status_code == 400, response.text
        assert _azure_texts(azure) == (prompt,), response.text
        assert provider.drain() == (), response.text


@pytest.mark.parametrize(
    ("guardrail_name", "marker"),
    [("shield", "benign responses stream shield"), ("moderation", "benign responses stream moderation")],
    ids=("H9-shield-stream", "H9-moderation-stream"),
)
def test_h9_responses_benign_stream_is_scanned_and_served(
    azure_rig: tuple[Gateway, Wire, Wire], guardrail_name: str, marker: str
) -> None:
    candidate, azure, provider = azure_rig
    prompt: Final = f"{marker} {uuid.uuid4().hex}"
    with candidate.scenario() as scenario:
        model: Final = _model(scenario, provider)
        status, text, headers = _stream_request(
            candidate,
            "/v1/responses",
            _responses_body(model, prompt, (guardrail_name,), stream=True),
        )
        assert status == 200, text
        assert "permitted response" in text, text
        assert _azure_texts(azure) == (prompt,), text
        assert provider_texts(provider) == (prompt,), text
        spend_metadata: Final = _spend_metadata(headers["x-litellm-call-id"])
        entries: Final = spend_metadata["guardrail_information"]
        assert isinstance(entries, list) and len(entries) == 1, text
        assert object_value(entries[0])["guardrail_status"] == "success", text


@pytest.mark.parametrize(
    ("guardrail_name", "marker"),
    [("shield", _ATTACK_MARKER), ("moderation", _MODERATION_MARKER)],
    ids=("H9-shield-list", "H9-moderation-list"),
)
def test_h9_responses_list_input_attack_control(
    azure_rig: tuple[Gateway, Wire, Wire], guardrail_name: str, marker: str
) -> None:
    candidate, azure, provider = azure_rig
    prompt: Final = f"responses list {marker} {uuid.uuid4().hex}"
    with candidate.scenario() as scenario:
        model: Final = _model(scenario, provider)
        response: Final = candidate.request(
            "POST",
            "/v1/responses",
            _responses_body(
                model,
                [{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
                (guardrail_name,),
            ),
        )
        assert response.status_code == 400, response.text
        assert _azure_texts(azure) == (prompt,), response.text
        assert provider.drain() == (), response.text


@pytest.mark.parametrize(
    ("guardrail_name", "marker"),
    [("all-turns-shield", _ATTACK_MARKER), ("all-turns-moderation", _MODERATION_MARKER)],
    ids=("H10-shield", "H10-moderation"),
)
def test_h10_subclass_override_scans_every_user_turn(
    azure_rig: tuple[Gateway, Wire, Wire], guardrail_name: str, marker: str
) -> None:
    candidate, azure, provider = azure_rig
    first_prompt: Final = f"synthetic prompt {marker} {uuid.uuid4().hex}"
    expected_prompt: Final = first_prompt + "\nbenign final user turn"
    with candidate.scenario() as scenario:
        model: Final = _model(scenario, provider)
        response: Final = candidate.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [
                    {"role": "user", "content": first_prompt},
                    {"role": "assistant", "content": "ok"},
                    {"role": "user", "content": "benign final user turn"},
                ],
                "guardrails": [guardrail_name],
            },
        )
        assert response.status_code == 400, response.text
        assert _azure_texts(azure) == (expected_prompt,), response.text
        assert provider.drain() == (), response.text


@pytest.mark.parametrize(
    ("guardrail_name", "marker"),
    [("all-turns-shield", _ATTACK_MARKER), ("all-turns-moderation", _MODERATION_MARKER)],
    ids=("H11-shield", "H11-moderation"),
)
def test_h11_tuple_subclass_override_scans_every_user_turn(
    azure_rig: tuple[Gateway, Wire, Wire], guardrail_name: str, marker: str
) -> None:
    candidate, azure, provider = azure_rig
    first_prompt: Final = f"tuple override {marker} {uuid.uuid4().hex}"
    expected_prompt: Final = first_prompt + "\nbenign final user turn"
    with candidate.scenario() as scenario:
        model: Final = _model(scenario, provider)
        response: Final = candidate.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [
                    {"role": "user", "content": first_prompt},
                    {"role": "assistant", "content": "ok"},
                    {"role": "user", "content": "benign final user turn"},
                ],
                "guardrails": ["tuple-writer", guardrail_name],
            },
        )
        assert response.status_code == 400, response.text
        assert _azure_texts(azure) == (expected_prompt,), response.text
        assert provider.drain() == (), response.text


@pytest.mark.parametrize(
    ("guardrail_name", "marker"),
    [("requiring-shield", "benign shield prompt"), ("requiring-moderation", "benign moderation prompt")],
    ids=("H12-shield-benign", "H12-moderation-benign"),
)
def test_h12_guardrail_subclass_can_call_get_user_prompt(
    azure_rig: tuple[Gateway, Wire, Wire], guardrail_name: str, marker: str
) -> None:
    candidate, azure, provider = azure_rig
    prompt: Final = f"{marker} {uuid.uuid4().hex}"
    with candidate.scenario() as scenario:
        model: Final = _model(scenario, provider)
        response: Final = candidate.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "guardrails": [guardrail_name],
            },
        )
        assert response.status_code == 200, response.text
        assert "permitted response" in response.text, response.text
        assert _azure_texts(azure) == (prompt,), response.text
        assert provider_texts(provider) == (prompt,), response.text


@pytest.mark.parametrize(
    ("guardrail_name", "marker"),
    [("requiring-shield", _ATTACK_MARKER), ("requiring-moderation", _MODERATION_MARKER)],
    ids=("H12-shield-attack", "H12-moderation-attack"),
)
def test_h12_subclass_call_blocks_attack(
    azure_rig: tuple[Gateway, Wire, Wire], guardrail_name: str, marker: str
) -> None:
    candidate, azure, provider = azure_rig
    prompt: Final = f"required method {marker} {uuid.uuid4().hex}"
    with candidate.scenario() as scenario:
        model: Final = _model(scenario, provider)
        response: Final = candidate.request(
            "POST",
            "/v1/chat/completions",
            _chat_body(model, prompt, (guardrail_name,)),
        )
        assert response.status_code == 400, response.text
        assert _azure_texts(azure) == (prompt,), response.text
        assert provider.drain() == (), response.text


@pytest.mark.parametrize("guardrail_name", ["shield", "moderation"], ids=("H13-shield", "H13-moderation"))
def test_h13_messages_less_embeddings_log_allow_without_azure_request(
    azure_rig: tuple[Gateway, Wire, Wire], guardrail_name: str
) -> None:
    candidate, azure, provider = azure_rig
    with candidate.scenario() as scenario:
        model: Final = _model(scenario, provider, model="openai/text-embedding-3-small")
        response: Final = candidate.request(
            "POST",
            "/v1/embeddings",
            {"model": model, "input": "synthetic benign embedding text", "guardrails": [guardrail_name]},
        )
        assert response.status_code == 200, response.text
        assert _azure_texts(azure) == (), response.text
        assert provider_texts(provider) == ("synthetic benign embedding text",), response.text
        entry: Final = _guardrail_entry(response)
        assert entry["guardrail_status"] == "success", response.text
        assert entry["guardrail_response"] == "allow", response.text


@pytest.mark.parametrize("guardrail_name", ["shield", "moderation"], ids=("H14-shield", "H14-moderation"))
def test_h14_completions_without_messages_log_allow(
    azure_rig: tuple[Gateway, Wire, Wire], guardrail_name: str
) -> None:
    candidate, azure, provider = azure_rig
    prompt: Final = f"synthetic completion input {uuid.uuid4().hex}"
    with candidate.scenario() as scenario:
        model: Final = _model(scenario, provider, model="openai/gpt-3.5-turbo-instruct")
        response: Final = candidate.request(
            "POST",
            "/v1/completions",
            {"model": model, "prompt": prompt, "guardrails": [guardrail_name]},
        )
        assert response.status_code == 200, response.text
        assert _azure_texts(azure) == (), response.text
        assert provider_texts(provider) == (prompt,), response.text
        entry: Final = _guardrail_entry(response)
        assert entry["guardrail_status"] == "success", response.text
        assert entry["guardrail_response"] == "allow", response.text


@pytest.mark.parametrize(
    ("guardrail_name", "marker"),
    [("shield", "cache benign shield"), ("moderation", "cache benign moderation")],
    ids=("C1-shield", "C1-moderation"),
)
def test_c1_tuple_benign_cache_twins_scan_each_request(
    azure_rig: tuple[Gateway, Wire, Wire], guardrail_name: str, marker: str
) -> None:
    candidate, azure, provider = azure_rig
    prompt: Final = f"{marker} {uuid.uuid4().hex}"
    with candidate.scenario() as scenario:
        model: Final = _model(scenario, provider)
        body: Final = _chat_body(model, prompt, ("tuple-writer", guardrail_name))
        first: Final = candidate.request("POST", "/v1/chat/completions", body)
        second: Final = candidate.request("POST", "/v1/chat/completions", body)
        assert first.status_code == 200, first.text
        assert second.status_code == 200, second.text
        assert _azure_texts(azure) == (prompt, prompt), second.text
        assert provider_texts(provider) == (prompt,), second.text
        assert _guardrail_entry(first)["guardrail_status"] == "success", first.text
        assert _guardrail_entry(second)["guardrail_status"] == "success", second.text


@pytest.mark.parametrize(
    ("guardrail_name", "marker"),
    [("shield", ATTACK_MARKER), ("moderation", MODERATION_MARKER)],
    ids=("C2-shield", "C2-moderation"),
)
def test_c2_tuple_attack_cache_twins_are_both_blocked(
    azure_rig: tuple[Gateway, Wire, Wire], guardrail_name: str, marker: str
) -> None:
    candidate, azure, provider = azure_rig
    prompt: Final = f"cache attack {marker} {uuid.uuid4().hex}"
    with candidate.scenario() as scenario:
        model: Final = _model(scenario, provider)
        body: Final = _chat_body(model, prompt, ("tuple-writer", guardrail_name))
        first: Final = candidate.request("POST", "/v1/chat/completions", body)
        second: Final = candidate.request("POST", "/v1/chat/completions", body)
        assert first.status_code == 400, first.text
        assert second.status_code == 400, second.text
        assert _azure_texts(azure) == (prompt, prompt), second.text
        assert provider.drain() == (), second.text


@pytest.mark.parametrize("guardrail_name", ["shield", "moderation"], ids=("C3-shield", "C3-moderation"))
def test_c3_embeddings_cache_twins_keep_allow_rows(
    azure_rig: tuple[Gateway, Wire, Wire], guardrail_name: str
) -> None:
    candidate, azure, provider = azure_rig
    prompt: Final = f"cache embedding {uuid.uuid4().hex}"
    with candidate.scenario() as scenario:
        model: Final = _model(scenario, provider, model="openai/text-embedding-3-small")
        body: Final = {"model": model, "input": prompt, "guardrails": [guardrail_name]}
        first: Final = candidate.request("POST", "/v1/embeddings", body)
        second: Final = candidate.request("POST", "/v1/embeddings", body)
        assert first.status_code == 200, first.text
        assert second.status_code == 200, second.text
        assert _azure_texts(azure) == (), second.text
        assert provider_texts(provider) == (prompt,), second.text
        assert _guardrail_entry(first)["guardrail_response"] == "allow", first.text
        assert _guardrail_entry(second)["guardrail_response"] == "allow", second.text


@pytest.mark.parametrize(
    ("guardrail_name", "marker"),
    [("shield", ATTACK_MARKER), ("moderation", MODERATION_MARKER)],
    ids=("C4-shield", "C4-moderation"),
)
def test_c4_list_attack_cache_twins_are_both_blocked(
    azure_rig: tuple[Gateway, Wire, Wire], guardrail_name: str, marker: str
) -> None:
    candidate, azure, provider = azure_rig
    prompt: Final = f"cache list {marker} {uuid.uuid4().hex}"
    with candidate.scenario() as scenario:
        model: Final = _model(scenario, provider)
        body: Final = _chat_body(model, prompt, (guardrail_name,))
        first: Final = candidate.request("POST", "/v1/chat/completions", body)
        second: Final = candidate.request("POST", "/v1/chat/completions", body)
        assert first.status_code == 400, first.text
        assert second.status_code == 400, second.text
        assert _azure_texts(azure) == (prompt, prompt), second.text
        assert provider.drain() == (), second.text


@pytest.mark.parametrize(
    ("guardrail_name", "marker"),
    [("shield", ATTACK_MARKER), ("moderation", MODERATION_MARKER)],
    ids=("C5-shield", "C5-moderation"),
)
def test_c5_anthropic_tuple_attack_cache_twins_are_blocked(
    azure_rig: tuple[Gateway, Wire, Wire], guardrail_name: str, marker: str
) -> None:
    candidate, azure, provider = azure_rig
    prompt: Final = f"Anthropic cache {marker} {uuid.uuid4().hex}"
    with candidate.scenario() as scenario:
        model: Final = scenario.model(
            model="anthropic/claude-sonnet-4-5-20250929",
            api_base=provider.url,
            api_key="synthetic-provider-key",
        )
        body: Final = _messages_body(model, prompt, ("tuple-writer", guardrail_name))
        first: Final = candidate.request("POST", "/v1/messages", body)
        second: Final = candidate.request("POST", "/v1/messages", body)
        assert first.status_code == 400, first.text
        assert second.status_code == 400, second.text
        assert _azure_texts(azure) == (prompt, prompt), second.text
        assert provider.drain() == (), second.text


@pytest.mark.parametrize("guardrail_name", ["shield", "moderation"], ids=("C6-shield", "C6-moderation"))
def test_c6_responses_benign_cache_twins_scan_each_request(
    azure_rig: tuple[Gateway, Wire, Wire], guardrail_name: str
) -> None:
    candidate, azure, provider = azure_rig
    prompt: Final = f"Responses cache benign {uuid.uuid4().hex}"
    with candidate.scenario() as scenario:
        model: Final = _model(scenario, provider)
        body: Final = _responses_body(model, prompt, (guardrail_name,))
        first: Final = candidate.request("POST", "/v1/responses", body)
        second: Final = candidate.request("POST", "/v1/responses", body)
        assert first.status_code == 200, first.text
        assert second.status_code == 200, second.text
        assert _azure_texts(azure) == (prompt, prompt), second.text
        assert provider_texts(provider) == (prompt,), second.text


def test_s1_integer_messages_fails_without_dispatching_edges(azure_rig: tuple[Gateway, Wire, Wire]) -> None:
    candidate, azure, provider = azure_rig
    with candidate.scenario() as scenario:
        model: Final = _model(scenario, provider)
        response: Final = candidate.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": 5, "guardrails": ["shield"]},
        )
        assert response.status_code == 500, response.text
        assert "error" in response.json(), response.text
        assert _azure_texts(azure) == (), response.text
        assert provider.drain() == (), response.text


@pytest.mark.parametrize(
    ("shape", "expected_status"),
    [
        ("string", 400),
        ("object", 200),
        ("empty-list", 200),
        ("oversized", 400),
        ("null", 400),
        ("missing", 400),
    ],
    ids=(
        "S2-string",
        "S2-object",
        "S2-empty-list",
        "S2-oversized",
        "S2-null",
        "S2-missing",
    ),
)
def test_s2_malformed_messages_shapes_match_base_pin(
    azure_rig: tuple[Gateway, Wire, Wire], shape: str, expected_status: int
) -> None:
    candidate, azure, provider = azure_rig
    prompt: Final = f"shape control {uuid.uuid4().hex}"
    with candidate.scenario() as scenario:
        model: Final = _model(scenario, provider)
        messages: Final = {
            "string": "malformed messages",
            "object": {},
            "empty-list": [],
            "oversized": "x" * 5000,
            "null": None,
            "missing": None,
        }[shape]
        body: Final = {
            "model": model,
            "messages": messages,
            "guardrails": ["shield"],
        }
        raw: Final = json.dumps(
            {"model": model, "guardrails": ["shield"]}
            if shape == "missing"
            else body
        )
        response: Final = candidate.client.post(
            "/v1/chat/completions",
            content=raw,
            headers={"Authorization": f"Bearer {candidate.key}", "Content-Type": "application/json"},
        )
        assert response.status_code == expected_status, f"{response.status_code}: {response.text}"
        assert _azure_texts(azure) == (), response.text
        provider.drain()


def test_s2d_duplicate_messages_key_matches_single_key_request(
    azure_rig: tuple[Gateway, Wire, Wire],
) -> None:
    candidate, azure, provider = azure_rig
    prompt: Final = f"duplicate messages key {uuid.uuid4().hex}"
    with candidate.scenario() as scenario:
        model: Final = _model(scenario, provider)
        body: Final = _chat_body(model, prompt, ("shield",))
        raw: Final = json.dumps(body)
        raw_with_duplicate: Final = (
            raw[:-1] + ',"messages":' + json.dumps(body["messages"]) + "}"
        )
        response: Final = candidate.client.post(
            "/v1/chat/completions",
            content=raw_with_duplicate,
            headers={"Authorization": f"Bearer {candidate.key}", "Content-Type": "application/json"},
        )
        assert response.status_code == 200, f"{response.status_code}: {response.text}"
        assert _azure_texts(azure) == (prompt,), response.text
        assert provider_texts(provider) == (prompt,), response.text


@pytest.mark.parametrize("authorization", ["", "Bearer invalid-key"], ids=("S3-missing", "S3-invalid"))
def test_s3_authentication_rejects_before_guardrails(
    azure_rig: tuple[Gateway, Wire, Wire], authorization: str
) -> None:
    candidate, azure, provider = azure_rig
    with candidate.scenario() as scenario:
        model: Final = _model(scenario, provider)
        response: Final = candidate.request(
            "POST",
            "/v1/chat/completions",
            _chat_body(model, f"auth {ATTACK_MARKER}", ("tuple-writer", "shield")),
            headers={"Authorization": authorization},
        )
        assert response.status_code == 401, response.text
        assert _azure_texts(azure) == (), response.text
        assert provider.drain() == (), response.text


@pytest.mark.parametrize("status", [500, 403, 404], ids=("S4-500", "S4-403", "S4-404"))
@pytest.mark.parametrize("guardrail_name", ["shield", "moderation"], ids=("S4-shield", "S4-moderation"))
def test_s4_azure_error_fails_closed(
    azure_rig: tuple[Gateway, Wire, Wire], status: int, guardrail_name: str
) -> None:
    candidate, azure, provider = azure_rig
    prompt: Final = f"AZURE_{status} benign {uuid.uuid4().hex}"
    with candidate.scenario() as scenario:
        model: Final = _model(scenario, provider)
        response: Final = candidate.request(
            "POST",
            "/v1/chat/completions",
            _chat_body(model, prompt, ("tuple-writer", guardrail_name)),
        )
        assert response.status_code != 200, response.text
        assert "synthetic Azure error" in response.text, response.text
        assert _azure_texts(azure) == (prompt,), response.text
        assert provider.drain() == (), response.text


@pytest.mark.parametrize("guardrail_name", ["shield", "moderation"], ids=("S5-shield", "S5-moderation"))
def test_s5_list_guardrail_azure_error_control(
    azure_rig: tuple[Gateway, Wire, Wire], guardrail_name: str
) -> None:
    candidate, azure, provider = azure_rig
    prompt: Final = f"AZURE_500 list benign {uuid.uuid4().hex}"
    with candidate.scenario() as scenario:
        model: Final = _model(scenario, provider)
        response: Final = candidate.request(
            "POST",
            "/v1/chat/completions",
            _chat_body(model, prompt, (guardrail_name,)),
        )
        assert response.status_code != 200, response.text
        assert "synthetic Azure error" in response.text, response.text
        assert _azure_texts(azure) == (prompt,), response.text
        assert provider.drain() == (), response.text


@pytest.mark.parametrize(
    ("marker", "expected_status", "expected_message"),
    [
        (PROVIDER_401_MARKER, 401, "synthetic provider unauthorized"),
        (OVERSIZED_MARKER, 400, "synthetic context length exceeded"),
    ],
    ids=("S6-provider-401", "S6-oversized"),
)
def test_s6_provider_errors_follow_azure_scan(
    azure_rig: tuple[Gateway, Wire, Wire], marker: str, expected_status: int, expected_message: str
) -> None:
    candidate, azure, provider = azure_rig
    prompt: Final = f"provider error {marker} {uuid.uuid4().hex}"
    with candidate.scenario() as scenario:
        model: Final = _model(scenario, provider)
        response: Final = candidate.request(
            "POST",
            "/v1/chat/completions",
            _chat_body(model, prompt, ("tuple-writer", "shield")),
        )
        assert response.status_code == expected_status, response.text
        assert expected_message in response.text, response.text
        assert _azure_texts(azure) == (prompt,), response.text
        assert provider_texts(provider) == (prompt,), response.text


def test_s6_unknown_model_matches_base_pin(azure_rig: tuple[Gateway, Wire, Wire]) -> None:
    candidate, azure, provider = azure_rig
    prompt: Final = f"unknown model benign {uuid.uuid4().hex}"
    model: Final = "openai/unknown-audit-model"
    response: Final = candidate.request(
        "POST",
        "/v1/chat/completions",
        _chat_body(model, prompt, ("tuple-writer", "shield")),
    )
    assert response.status_code == 400, f"{response.status_code}: {response.text}"
    assert "Invalid model name" in response.text, response.text
    scanned: Final = _azure_texts(azure)
    assert scanned == (prompt,), f"{response.text}: {scanned!r}"
    assert provider_texts(provider) == (), response.text


@pytest.mark.parametrize(
    ("guardrail_name", "marker"),
    [("shield", "last user benign"), ("moderation", "last user benign")],
    ids=("S7-shield-benign", "S7-moderation-benign"),
)
def test_s7_tuple_multi_item_text_scans_exact_last_user_block(
    azure_rig: tuple[Gateway, Wire, Wire], guardrail_name: str, marker: str
) -> None:
    candidate, azure, provider = azure_rig
    prompt: Final = f"{marker} {uuid.uuid4().hex}"
    expected: Final = "part one " + prompt
    messages: Final = [
        {"role": "system", "content": "system context"},
        {"role": "user", "content": "earlier user"},
        {"role": "assistant", "content": "assistant response"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "part one "},
                {"type": "text", "text": prompt},
            ],
        },
    ]
    with candidate.scenario() as scenario:
        model: Final = _model(scenario, provider)
        response: Final = candidate.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": model,
                "messages": messages,
                "guardrails": ["tuple-writer", guardrail_name],
            },
        )
        assert response.status_code == 200, response.text
        assert _azure_texts(azure) == (expected,), response.text
        assert provider_messages(provider) == (messages,), response.text


@pytest.mark.parametrize(
    ("guardrail_name", "marker"),
    [("shield", ATTACK_MARKER), ("moderation", MODERATION_MARKER)],
    ids=("S7-shield-attack", "S7-moderation-attack"),
)
def test_s7_tuple_multi_item_attack_is_blocked(
    azure_rig: tuple[Gateway, Wire, Wire], guardrail_name: str, marker: str
) -> None:
    candidate, azure, provider = azure_rig
    prompt: Final = f"multi part {marker} {uuid.uuid4().hex}"
    expected: Final = "part one " + prompt
    with candidate.scenario() as scenario:
        model: Final = _model(scenario, provider)
        response: Final = candidate.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [
                    {"role": "system", "content": "system context"},
                    {"role": "user", "content": "earlier benign"},
                    {"role": "assistant", "content": "assistant response"},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "part one "},
                            {"type": "text", "text": prompt},
                        ],
                    },
                ],
                "guardrails": ["tuple-writer", guardrail_name],
            },
        )
        assert response.status_code == 400, response.text
        assert _azure_texts(azure) == (expected,), response.text
        assert provider.drain() == (), response.text


def test_s8_unknown_guardrail_matches_base_pin(azure_rig: tuple[Gateway, Wire, Wire]) -> None:
    candidate, azure, provider = azure_rig
    with candidate.scenario() as scenario:
        model: Final = _model(scenario, provider)
        response: Final = candidate.request(
            "POST",
            "/v1/chat/completions",
            _chat_body(model, "unknown guardrail control", ("unknown-guardrail",)),
        )
        assert response.status_code == 200, f"{response.status_code}: {response.text}"
        assert _azure_texts(azure) == (), response.text
        assert provider_texts(provider) == ("unknown guardrail control",), response.text


def test_s9_guardrails_list_contains_azure_guardrails(azure_rig: tuple[Gateway, Wire, Wire]) -> None:
    candidate, azure, provider = azure_rig
    response: Final = candidate.request("GET", "/guardrails/list")
    assert response.status_code == 200, response.text
    assert "shield" in response.text and "moderation" in response.text, response.text
    assert _azure_texts(azure) == (), response.text
    assert provider.drain() == (), response.text


def test_s10_malformed_request_does_not_poison_proxy(azure_rig: tuple[Gateway, Wire, Wire]) -> None:
    candidate, azure, provider = azure_rig
    with candidate.scenario() as scenario:
        model: Final = _model(scenario, provider)
        malformed: Final = candidate.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": 5, "guardrails": ["shield"]},
        )
        assert malformed.status_code == 500, malformed.text
        assert _azure_texts(azure) == (), malformed.text
        assert provider.drain() == (), malformed.text
        malformed_shape: Final = candidate.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": "malformed messages", "guardrails": ["shield"]},
        )
        assert malformed_shape.status_code == 400, malformed_shape.text
        assert _azure_texts(azure) == (), malformed_shape.text
        provider.drain()
        prompt: Final = f"post malformed benign {uuid.uuid4().hex}"
        valid: Final = candidate.request(
            "POST",
            "/v1/chat/completions",
            _chat_body(model, prompt, ("shield",)),
        )
        health: Final = candidate.request("GET", "/health/liveliness")
        assert valid.status_code == 200, valid.text
        assert health.status_code == 200, health.text
        assert _azure_texts(azure) == (prompt,), valid.text
        assert provider_texts(provider) == (prompt,), valid.text


@pytest.mark.parametrize("guardrail_name", ["shield", "moderation"], ids=("E1-shield", "E1-moderation"))
def test_e1_empty_messages_matches_base_guardrail_response(
    azure_rig: tuple[Gateway, Wire, Wire], guardrail_name: str
) -> None:
    candidate, azure, provider = azure_rig
    with candidate.scenario() as scenario:
        model: Final = _model(scenario, provider, model="openai/text-embedding-3-small")
        response: Final = candidate.request(
            "POST",
            "/v1/embeddings",
            {
                "model": model,
                "input": f"empty messages {uuid.uuid4().hex}",
                "messages": [],
                "guardrails": [guardrail_name],
            },
        )
        assert response.status_code == 200, response.text
        assert _azure_texts(azure) == (), response.text
        entry: Final = _guardrail_entry(response)
        assert entry["guardrail_response"] == {}, f"{response.text}: {entry!r}"
        assert len(provider.drain()) == 1, response.text


def test_e2_key_and_request_guardrail_precedence_matches_base_pin(
    azure_rig: tuple[Gateway, Wire, Wire],
) -> None:
    candidate, azure, provider = azure_rig
    prompt: Final = f"precedence {ATTACK_MARKER} {uuid.uuid4().hex}"
    with candidate.scenario() as scenario:
        key: Final = scenario.key(guardrails=["shield"])
        model: Final = _model(scenario, provider)
        response: Final = candidate.request(
            "POST",
            "/v1/chat/completions",
            _chat_body(model, prompt, ("tuple-writer",)),
            key=key,
        )
        assert response.status_code == 400, f"{response.status_code}: {response.text}"
        assert _azure_texts(azure) == (prompt,), response.text
        assert provider.drain() == (), response.text


def test_e3_key_update_applies_guardrails_during_traffic(
    key_update_rig: tuple[Gateway, Wire, Wire, AzureBehavior],
) -> None:
    candidate, azure, provider, behavior = key_update_rig
    with candidate.scenario() as scenario:
        key: Final = scenario.key()
        model: Final = _model(scenario, provider)
        prompt: Final = f"key update {ATTACK_MARKER} {uuid.uuid4().hex}"
        before: Final = candidate.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "cache": {"no-cache": True},
            },
            key=key,
        )
        assert before.status_code == 200, before.text
        assert _azure_texts(azure) == (), before.text
        assert provider_texts(provider) == (prompt,), before.text
        active_prompt: Final = f"E3_BLOCK benign {uuid.uuid4().hex}"
        entered: Final = behavior.entered
        release: Final = behavior.release
        assert entered is not None and release is not None, "E3 barrier events were not configured"
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            active_request: Final = executor.submit(
                candidate.request,
                "POST",
                "/v1/chat/completions",
                _chat_body(model, active_prompt, ("shield",), no_cache=True),
                key=key,
            )
            try:
                assert entered.wait(timeout=30), "Concurrent request did not reach the Azure responder"
                updated: Final = candidate.request(
                    "POST",
                    "/key/update",
                    {"key": key, "guardrails": ["tuple-writer", "shield"]},
                )
                assert updated.status_code == 200, updated.text
            finally:
                release.set()
            active_response: Final = active_request.result(timeout=60)
        assert active_response.status_code == 200, active_response.text
        assert _azure_texts(azure) == (active_prompt,), active_response.text
        assert provider_texts(provider) == (active_prompt,), active_response.text
        attacked: Final = f"after key update {ATTACK_MARKER} {uuid.uuid4().hex}"
        after: Final = candidate.request(
            "POST",
            "/v1/chat/completions",
            _chat_body(model, attacked, (), no_cache=True),
            key=key,
        )
        assert after.status_code == 400, after.text
        assert _azure_texts(azure) == (attacked,), after.text
        assert provider.drain() == (), after.text


@pytest.mark.parametrize("guardrail_name", ["shield", "moderation"], ids=("E4-shield", "E4-moderation"))
def test_e4_cache_disabled_twin_requests_have_distinct_spend_rows(
    azure_rig: tuple[Gateway, Wire, Wire], guardrail_name: str
) -> None:
    candidate, azure, provider = azure_rig
    prompt: Final = f"cache disabled {uuid.uuid4().hex}"
    with candidate.scenario() as scenario:
        model: Final = _model(scenario, provider)
        responses: Final = tuple(
            candidate.request(
                "POST",
                "/v1/chat/completions",
                _chat_body(model, prompt, ("tuple-writer", guardrail_name), no_cache=True),
            )
            for _ in range(3)
        )
        assert all(response.status_code == 200 for response in responses), tuple(
            response.text for response in responses
        )
        assert len(set(response.headers["x-litellm-call-id"] for response in responses)) == 3
        assert _azure_texts(azure) == (prompt, prompt, prompt), responses[-1].text
        assert provider_texts(provider) == (prompt, prompt, prompt), responses[-1].text
        assert all(_guardrail_entry(response)["guardrail_status"] == "success" for response in responses), (
            responses[-1].text
        )


@pytest.mark.parametrize("async_client", [False, True], ids=("sync", "async"))
def test_h15_openai_sdk_tuple_attack_is_blocked(
    azure_rig: tuple[Gateway, Wire, Wire], async_client: bool
) -> None:
    candidate, azure, provider = azure_rig
    prompt: Final = f"OpenAI SDK tuple {ATTACK_MARKER} {uuid.uuid4().hex}"
    with candidate.scenario() as scenario:
        model: Final = _model(scenario, provider)
        base_url: Final = str(candidate.client.base_url).rstrip("/") + "/v1"
        if async_client:
            with pytest.raises(openai.BadRequestError, match="Azure Prompt Shield"):
                asyncio.run(_openai_chat_async(base_url, candidate.key, model, prompt, ("tuple-writer", "shield")))
        else:
            with pytest.raises(openai.BadRequestError, match="Azure Prompt Shield"):
                _openai_chat_sync(base_url, candidate.key, model, prompt, ("tuple-writer", "shield"))
        assert _azure_texts(azure) == (prompt,), "SDK request did not reach Azure with the expected prompt"
        assert provider.drain() == (), "Blocked SDK request reached the provider"


@pytest.mark.parametrize("async_client", [False, True], ids=("sync", "async"))
def test_h15_openai_sdk_tuple_benign_is_scanned_and_served(
    azure_rig: tuple[Gateway, Wire, Wire], async_client: bool
) -> None:
    candidate, azure, provider = azure_rig
    prompt: Final = f"OpenAI SDK benign {uuid.uuid4().hex}"
    with candidate.scenario() as scenario:
        model: Final = _model(scenario, provider)
        base_url: Final = str(candidate.client.base_url).rstrip("/") + "/v1"
        if async_client:
            asyncio.run(_openai_chat_async(base_url, candidate.key, model, prompt, ("tuple-writer", "shield")))
        else:
            _openai_chat_sync(base_url, candidate.key, model, prompt, ("tuple-writer", "shield"))
        assert _azure_texts(azure) == (prompt,), "SDK request did not reach Azure with the expected prompt"
        assert provider_texts(provider) == (prompt,), "SDK request did not reach the provider with the expected prompt"


@pytest.mark.parametrize("async_client", [False, True], ids=("sync", "async"))
def test_h16_anthropic_sdk_tuple_attack_is_blocked(
    azure_rig: tuple[Gateway, Wire, Wire], async_client: bool
) -> None:
    candidate, azure, provider = azure_rig
    prompt: Final = f"Anthropic SDK tuple {ATTACK_MARKER} {uuid.uuid4().hex}"
    with candidate.scenario() as scenario:
        model: Final = scenario.model(
            model="anthropic/claude-sonnet-4-5-20250929",
            api_base=provider.url,
            api_key="synthetic-provider-key",
        )
        base_url: Final = str(candidate.client.base_url).rstrip("/")
        if async_client:
            with pytest.raises(anthropic.BadRequestError, match="Azure Prompt Shield"):
                asyncio.run(_anthropic_messages_async(base_url, candidate.key, model, prompt))
        else:
            with pytest.raises(anthropic.BadRequestError, match="Azure Prompt Shield"):
                _anthropic_messages_sync(base_url, candidate.key, model, prompt)
        assert _azure_texts(azure) == (prompt,), "SDK request did not reach Azure with the expected prompt"
        assert provider.drain() == (), "Blocked SDK request reached the provider"


@pytest.mark.parametrize("async_client", [False, True], ids=("sync", "async"))
def test_h17_openai_sdk_responses_control_stays_blocked(
    azure_rig: tuple[Gateway, Wire, Wire], async_client: bool
) -> None:
    candidate, azure, provider = azure_rig
    prompt: Final = f"OpenAI Responses {ATTACK_MARKER} {uuid.uuid4().hex}"
    with candidate.scenario() as scenario:
        model: Final = _model(scenario, provider)
        base_url: Final = str(candidate.client.base_url).rstrip("/") + "/v1"
        if async_client:
            with pytest.raises(openai.BadRequestError, match="Azure Prompt Shield"):
                asyncio.run(_openai_responses_async(base_url, candidate.key, model, prompt))
        else:
            with pytest.raises(openai.BadRequestError, match="Azure Prompt Shield"):
                _openai_responses_sync(base_url, candidate.key, model, prompt)
        assert _azure_texts(azure) == (prompt,), "SDK request did not reach Azure with the expected prompt"
        assert provider.drain() == (), "Blocked SDK request reached the provider"


@pytest.mark.parametrize("guardrail_source", ["key", "team"], ids=("H18-key", "H19-team"))
def test_h18_h19_key_and_team_tuple_guardrails_are_applied(
    azure_rig: tuple[Gateway, Wire, Wire], guardrail_source: str
) -> None:
    candidate, azure, provider = azure_rig
    prompt: Final = f"{guardrail_source} metadata {ATTACK_MARKER} {uuid.uuid4().hex}"
    with candidate.scenario() as scenario:
        team_id: Final = scenario.team(guardrails=["tuple-writer", "shield"]) if guardrail_source == "team" else ""
        key_fields: Final = (
            {"team_id": team_id}
            if guardrail_source == "team"
            else {"guardrails": ["tuple-writer", "shield"]}
        )
        key: Final = scenario.key(**key_fields)
        model: Final = _model(scenario, provider)
        response: Final = candidate.request(
            "POST",
            "/v1/chat/completions",
            _chat_body(model, prompt, ()),
            key=key,
        )
        assert response.status_code == 400, response.text
        assert _azure_texts(azure) == (prompt,), response.text
        assert provider.drain() == (), response.text


def _azure_texts(azure: Wire) -> tuple[str, ...]:
    return azure_texts(azure)
