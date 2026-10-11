import json
import threading
import uuid
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path
from typing import Final

import psutil
import pytest
import yaml
from integration._support.client import Gateway, eventually, gateway_from_environment, object_value
from integration._support.database import read_rows
from integration._support.process import OwnedProxy, owned_proxy_process
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue

_ATTACK_MARKER: Final = "synthetic-attack-marker"
_MODERATION_MARKER: Final = "synthetic-moderation-marker"

_SHIELD_TARGET_PREFIX: Final = "/contentsafety/text:shieldPrompt?api-version="
_ANALYZE_TARGET_PREFIX: Final = "/contentsafety/text:analyze?api-version="

_OPT_IN_SHIELD: Final = "audit-shield-optin"
_TEXT_MODERATION: Final = "audit-text-mod"


def _chat_frame(identity: str, delta: dict[str, JsonValue], finish: str | None = None) -> bytes:
    return (
        b"data: "
        + json.dumps(
            {
                "id": identity,
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "gpt-4o-mini",
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            }
        ).encode()
        + b"\n\n"
    )


def _chat_stream_chunks() -> tuple[bytes, ...]:
    identity: Final = "chatcmpl-" + uuid.uuid4().hex
    usage: Final = {
        "id": identity,
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "gpt-4o-mini",
        "choices": [],
        "usage": {"prompt_tokens": 11, "completion_tokens": 4, "total_tokens": 15},
    }
    return (
        _chat_frame(identity, {"role": "assistant", "content": "permitted "}),
        _chat_frame(identity, {"content": "response"}, finish="stop"),
        b"data: " + json.dumps(usage).encode() + b"\n\n",
        b"data: [DONE]\n\n",
    )


def _provider(request: Request) -> Reply:
    if request.method != "POST":
        return Reply(body=b'{"object":"list","data":[]}')
    parsed: Final = object_value(json.loads(request.body)) if request.body else {}
    if request.target == "/v1/messages":
        return Reply(
            body=json.dumps(
                {
                    "id": "msg_" + uuid.uuid4().hex,
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-sonnet-4-5-20250929",
                    "content": [{"type": "text", "text": "permitted response"}],
                    "stop_reason": "end_turn",
                    "stop_sequence": None,
                    "usage": {"input_tokens": 11, "output_tokens": 4},
                }
            ).encode()
        )
    if request.target == "/v1/responses":
        return Reply(
            body=json.dumps(
                {
                    "id": "resp_" + uuid.uuid4().hex,
                    "object": "response",
                    "created_at": 1700000000,
                    "status": "completed",
                    "model": "gpt-4.1-mini",
                    "output": [
                        {
                            "type": "message",
                            "id": "msg_" + uuid.uuid4().hex,
                            "status": "completed",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": "permitted response", "annotations": []}],
                        }
                    ],
                    "usage": {"input_tokens": 11, "output_tokens": 4, "total_tokens": 15},
                }
            ).encode()
        )
    assert request.target == "/v1/chat/completions", request.target
    if parsed.get("stream") is True:
        return Reply(content_type="text/event-stream", chunks=_chat_stream_chunks())
    return Reply(
        body=json.dumps(
            {
                "id": "chatcmpl-" + uuid.uuid4().hex,
                "object": "chat.completion",
                "created": 1700000000,
                "model": "gpt-4.1-mini",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "permitted response"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 11, "completion_tokens": 4, "total_tokens": 15},
            }
        ).encode()
    )


def _azure(outage: threading.Event) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        if request.method != "POST":
            return Reply(status=404)
        if outage.is_set():
            return Reply(status=503)
        body: Final = object_value(json.loads(request.body))
        if request.target.startswith(_SHIELD_TARGET_PREFIX):
            user_prompt: Final = body["userPrompt"]
            assert isinstance(user_prompt, str)
            return Reply(
                body=json.dumps(
                    {
                        "userPromptAnalysis": {"attackDetected": _ATTACK_MARKER in user_prompt},
                        "documentsAnalysis": [],
                    }
                ).encode()
            )
        assert request.target.startswith(_ANALYZE_TARGET_PREFIX), request.target
        text: Final = body["text"]
        assert isinstance(text, str)
        severity: Final = 4 if _MODERATION_MARKER in text else 0
        return Reply(
            body=json.dumps(
                {
                    "blocklistsMatch": [],
                    "categoriesAnalysis": [
                        {"category": "Hate", "severity": severity},
                        {"category": "Sexual", "severity": 0},
                        {"category": "SelfHarm", "severity": 0},
                        {"category": "Violence", "severity": 0},
                    ],
                }
            ).encode()
        )

    return respond


def _config(directory: Path, azure: Wire, guardrails: list[dict[str, JsonValue]]) -> Path:
    config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    config["guardrails"] = guardrails
    path: Final = directory / "azure-audit.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def _shield_params(azure: Wire, *, mode: str, default_on: bool) -> dict[str, JsonValue]:
    return {
        "guardrail": "azure/prompt_shield",
        "mode": mode,
        "default_on": default_on,
        "api_base": azure.url,
        "api_key": "synthetic-azure-key",
        "cost_tier": "paid",
        "price_per_1000_text_records": 0.38,
    }


@pytest.fixture(scope="module")
def audit_rig(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[tuple[OwnedProxy, Wire, Wire, threading.Event]]:
    directory: Final = tmp_path_factory.mktemp("azure-audit")
    outage: Final = threading.Event()
    with ExitStack() as stack:
        gateway: Final = stack.enter_context(gateway_from_environment())
        azure: Final = stack.enter_context(wire_server(_azure(outage)))
        provider: Final = stack.enter_context(wire_server(_provider))
        config: Final = _config(
            directory,
            azure,
            [
                {
                    "guardrail_name": "audit-shield",
                    "litellm_params": _shield_params(azure, mode="pre_call", default_on=True),
                },
                {
                    "guardrail_name": _TEXT_MODERATION,
                    "litellm_params": {
                        "guardrail": "azure/text_moderations",
                        "mode": "pre_call",
                        "default_on": False,
                        "api_base": azure.url,
                        "api_key": "synthetic-azure-key",
                    },
                },
            ],
        )
        owned: Final = stack.enter_context(owned_proxy_process(gateway, directory, {}, config=config, workers=2))
        yield owned, azure, provider, outage


@pytest.fixture(scope="module")
def optin_rig(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[tuple[Gateway, Wire, Wire]]:
    directory: Final = tmp_path_factory.mktemp("azure-optin")
    with ExitStack() as stack:
        gateway: Final = stack.enter_context(gateway_from_environment())
        azure: Final = stack.enter_context(wire_server(_azure(threading.Event())))
        provider: Final = stack.enter_context(wire_server(_provider))
        config: Final = _config(
            directory,
            azure,
            [
                {
                    "guardrail_name": _OPT_IN_SHIELD,
                    "litellm_params": _shield_params(azure, mode="pre_call", default_on=False),
                }
            ],
        )
        yield (
            stack.enter_context(owned_proxy_process(gateway, directory, {}, config=config, workers=2)).gateway,
            azure,
            provider,
        )


@pytest.fixture(scope="module")
def chaos_rig(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[tuple[OwnedProxy, Wire, Wire, threading.Event]]:
    directory: Final = tmp_path_factory.mktemp("azure-chaos")
    outage: Final = threading.Event()
    with ExitStack() as stack:
        gateway: Final = stack.enter_context(gateway_from_environment())
        azure: Final = stack.enter_context(wire_server(_azure(outage)))
        provider: Final = stack.enter_context(wire_server(_provider))
        config: Final = _config(
            directory,
            azure,
            [
                {
                    "guardrail_name": "audit-shield",
                    "litellm_params": _shield_params(azure, mode="pre_call", default_on=True),
                },
                {
                    "guardrail_name": _TEXT_MODERATION,
                    "litellm_params": {
                        "guardrail": "azure/text_moderations",
                        "mode": "pre_call",
                        "default_on": False,
                        "api_base": azure.url,
                        "api_key": "synthetic-azure-key",
                    },
                },
            ],
        )
        owned: Final = stack.enter_context(owned_proxy_process(gateway, directory, {}, config=config, workers=2))
        yield owned, azure, provider, outage


@pytest.fixture(scope="module")
def during_rig(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[tuple[Gateway, Wire, Wire]]:
    directory: Final = tmp_path_factory.mktemp("azure-during")
    with ExitStack() as stack:
        gateway: Final = stack.enter_context(gateway_from_environment())
        azure: Final = stack.enter_context(wire_server(_azure(threading.Event())))
        provider: Final = stack.enter_context(wire_server(_provider))
        config: Final = _config(
            directory,
            azure,
            [
                {
                    "guardrail_name": "audit-shield-during",
                    "litellm_params": _shield_params(azure, mode="during_call", default_on=True),
                }
            ],
        )
        yield (
            stack.enter_context(owned_proxy_process(gateway, directory, {}, config=config, workers=2)).gateway,
            azure,
            provider,
        )


@pytest.fixture(autouse=True)
def _clear_wires(request: pytest.FixtureRequest) -> None:
    for name in ("audit_rig", "optin_rig", "during_rig", "chaos_rig"):
        if name in request.fixturenames:
            rig: Final = request.getfixturevalue(name)
            rig[1].drain()
            rig[2].drain()


def _shield_prompts(requests: tuple[Request, ...]) -> tuple[JsonValue, ...]:
    return tuple(
        object_value(json.loads(scan.body))["userPrompt"]
        for scan in requests
        if scan.target.startswith(_SHIELD_TARGET_PREFIX)
    )


def _analyze_texts(requests: tuple[Request, ...]) -> tuple[JsonValue, ...]:
    return tuple(
        object_value(json.loads(scan.body))["text"]
        for scan in requests
        if scan.target.startswith(_ANALYZE_TARGET_PREFIX)
    )


def _guardrail_entries(model: str, count: int = 1) -> list[JsonValue]:
    rows: Final = eventually(
        lambda: read_rows('SELECT metadata FROM "LiteLLM_SpendLogs" WHERE model_group=%s', (model,)),
        lambda values: len(values) == 1,
        seconds=70,
    )
    saved: Final = object_value(rows[0]["metadata"])
    entries: Final = saved["guardrail_information"]
    assert isinstance(entries, list) and len(entries) == count, saved
    return entries


def _provider_calls(provider: Wire) -> tuple[Request, ...]:
    return tuple(call for call in provider.drain() if call.method == "POST")


def _entries_by_request_id(request_id: str) -> list[JsonValue]:
    rows: Final = eventually(
        lambda: read_rows('SELECT metadata FROM "LiteLLM_SpendLogs" WHERE request_id=%s', (request_id,)),
        lambda values: len(values) == 1,
        seconds=70,
    )
    saved: Final = object_value(rows[0]["metadata"])
    entries: Final = saved["guardrail_information"]
    assert isinstance(entries, list) and len(entries) == 1, saved
    return entries


@pytest.mark.parametrize("missing_messages", [{"messages": None}, {}], ids=["null-messages", "absent-messages"])
def test_responses_input_scanned_without_a_messages_list(
    audit_rig: tuple[OwnedProxy, Wire, Wire, threading.Event], missing_messages: dict[str, JsonValue]
) -> None:
    owned, azure, provider, _ = audit_rig
    prompt: Final = "synthetic prompt no-messages " + uuid.uuid4().hex
    with owned.gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="openai/gpt-4.1-mini", api_base=provider.url + "/v1", api_key="synthetic-provider-key"
        )
        response: Final = owned.gateway.request(
            "POST", "/v1/responses", {"model": model, "input": prompt, **missing_messages}
        )
        assert response.status_code == 200, response.text
        assert _shield_prompts(azure.drain()) == (prompt,)
        assert len(_provider_calls(provider)) == 1
        entry: Final = object_value(_guardrail_entries(model)[0])
        assert entry["guardrail_usage"] == {"requests": 1, "input_characters": len(prompt), "text_records": 1}, entry
        assert entry["guardrail_cost"] == pytest.approx(0.38 / 1000), entry


def test_responses_streaming_input_is_scanned_and_billed(
    audit_rig: tuple[OwnedProxy, Wire, Wire, threading.Event],
) -> None:
    owned, azure, provider, _ = audit_rig
    prompt: Final = "synthetic prompt streaming " + uuid.uuid4().hex
    with owned.gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="deepseek/gpt-4o-mini", api_base=provider.url + "/v1", api_key="synthetic-provider-key"
        )
        with owned.gateway.client.stream(
            "POST",
            "/v1/responses",
            json={"model": model, "input": prompt, "stream": True},
            headers={"Authorization": f"Bearer {owned.gateway.key}"},
        ) as response:
            text: Final = response.read().decode()
        assert response.status_code == 200, text
        assert response.headers["content-type"].startswith("text/event-stream"), text
        assert _shield_prompts(azure.drain()) == (prompt,)
        assert len(_provider_calls(provider)) == 1
        entry: Final = object_value(_guardrail_entries(model)[0])
        assert entry["guardrail_usage"] == {"requests": 1, "input_characters": len(prompt), "text_records": 1}, entry
        assert entry["guardrail_cost"] == pytest.approx(0.38 / 1000), entry


@pytest.mark.parametrize(
    "body",
    [
        pytest.param(lambda prompt: {"input": prompt}, id="string-input"),
        pytest.param(
            lambda prompt: {"input": [{"role": "user", "content": [{"type": "input_text", "text": prompt}]}]},
            id="list-input",
        ),
        pytest.param(lambda prompt: {"messages": [], "input": prompt}, id="empty-messages-stub"),
    ],
)
def test_text_moderation_opt_in_scans_responses_input(
    audit_rig: tuple[OwnedProxy, Wire, Wire, threading.Event],
    request: pytest.FixtureRequest,
    body: Callable[[str], dict[str, JsonValue]],
) -> None:
    owned, azure, provider, _ = audit_rig
    prompt: Final = f"synthetic benign prompt {request.node.callspec.id} {uuid.uuid4().hex}"
    with owned.gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="openai/gpt-4.1-mini", api_base=provider.url + "/v1", api_key="synthetic-provider-key"
        )
        response: Final = owned.gateway.request(
            "POST", "/v1/responses", {"model": model, "guardrails": [_TEXT_MODERATION], **body(prompt)}
        )
        assert response.status_code == 200, response.text
        calls: Final = azure.drain()
        assert _analyze_texts(calls) == (prompt,)
        assert _shield_prompts(calls) == (prompt,)
        assert len(_provider_calls(provider)) == 1
        entries: Final = _guardrail_entries(model, count=2)
        assert {object_value(entry)["guardrail_name"] for entry in entries} == {"audit-shield", _TEXT_MODERATION}, (
            entries
        )


def test_text_moderation_opt_in_scans_chat_messages(audit_rig: tuple[OwnedProxy, Wire, Wire, threading.Event]) -> None:
    owned, azure, provider, _ = audit_rig
    prompt: Final = "synthetic benign prompt chat-optin " + uuid.uuid4().hex
    with owned.gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="openai/gpt-4.1-mini", api_base=provider.url + "/v1", api_key="synthetic-provider-key"
        )
        response: Final = owned.gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "guardrails": [_TEXT_MODERATION], "messages": [{"role": "user", "content": prompt}]},
        )
        assert response.status_code == 200, response.text
        calls: Final = azure.drain()
        assert _analyze_texts(calls) == (prompt,)
        assert _shield_prompts(calls) == (prompt,)


def test_chat_with_input_key_still_scans_messages_only(
    audit_rig: tuple[OwnedProxy, Wire, Wire, threading.Event],
) -> None:
    owned, azure, provider, _ = audit_rig
    prompt: Final = "synthetic prompt chat-shadow " + uuid.uuid4().hex
    shadow: Final = "shadow input value " + uuid.uuid4().hex
    with owned.gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="openai/gpt-4.1-mini", api_base=provider.url + "/v1", api_key="synthetic-provider-key"
        )
        response: Final = owned.gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": prompt}], "input": shadow},
        )
        assert response.status_code == 200, response.text
        assert _shield_prompts(azure.drain()) == (prompt,)


def test_responses_multi_turn_input_scans_last_user_text_only(
    audit_rig: tuple[OwnedProxy, Wire, Wire, threading.Event],
) -> None:
    owned, azure, provider, _ = audit_rig
    last_user: Final = "synthetic prompt last-turn " + uuid.uuid4().hex
    with owned.gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="openai/gpt-4.1-mini", api_base=provider.url + "/v1", api_key="synthetic-provider-key"
        )
        response: Final = owned.gateway.request(
            "POST",
            "/v1/responses",
            {
                "model": model,
                "input": [
                    {"role": "user", "content": [{"type": "input_text", "text": "first question"}]},
                    {"role": "assistant", "content": [{"type": "output_text", "text": "an answer"}]},
                    {"role": "user", "content": [{"type": "input_text", "text": last_user}]},
                ],
            },
        )
        assert response.status_code == 200, response.text
        assert _shield_prompts(azure.drain()) == (last_user,)


def test_openai_sdk_responses_calls_are_scanned_and_billed(
    audit_rig: tuple[OwnedProxy, Wire, Wire, threading.Event],
) -> None:
    import asyncio

    from openai import AsyncOpenAI, OpenAI
    from openai.types.responses import Response

    owned, azure, provider, _ = audit_rig
    base_url: Final = f"http://127.0.0.1:{owned.gateway.client.base_url.port}/v1"
    sync_prompt: Final = "synthetic prompt sdk-sync " + uuid.uuid4().hex
    async_prompt: Final = "synthetic prompt sdk-async " + uuid.uuid4().hex
    with owned.gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="openai/gpt-4.1-mini", api_base=provider.url + "/v1", api_key="synthetic-provider-key"
        )
        sync_response: Final[Response] = OpenAI(base_url=base_url, api_key=owned.gateway.key).responses.create(
            model=model, input=sync_prompt
        )
        assert sync_response.status == "completed"

        async def create_async() -> Response:
            return await AsyncOpenAI(base_url=base_url, api_key=owned.gateway.key).responses.create(
                model=model, input=async_prompt
            )

        async_response: Final[Response] = asyncio.run(create_async())
        assert async_response.status == "completed"
        assert _shield_prompts(azure.drain()) == (sync_prompt, async_prompt)
        assert len(_provider_calls(provider)) == 2
        for response_id in (sync_response.id, async_response.id):
            entry: Final = object_value(_entries_by_request_id(response_id)[0])
            assert entry["guardrail_usage"]["requests"] == 1, entry
            assert entry["guardrail_cost"] == pytest.approx(0.38 / 1000), entry


@pytest.mark.parametrize(
    ("bad_input", "expected_status", "max_provider_calls"),
    [
        pytest.param(123, 500, 0, id="int-input"),
        pytest.param({"a": 1}, 200, 1, id="dict-input"),
        pytest.param("", 200, 1, id="empty-string-input"),
    ],
)
def test_unscannable_responses_input_matches_base_behavior(
    audit_rig: tuple[OwnedProxy, Wire, Wire, threading.Event],
    bad_input: JsonValue,
    expected_status: int,
    max_provider_calls: int,
) -> None:
    owned, azure, provider, _ = audit_rig
    with owned.gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="openai/gpt-4.1-mini", api_base=provider.url + "/v1", api_key="synthetic-provider-key"
        )
        response: Final = owned.gateway.request(
            "POST",
            "/v1/responses",
            {"model": model, "input": bad_input, "metadata": {"cell": uuid.uuid4().hex}},
        )
        assert response.status_code == expected_status, response.text
        assert _shield_prompts(azure.drain()) == ()
        assert len(_provider_calls(provider)) <= max_provider_calls


def test_long_responses_input_is_chunked_and_billed(audit_rig: tuple[OwnedProxy, Wire, Wire, threading.Event]) -> None:
    owned, azure, provider, _ = audit_rig
    prompt: Final = "synthetic " + ("x" * 5000) + " " + uuid.uuid4().hex
    with owned.gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="openai/gpt-4.1-mini", api_base=provider.url + "/v1", api_key="synthetic-provider-key"
        )
        response: Final = owned.gateway.request("POST", "/v1/responses", {"model": model, "input": prompt})
        assert response.status_code == 200, response.text
        assert _shield_prompts(azure.drain()) == (prompt,)
        entry: Final = object_value(_guardrail_entries(model)[0])
        assert entry["guardrail_usage"] == {
            "requests": 1,
            "input_characters": len(prompt),
            "text_records": -(-len(prompt) // 1000),
        }, entry
        assert entry["guardrail_cost"] == pytest.approx(-(-len(prompt) // 1000) * 0.38 / 1000), entry


def test_multi_chunk_responses_input_bills_every_azure_request(
    audit_rig: tuple[OwnedProxy, Wire, Wire, threading.Event],
) -> None:
    owned, azure, provider, _ = audit_rig
    prompt: Final = "synthetic " + ("y " * 6400).strip() + " " + uuid.uuid4().hex
    expected_records: Final = sum(-(-len(chunk) // 1000) for chunk in _chunks(prompt))
    with owned.gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="openai/gpt-4.1-mini", api_base=provider.url + "/v1", api_key="synthetic-provider-key"
        )
        response: Final = owned.gateway.request("POST", "/v1/responses", {"model": model, "input": prompt})
        assert response.status_code == 200, response.text
        scans: Final = _shield_prompts(azure.drain())
        entry: Final = object_value(_guardrail_entries(model)[0])
        usage: Final = entry["guardrail_usage"]
        assert len(scans) == usage["requests"], entry
        assert usage["text_records"] == expected_records, entry
        assert usage["input_characters"] == len(prompt), entry


def _chunks(prompt: str) -> tuple[str, ...]:
    return (prompt[:10000], prompt[10000:])


def test_streaming_responses_attack_is_blocked_before_any_stream_bytes(
    audit_rig: tuple[OwnedProxy, Wire, Wire, threading.Event],
) -> None:
    owned, azure, provider, _ = audit_rig
    prompt: Final = f"synthetic prompt {_ATTACK_MARKER} " + uuid.uuid4().hex
    with owned.gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="deepseek/gpt-4o-mini", api_base=provider.url + "/v1", api_key="synthetic-provider-key"
        )
        with owned.gateway.client.stream(
            "POST",
            "/v1/responses",
            json={"model": model, "input": prompt, "stream": True},
            headers={"Authorization": f"Bearer {owned.gateway.key}"},
        ) as response:
            body: Final = response.read().decode()
        assert response.status_code == 400, body
        assert "Violated Azure Prompt Shield guardrail policy" in body, body
        assert _shield_prompts(azure.drain()) == (prompt,)
        assert _provider_calls(provider) == ()


def test_text_moderation_opt_in_blocks_responses_input_above_threshold(
    audit_rig: tuple[OwnedProxy, Wire, Wire, threading.Event],
) -> None:
    owned, azure, provider, _ = audit_rig
    prompt: Final = f"synthetic prompt {_MODERATION_MARKER} " + uuid.uuid4().hex
    with owned.gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="openai/gpt-4.1-mini", api_base=provider.url + "/v1", api_key="synthetic-provider-key"
        )
        response: Final = owned.gateway.request(
            "POST", "/v1/responses", {"model": model, "guardrails": [_TEXT_MODERATION], "input": prompt}
        )
        assert response.status_code == 400, response.text
        assert _analyze_texts(azure.drain()) == (prompt,)
        assert _provider_calls(provider) == ()


def test_text_moderation_opt_in_blocks_streamed_responses_input_above_threshold(
    audit_rig: tuple[OwnedProxy, Wire, Wire, threading.Event],
) -> None:
    owned, azure, provider, _ = audit_rig
    prompt: Final = f"synthetic prompt {_MODERATION_MARKER} " + uuid.uuid4().hex
    with owned.gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="openai/gpt-4.1-mini", api_base=provider.url + "/v1", api_key="synthetic-provider-key"
        )
        with owned.gateway.client.stream(
            "POST",
            "/v1/responses",
            json={"model": model, "guardrails": [_TEXT_MODERATION], "input": prompt, "stream": True},
            headers={"Authorization": f"Bearer {owned.gateway.key}"},
        ) as response:
            body: Final = response.read().decode()
        assert response.status_code == 400, body
        assert "Prompt Shield" not in body, body
        assert _analyze_texts(azure.drain()) == (prompt,)
        assert _provider_calls(provider) == ()


def test_azure_outage_produces_the_same_outcome_on_responses_and_chat(
    audit_rig: tuple[OwnedProxy, Wire, Wire, threading.Event],
) -> None:
    owned, azure, provider, outage = audit_rig
    with owned.gateway.scenario() as scenario:
        chat_model: Final = scenario.model(
            model="openai/gpt-4.1-mini", api_base=provider.url + "/v1", api_key="synthetic-provider-key"
        )
        responses_model: Final = scenario.model(
            model="openai/gpt-4.1-mini", api_base=provider.url + "/v1", api_key="synthetic-provider-key"
        )
        outage.set()
        try:
            chat_response: Final = owned.gateway.request(
                "POST",
                "/v1/chat/completions",
                {"model": chat_model, "messages": [{"role": "user", "content": "outage probe " + uuid.uuid4().hex}]},
            )
            responses_response: Final = owned.gateway.request(
                "POST", "/v1/responses", {"model": responses_model, "input": "outage probe " + uuid.uuid4().hex}
            )
        finally:
            outage.clear()
        assert chat_response.status_code == responses_response.status_code, (
            chat_response.status_code,
            chat_response.text,
            responses_response.status_code,
            responses_response.text,
        )
        assert len(_provider_calls(provider)) == (1 if chat_response.status_code == 200 else 0) * 2


def test_responses_without_auth_is_rejected_without_scanning(
    audit_rig: tuple[OwnedProxy, Wire, Wire, threading.Event],
) -> None:
    owned, azure, provider, _ = audit_rig
    response: Final = owned.gateway.request(
        "POST", "/v1/responses", {"model": "anything", "input": "probe"}, key="invalid-key"
    )
    assert response.status_code == 401, response.text
    assert _shield_prompts(azure.drain()) == ()
    assert _provider_calls(provider) == ()


def test_attack_in_an_earlier_turn_is_not_scanned(audit_rig: tuple[OwnedProxy, Wire, Wire, threading.Event]) -> None:
    owned, azure, provider, _ = audit_rig
    last_user: Final = "synthetic prompt benign-tail " + uuid.uuid4().hex
    with owned.gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="openai/gpt-4.1-mini", api_base=provider.url + "/v1", api_key="synthetic-provider-key"
        )
        response: Final = owned.gateway.request(
            "POST",
            "/v1/responses",
            {
                "model": model,
                "input": [
                    {"role": "user", "content": [{"type": "input_text", "text": _ATTACK_MARKER}]},
                    {"role": "assistant", "content": [{"type": "output_text", "text": "an answer"}]},
                    {"role": "user", "content": [{"type": "input_text", "text": last_user}]},
                ],
            },
        )
        assert response.status_code == 200, response.text
        assert _shield_prompts(azure.drain()) == (last_user,)


def test_repeated_responses_body_bills_each_call_once(
    audit_rig: tuple[OwnedProxy, Wire, Wire, threading.Event],
) -> None:
    owned, azure, provider, _ = audit_rig
    prompt: Final = "synthetic prompt repeat " + uuid.uuid4().hex
    with owned.gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="openai/gpt-4.1-mini", api_base=provider.url + "/v1", api_key="synthetic-provider-key"
        )
        for _ in range(2):
            response: Final = owned.gateway.request("POST", "/v1/responses", {"model": model, "input": prompt})
            assert response.status_code == 200, response.text
        assert _shield_prompts(azure.drain()) == (prompt, prompt)
        rows: Final = eventually(
            lambda: read_rows('SELECT metadata FROM "LiteLLM_SpendLogs" WHERE model_group=%s', (model,)),
            lambda values: len(values) == 2,
            seconds=70,
        )
        for row in rows:
            entries: Final = object_value(row["metadata"])["guardrail_information"]
            assert isinstance(entries, list) and len(entries) == 1, row


def test_opt_in_shield_scans_responses_input_exactly_once(
    optin_rig: tuple[Gateway, Wire, Wire],
) -> None:
    gateway, azure, provider = optin_rig
    prompt: Final = "synthetic prompt optin " + uuid.uuid4().hex
    with gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="openai/gpt-4.1-mini", api_base=provider.url + "/v1", api_key="synthetic-provider-key"
        )
        skipped: Final = gateway.request("POST", "/v1/responses", {"model": model, "input": prompt})
        assert skipped.status_code == 200, skipped.text
        assert _shield_prompts(azure.drain()) == ()
        response: Final = gateway.request(
            "POST", "/v1/responses", {"model": model, "guardrails": [_OPT_IN_SHIELD], "input": prompt}
        )
        assert response.status_code == 200, response.text
        assert _shield_prompts(azure.drain()) == (prompt,)
        rows: Final = eventually(
            lambda: read_rows(
                "SELECT metadata FROM \"LiteLLM_SpendLogs\" WHERE model_group=%s AND metadata->>'guardrail_information' IS NOT NULL",
                (model,),
            ),
            lambda values: len(values) == 1,
            seconds=70,
        )
        entries: Final = object_value(rows[0]["metadata"])["guardrail_information"]
        assert isinstance(entries, list) and len(entries) == 1, rows
        entry: Final = object_value(entries[0])
        assert entry["guardrail_name"] == _OPT_IN_SHIELD, entry


def test_during_call_shield_does_not_scan_any_endpoint(during_rig: tuple[Gateway, Wire, Wire]) -> None:
    gateway, azure, provider = during_rig
    chat_prompt: Final = "synthetic prompt during-chat " + uuid.uuid4().hex
    responses_prompt: Final = "synthetic prompt during-responses " + uuid.uuid4().hex
    with gateway.scenario() as scenario:
        chat_model: Final = scenario.model(
            model="openai/gpt-4.1-mini", api_base=provider.url + "/v1", api_key="synthetic-provider-key"
        )
        responses_model: Final = scenario.model(
            model="openai/gpt-4.1-mini", api_base=provider.url + "/v1", api_key="synthetic-provider-key"
        )
        chat_response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": chat_model, "messages": [{"role": "user", "content": chat_prompt}]},
        )
        responses_response: Final = gateway.request(
            "POST", "/v1/responses", {"model": responses_model, "input": responses_prompt}
        )
        assert chat_response.status_code == responses_response.status_code == 200, (
            chat_response.text,
            responses_response.text,
        )
        assert _shield_prompts(azure.drain()) == ()
        assert len(_provider_calls(provider)) == 2


def test_concurrent_mixed_requests_scan_each_prompt_once(
    audit_rig: tuple[OwnedProxy, Wire, Wire, threading.Event],
) -> None:
    owned, azure, provider, _ = audit_rig
    cells: Final = tuple((f"c1-{index}-{uuid.uuid4().hex[:8]}", index // 10, index % 10 < 5) for index in range(30))
    with owned.gateway.scenario() as scenario:
        chat_model: Final = scenario.model(
            model="openai/gpt-4.1-mini", api_base=provider.url + "/v1", api_key="synthetic-provider-key"
        )
        messages_model: Final = scenario.model(
            model="anthropic/claude-sonnet-4-5-20250929", api_base=provider.url, api_key="synthetic-provider-key"
        )
        responses_model: Final = scenario.model(
            model="deepseek/gpt-4o-mini", api_base=provider.url + "/v1", api_key="synthetic-provider-key"
        )

        def call(cell: tuple[str, int, bool]) -> tuple[str, int]:
            identity, kind, stream = cell
            if kind == 0:
                reply: Final = owned.gateway.request(
                    "POST",
                    "/v1/chat/completions",
                    {"model": chat_model, "messages": [{"role": "user", "content": identity}], "max_tokens": 16},
                )
                return identity, reply.status_code
            if kind == 1:
                reply2: Final = owned.gateway.request(
                    "POST",
                    "/v1/messages",
                    {"model": messages_model, "messages": [{"role": "user", "content": identity}], "max_tokens": 16},
                )
                return identity, reply2.status_code
            if stream:
                with owned.gateway.client.stream(
                    "POST",
                    "/v1/responses",
                    json={"model": responses_model, "input": identity, "stream": True},
                    headers={"Authorization": f"Bearer {owned.gateway.key}"},
                ) as reply3:
                    reply3.read()
                return identity, reply3.status_code
            reply4: Final = owned.gateway.request(
                "POST", "/v1/responses", {"model": responses_model, "input": identity}
            )
            return identity, reply4.status_code

        with ThreadPoolExecutor(max_workers=15) as pool:
            outcomes: Final = tuple(pool.map(call, cells))
    assert {status for _, status in outcomes} == {200}, outcomes
    scans: Final = _shield_prompts(azure.drain())
    expected: Final = tuple(identity for identity, _, _ in cells)
    assert sorted(scans) == sorted(expected), scans
    assert len(_provider_calls(provider)) == 30
    for model_group in (chat_model, messages_model, responses_model):
        rows: Final = eventually(
            lambda group=model_group: read_rows(
                'SELECT metadata FROM "LiteLLM_SpendLogs" WHERE model_group=%s', (group,)
            ),
            lambda values: len(values) == 10,
            seconds=70,
        )
        for row in rows:
            entries: Final = object_value(row["metadata"])["guardrail_information"]
            assert isinstance(entries, list) and len(entries) == 1, row


def test_azure_outage_burst_then_recovery_bills_fresh_requests_once(
    audit_rig: tuple[OwnedProxy, Wire, Wire, threading.Event],
) -> None:
    owned, azure, provider, outage = audit_rig
    with owned.gateway.scenario() as scenario:
        chat_model: Final = scenario.model(
            model="openai/gpt-4.1-mini", api_base=provider.url + "/v1", api_key="synthetic-provider-key"
        )
        responses_model: Final = scenario.model(
            model="openai/gpt-4.1-mini", api_base=provider.url + "/v1", api_key="synthetic-provider-key"
        )
        outage.set()
        try:
            burst: Final = (
                owned.gateway.request(
                    "POST",
                    "/v1/chat/completions",
                    {"model": chat_model, "messages": [{"role": "user", "content": "outage " + uuid.uuid4().hex}]},
                ),
                owned.gateway.request(
                    "POST", "/v1/responses", {"model": responses_model, "input": "outage " + uuid.uuid4().hex}
                ),
            )
        finally:
            outage.clear()
        classes: Final = {response.status_code // 100 for response in burst}
        assert len(classes) == 1, [(r.status_code, r.text) for r in burst]
        _provider_calls(provider)
        azure.drain()
        recovery_chat_model: Final = scenario.model(
            model="openai/gpt-4.1-mini", api_base=provider.url + "/v1", api_key="synthetic-provider-key"
        )
        recovery_responses_model: Final = scenario.model(
            model="openai/gpt-4.1-mini", api_base=provider.url + "/v1", api_key="synthetic-provider-key"
        )
        chat_prompt: Final = "recovered chat " + uuid.uuid4().hex
        responses_prompt: Final = "recovered responses " + uuid.uuid4().hex
        chat_reply: Final = owned.gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": recovery_chat_model, "messages": [{"role": "user", "content": chat_prompt}]},
        )
        responses_reply: Final = owned.gateway.request(
            "POST", "/v1/responses", {"model": recovery_responses_model, "input": responses_prompt}
        )
        assert chat_reply.status_code == 200 and responses_reply.status_code == 200, (
            chat_reply.text,
            responses_reply.text,
        )
        assert _shield_prompts(azure.drain()) == (chat_prompt, responses_prompt)
        assert len(_provider_calls(provider)) == 2
        rows: Final = eventually(
            lambda: read_rows(
                'SELECT metadata FROM "LiteLLM_SpendLogs" WHERE model_group IN (%s, %s) ORDER BY request_id',
                (recovery_chat_model, recovery_responses_model),
            ),
            lambda values: len(values) == 2,
            seconds=70,
        )
        for row in rows:
            entries: Final = object_value(row["metadata"])["guardrail_information"]
            assert isinstance(entries, list) and len(entries) == 1, row
            entry: Final = object_value(entries[0])
            assert entry["guardrail_status"] == "success", entry


def test_killing_a_worker_mid_burst_leaves_no_duplicate_rows(
    chaos_rig: tuple[OwnedProxy, Wire, Wire, threading.Event],
) -> None:
    owned, azure, provider, _ = chaos_rig
    port: Final = owned.gateway.client.base_url.port
    workers: Final = tuple(
        child
        for child in psutil.Process(owned.process.pid).children(recursive=False)
        if any(connection.laddr.port == port for connection in child.net_connections(kind="tcp"))
    )
    assert len(workers) == 2, [worker.pid for worker in workers]
    with owned.gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="openai/gpt-4.1-mini", api_base=provider.url + "/v1", api_key="synthetic-provider-key"
        )
        identities: Final = tuple(f"c3-{index}-{uuid.uuid4().hex[:8]}" for index in range(12))

        def call(identity: str) -> tuple[str, int]:
            reply: Final = owned.gateway.request("POST", "/v1/responses", {"model": model, "input": identity})
            return identity, reply.status_code

        with ThreadPoolExecutor(max_workers=6) as pool:
            future_map: Final = tuple(pool.submit(call, identity) for identity in identities)
            workers[0].kill()
            outcomes: Final = tuple(
                future.result() if not future.exception() else (identities[index], -1)
                for index, future in enumerate(future_map)
            )
    survivors: Final = tuple(status for _, status in outcomes if status != -1)
    assert survivors and {status for status in survivors} == {200}, outcomes
    scans: Final = _shield_prompts(azure.drain())
    assert len(scans) == len(set(scans)), scans
    assert set(scans) <= set(identities), scans
    assert {identity for identity, status in outcomes if status == 200} <= set(scans), (outcomes, scans)
    rows: Final = eventually(
        lambda: read_rows('SELECT request_id, metadata FROM "LiteLLM_SpendLogs" WHERE model_group=%s', (model,)),
        lambda values: len(values) >= len(survivors),
        seconds=30,
        return_last_on_timeout=True,
    )
    assert rows, outcomes
    assert len(rows) <= len(survivors), (outcomes, rows)
    assert len({row["request_id"] for row in rows}) == len(rows), rows
    for row in rows:
        entries: Final = object_value(row["metadata"])["guardrail_information"]
        assert isinstance(entries, list) and len(entries) == 1, row
