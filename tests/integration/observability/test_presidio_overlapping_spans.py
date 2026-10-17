import json
import re
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from types import MappingProxyType
from typing import Final, Literal

import pytest
import yaml
from integration._support.client import Gateway, gateway_from_environment
from integration._support.process import owned_proxy_process
from integration._support.wire import Reply, Request, Wire, wire_server

MODEL: Final = "gpt-6.1-sol"
PHONE: Final = "4251234567"
EMAIL: Final = "john.smith@example.com"
INTERNATIONAL_PHONE: Final = "+33 1 42 68 53 00"
US_PHONE: Final = "+1 (206) 555-0199"
SAME_SPAN: Final = f"Repeat exactly: my number is {PHONE}"
NESTED: Final = f"contact {EMAIL} today"
CONTAINED_FRAGMENT: Final = f"her phone is {INTERNATIONAL_PHONE}."
SEPARATE_PARTS: Final = (f"My number is {PHONE}.", f"Also call {US_PHONE}.")
# Spans, scores and anonymized texts recorded from presidio-analyzer and presidio-anonymizer 2.2.364 on 2026-10-16
DETECTIONS: Final = MappingProxyType(
    {
        PHONE: (("PHONE_NUMBER", 0, 10, 0.75), ("US_BANK_NUMBER", 0, 10, 0.05), ("US_DRIVER_LICENSE", 0, 10, 0.01)),
        EMAIL: (("EMAIL_ADDRESS", 0, 22, 1.0), ("URL", 0, 7, 0.5), ("URL", 11, 22, 0.5)),
        "today": (("DATE_TIME", 0, 5, 0.85),),
        INTERNATIONAL_PHONE: (("DATE_TIME", 12, 17, 0.85), ("PHONE_NUMBER", 0, 17, 0.75)),
        US_PHONE: (("PHONE_NUMBER", 0, 17, 0.4),),
    }
)
ANONYMIZED: Final = MappingProxyType(
    {
        SAME_SPAN: "Repeat exactly: my number is <PHONE_NUMBER>",
        NESTED: "contact <EMAIL_ADDRESS> <DATE_TIME>",
        CONTAINED_FRAGMENT: "her phone is <PHONE_NUMBER>.",
        SEPARATE_PARTS[0]: "My number is <PHONE_NUMBER>.",
        SEPARATE_PARTS[1]: "Also call <PHONE_NUMBER>.",
    }
)
CASES: Final = (
    pytest.param(SAME_SPAN, "Repeat exactly: my number is <PHONE_NUMBER_1>", id="same_span"),
    pytest.param(NESTED, "contact <EMAIL_ADDRESS_1> <DATE_TIME_2>", id="nested"),
    pytest.param(CONTAINED_FRAGMENT, "her phone is <PHONE_NUMBER_1>.", id="container_outranks_higher_scored_fragment"),
)
STREAM: Final = pytest.mark.parametrize("stream", [False, True], ids=["non_streaming", "streaming"])


def _detections(text: str, value: str) -> Iterator[dict[str, str | int | float]]:
    for match in re.finditer(rf"(?<!\w){re.escape(value)}(?!\w)", text):
        for entity, start, end, score in DETECTIONS[value]:
            yield {"entity_type": entity, "start": match.start() + start, "end": match.start() + end, "score": score}


def _findings(text: str) -> list[dict[str, str | int | float]]:
    findings: Final = chain.from_iterable(_detections(text, value) for value in DETECTIONS)
    return sorted(findings, key=lambda item: -item["score"])


def _spans(results: list[dict[str, str | int | float]]) -> list[tuple[str | int | float, ...]]:
    return sorted(tuple(result[key] for key in ("entity_type", "start", "end", "score")) for result in results)


def _analyze(request: Request) -> Reply:
    assert request.target == "/analyze", request.target
    return Reply(body=json.dumps(_findings(json.loads(request.body)["text"])).encode())


def _anonymize(request: Request) -> Reply:
    assert request.target == "/anonymize", request.target
    body: Final = json.loads(request.body)
    text: Final = body["text"]
    if text not in ANONYMIZED:
        return Reply(status=500, body=json.dumps({"error": f"no recorded anonymization for {text!r}"}).encode())
    if _spans(body["analyzer_results"]) != _spans(_findings(text)):
        return Reply(
            status=500, body=json.dumps({"error": f"analyzer_results differ from the detections for {text!r}"}).encode()
        )
    return Reply(body=json.dumps({"text": ANONYMIZED[text], "items": []}).encode())


def _texts(messages: list[dict]) -> tuple[str, ...]:
    return tuple(
        chain.from_iterable(
            (message["content"],)
            if isinstance(message["content"], str)
            else (part["text"] for part in message["content"])
            for message in messages
        )
    )


def _chunk(text: str, finish: str | None) -> bytes:
    delta: Final = {"role": "assistant", "content": text} if finish is None else {}
    payload: Final = {
        "id": "chatcmpl-echo",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": MODEL,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    return b"data: " + json.dumps(payload).encode() + b"\n\n"


def _echo(request: Request) -> Reply:
    assert request.target == "/v1/chat/completions", request.target
    body: Final = json.loads(request.body)
    text: Final = " | ".join(_texts(body["messages"]))
    if body.get("stream"):
        pieces: Final = tuple(text[index : index + 6] for index in range(0, len(text), 6))
        return Reply(
            content_type="text/event-stream",
            chunks=(*(_chunk(piece, None) for piece in pieces), _chunk("", "stop"), b"data: [DONE]\n\n"),
        )
    reply: Final = {
        "id": "chatcmpl-echo",
        "object": "chat.completion",
        "created": 1,
        "model": MODEL,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 9, "completion_tokens": 9, "total_tokens": 18},
    }
    return Reply(body=json.dumps(reply).encode())


@dataclass(frozen=True, slots=True)
class Rig:
    proxy: Gateway
    provider: Wire


def _config(root: Path, provider: Wire, analyzer: Wire, anonymizer: Wire, scope: Literal["input"] | None) -> Path:
    config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    config["litellm_settings"]["cache"] = False
    config["model_list"].append(
        {
            "model_name": MODEL,
            "litellm_params": {"model": f"openai/{MODEL}", "api_base": provider.url + "/v1", "api_key": "synthetic"},
        }
    )
    guardrail: Final = {
        "guardrail": "presidio",
        "mode": "pre_call",
        "output_parse_pii": True,
        "default_on": True,
        "presidio_analyzer_api_base": analyzer.url + "/",
        "presidio_anonymizer_api_base": anonymizer.url + "/",
    }
    scoped: Final = guardrail if scope is None else {**guardrail, "presidio_filter_scope": scope}
    config["guardrails"] = [{"guardrail_name": "presidio-pii", "litellm_params": scoped}]
    path: Final = root / "presidio-overlapping-spans.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


@contextmanager
def _rig(root: Path, scope: Literal["input"] | None) -> Iterator[Rig]:
    with (
        gateway_from_environment() as gateway,
        wire_server(_echo) as provider,
        wire_server(_analyze) as analyzer,
        wire_server(_anonymize) as anonymizer,
        owned_proxy_process(
            gateway, root, {}, config=_config(root, provider, analyzer, anonymizer, scope), workers=2
        ) as owned,
    ):
        yield Rig(owned.gateway, provider)


@pytest.fixture(scope="module")
def default_scope_rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    with _rig(tmp_path_factory.mktemp("presidio-default-scope"), None) as rig:
        yield rig


@pytest.fixture(scope="module")
def input_scope_rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    with _rig(tmp_path_factory.mktemp("presidio-input-scope"), "input") as rig:
        yield rig


def _send(rig: Rig, messages: list[dict], stream: bool) -> tuple[int, str]:
    rig.provider.drain()
    body: Final = {"model": MODEL, "messages": messages, "stream": stream}
    headers: Final = {"Authorization": f"Bearer {rig.proxy.key}"}
    if not stream:
        response: Final = rig.proxy.client.post("/v1/chat/completions", json=body, headers=headers)
        if response.status_code != 200:
            return response.status_code, response.text
        return response.status_code, response.json()["choices"][0]["message"]["content"]
    with rig.proxy.client.stream("POST", "/v1/chat/completions", json=body, headers=headers) as streamed:
        if streamed.status_code != 200:
            return streamed.status_code, streamed.read().decode()
        frames: Final = tuple(
            json.loads(line.removeprefix("data: "))
            for line in streamed.iter_lines()
            if line.startswith("data: ") and line != "data: [DONE]"
        )
    return 200, "".join(frame["choices"][0]["delta"].get("content") or "" for frame in frames if frame["choices"])


def _forwarded(rig: Rig) -> tuple[str, ...]:
    received: Final = rig.provider.drain()
    assert len(received) == 1, received
    return _texts(json.loads(received[0].body)["messages"])


@STREAM
@pytest.mark.parametrize(("text", "forwarded"), CASES)
def test_overlapping_detections_reach_the_provider_as_one_token_and_come_back_unmasked(
    input_scope_rig: Rig, text: str, forwarded: str, stream: bool
) -> None:
    status, replied = _send(input_scope_rig, [{"role": "user", "content": text}], stream)
    assert status == 200, replied
    assert _forwarded(input_scope_rig) == (forwarded,)
    assert replied == text


@STREAM
@pytest.mark.parametrize(("text", "forwarded"), CASES)
def test_default_scope_masks_the_cleanly_restored_reply_like_presidio_masks_the_original(
    default_scope_rig: Rig, text: str, forwarded: str, stream: bool
) -> None:
    status, replied = _send(default_scope_rig, [{"role": "user", "content": text}], stream)
    assert status == 200, replied
    assert _forwarded(default_scope_rig) == (forwarded,)
    assert replied == ANONYMIZED[text]


@STREAM
@pytest.mark.parametrize(
    "messages",
    [
        pytest.param(
            [{"role": "user", "content": [{"type": "text", "text": part} for part in SEPARATE_PARTS]}],
            id="content_parts",
        ),
        pytest.param([{"role": "user", "content": part} for part in SEPARATE_PARTS], id="messages"),
    ],
)
def test_one_type_in_separate_parts_gets_distinct_tokens_and_each_part_comes_back_as_itself(
    input_scope_rig: Rig, messages: list[dict], stream: bool
) -> None:
    status, replied = _send(input_scope_rig, messages, stream)
    assert status == 200, replied
    assert _forwarded(input_scope_rig) == ("My number is <PHONE_NUMBER_1>.", "Also call <PHONE_NUMBER_2>.")
    assert replied == " | ".join(SEPARATE_PARTS)
