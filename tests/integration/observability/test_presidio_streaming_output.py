import json
import re
import signal
import threading
import uuid
from collections.abc import Callable, Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import psutil
import yaml
from integration._support.client import Gateway, eventually
from integration._support.process import OwnedProxy, group_members, owned_proxy_process
from integration._support.wire import Reply, Request, Wire, wire_server
from openai import OpenAI
from pydantic import BaseModel

PERSON: Final = "John Smith"
MASK: Final = "<PERSON>"
GEMINI_MODEL: Final = "gemini-2.5-flash"


def gemini_frame(text: str) -> bytes:
    payload: Final = {
        "candidates": [{"content": {"parts": [{"text": text}], "role": "model"}, "index": 0}],
        "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5, "totalTokenCount": 15},
        "modelVersion": GEMINI_MODEL,
    }
    return b"data: " + json.dumps(payload).encode() + b"\r\n\r\n"


class GeminiPart(BaseModel):
    text: str


class GeminiContent(BaseModel):
    parts: list[GeminiPart]


class GeminiCandidate(BaseModel):
    content: GeminiContent


class GeminiFrame(BaseModel):
    candidates: list[GeminiCandidate]


def data_payloads(raw: bytes) -> tuple[dict[str, object], ...]:
    """JSON payload of each ``data:`` frame, whatever line ending the sender used."""
    return tuple(json.loads(line[len("data: ") :]) for line in raw.decode().splitlines() if line.startswith("data: "))


def gemini_text(payload: Mapping[str, object]) -> str:
    return GeminiFrame.model_validate(payload).candidates[0].content.parts[0].text


def gemini_texts(raw: bytes) -> tuple[str, ...]:
    return tuple(gemini_text(payload) for payload in data_payloads(raw))


def anthropic_frame(event_type: str, payload: dict[str, object]) -> bytes:
    return f"event: {event_type}\ndata: {json.dumps(payload)}\n\n".encode()


def anthropic_stream(identity: str, text: str) -> tuple[bytes, ...]:
    return (
        anthropic_frame(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": identity,
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-sonnet-4-5-20250929",
                    "content": [],
                    "stop_reason": None,
                    "usage": {"input_tokens": 11, "output_tokens": 0},
                },
            },
        ),
        anthropic_frame(
            "content_block_start",
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        ),
        anthropic_frame(
            "content_block_delta",
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}},
        ),
        anthropic_frame("content_block_stop", {"type": "content_block_stop", "index": 0}),
        anthropic_frame(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": 4},
            },
        ),
        anthropic_frame("message_stop", {"type": "message_stop"}),
    )


def openai_frame(identity: str, delta: dict[str, str], finish: str | None = None) -> bytes:
    payload: Final = {
        "id": identity,
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "gpt-4o-mini",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    return b"data: " + json.dumps(payload).encode() + b"\n\n"


def analyzer(request: Request) -> Reply:
    assert request.target == "/analyze", request.target
    text: Final = json.loads(request.body)["text"]
    findings: Final = [
        {"entity_type": "PERSON", "start": match.start(), "end": match.end(), "score": 0.85}
        for match in re.finditer(re.escape(PERSON), text)
    ]
    return Reply(body=json.dumps(findings).encode())


def anonymizer(request: Request) -> Reply:
    assert request.target == "/anonymize", request.target
    body: Final = json.loads(request.body)
    text: Final = body["text"]
    items: Final = [
        {"entity_type": "PERSON", "start": item["start"], "end": item["end"], "operator": "replace", "text": MASK}
        for item in body["analyzer_results"]
    ]
    return Reply(body=json.dumps({"text": text.replace(PERSON, MASK), "items": items}).encode())


def broken(request: Request) -> Reply:
    return Reply(status=500, body=b'{"error": "scripted outage"}')


@dataclass(frozen=True, slots=True)
class Received:
    status: int
    frames: tuple[bytes, ...]

    @property
    def text(self) -> str:
        return b"".join(self.frames).decode()


@dataclass(frozen=True, slots=True)
class Rig:
    proxy: OwnedProxy
    upstream: Wire
    analyzer: Wire
    anonymizer: Wire
    guardrail: str
    gemini: str
    anthropic: str
    openai: str

    @property
    def gateway(self) -> Gateway:
        return self.proxy.gateway

    def stream(self, path: str, body: dict[str, object] | None = None, *, key: str | None = None) -> Received:
        with self.gateway.client.stream(
            "POST", path, json=body, headers={"Authorization": f"Bearer {key or self.gateway.key}"}
        ) as response:
            return Received(response.status_code, tuple(response.iter_raw()))

    def gemini_path(self) -> str:
        return f"/v1beta/models/{self.gemini}:streamGenerateContent?alt=sse"

    def gemini_body(self) -> dict[str, object]:
        return {"contents": [{"role": "user", "parts": [{"text": "who designed it"}]}]}

    def messages_body(self, *, guardrails: tuple[str, ...] | None = None) -> dict[str, object]:
        return {
            "model": self.anthropic,
            "max_tokens": 64,
            "stream": True,
            "messages": [{"role": "user", "content": f"who designed it {uuid.uuid4().hex}"}],
            **({"guardrails": list(guardrails)} if guardrails is not None else {}),
        }


def anthropic_text(received: Received) -> str:
    events: Final = tuple(
        json.loads(line.removeprefix("data: ")) for line in received.text.split("\n") if line.startswith("data: ")
    )
    return "".join(event["delta"]["text"] for event in events if event.get("type") == "content_block_delta")


def anthropic_message_id(received: Received) -> str:
    events: Final = tuple(
        json.loads(line.removeprefix("data: ")) for line in received.text.split("\n") if line.startswith("data: ")
    )
    return "".join(event["message"]["id"] for event in events if event.get("type") == "message_start")


@contextmanager
def presidio_rig(
    gateway: Gateway,
    tmp_path: Path,
    provider: Callable[[Request], Reply],
    *,
    analyze: Callable[[Request], Reply] = analyzer,
    anonymize: Callable[[Request], Reply] = anonymizer,
    default_on: bool = True,
) -> Iterator[Rig]:
    guardrail: Final = "presidio" + uuid.uuid4().hex
    with ExitStack() as stack:
        upstream: Final = stack.enter_context(wire_server(provider))
        analyze_sink: Final = stack.enter_context(wire_server(analyze))
        anonymize_sink: Final = stack.enter_context(wire_server(anonymize))
        config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
        config["guardrails"] = [
            {
                "guardrail_name": guardrail,
                "litellm_params": {
                    "guardrail": "presidio",
                    "mode": "post_call",
                    "default_on": default_on,
                    "presidio_analyzer_api_base": analyze_sink.url,
                    "presidio_anonymizer_api_base": anonymize_sink.url,
                    "presidio_filter_scope": "output",
                },
            }
        ]
        path: Final = tmp_path / f"{guardrail}.yaml"
        path.write_text(yaml.safe_dump(config))
        proxy: Final = stack.enter_context(owned_proxy_process(gateway, tmp_path, {}, config=path, workers=2))
        scenario: Final = stack.enter_context(proxy.gateway.scenario())
        yield Rig(
            proxy=proxy,
            upstream=upstream,
            analyzer=analyze_sink,
            anonymizer=anonymize_sink,
            guardrail=guardrail,
            gemini=scenario.model(
                model=f"gemini/{GEMINI_MODEL}", api_base=upstream.url, api_key="synthetic-gemini-key"
            ),
            anthropic=scenario.model(
                model="anthropic/claude-sonnet-4-5-20250929", api_base=upstream.url, api_key="synthetic-anthropic-key"
            ),
            openai=scenario.model(model="openai/gpt-4o-mini", api_base=upstream.url + "/v1", api_key="synthetic-key"),
        )


def gemini_provider(reply: Reply) -> Callable[[Request], Reply]:
    def provider(request: Request) -> Reply:
        assert "streamGenerateContent" in request.target, request.target
        return reply

    return provider


def test_native_gemini_first_frame_reaches_caller_before_upstream_sends_the_second(
    gateway: Gateway, tmp_path: Path
) -> None:
    gate: Final = threading.Event()
    first: Final = gemini_frame("first ")
    second: Final = gemini_frame("second ")
    provider: Final = gemini_provider(
        Reply(content_type="text/event-stream", chunks=(first, second), gate_after_first=gate)
    )
    with presidio_rig(gateway, tmp_path, provider) as rig:
        with rig.gateway.client.stream(
            "POST", rig.gemini_path(), json=rig.gemini_body(), headers={"Authorization": f"Bearer {rig.gateway.key}"}
        ) as response:
            assert response.status_code == 200, response.read().decode()
            chunks: Final = response.iter_raw()
            arrived: Final = next(chunks)
            assert gemini_texts(arrived) == ("first ",), f"first chunk while upstream is gated: {arrived!r}"
            gate.set()
            rest: Final = b"".join(chunks)
        assert gemini_texts(rest) == ("second ",), rest
        assert len(rig.upstream.drain()) == 1
        assert rig.analyzer.drain() == () and rig.anonymizer.drain() == ()


def test_native_gemini_frames_received_before_upstream_abort_reach_caller(gateway: Gateway, tmp_path: Path) -> None:
    frames: Final = (gemini_frame(f"chunk {index} from {PERSON}. ") for index in range(3))
    provider: Final = gemini_provider(
        Reply(content_type="text/event-stream", chunks=tuple(frames), abort_after=2, pause_between_chunks=0.2)
    )
    with presidio_rig(gateway, tmp_path, provider) as rig:
        received: Final = rig.stream(rig.gemini_path(), rig.gemini_body())
        assert received.status == 200, received.text
        *frames_before_abort, trailer = data_payloads(b"".join(received.frames))
        assert [gemini_text(frame) for frame in frames_before_abort] == [
            f"chunk 0 from {PERSON}. ",
            f"chunk 1 from {PERSON}. ",
        ], received.text
        assert "candidates" not in trailer and json.dumps(trailer).count('"code": "500"') == 1, received.text
        assert len(rig.upstream.drain()) == 1


def test_native_gemini_first_frame_split_into_transport_fragments_streams_every_byte(
    gateway: Gateway, tmp_path: Path
) -> None:
    first: Final = gemini_frame(f"fragmented {PERSON}")
    second: Final = gemini_frame("whole")
    chunks: Final = (first[:7], first[7:19], first[19:], second)
    provider: Final = gemini_provider(Reply(content_type="text/event-stream", chunks=chunks))
    with presidio_rig(gateway, tmp_path, provider) as rig:
        received: Final = rig.stream(rig.gemini_path(), rig.gemini_body())
        assert received.status == 200, received.text
        assert gemini_texts(b"".join(received.frames)) == (f"fragmented {PERSON}", "whole")


def test_native_gemini_non_json_frame_passes_through_unchanged(gateway: Gateway, tmp_path: Path) -> None:
    frames: Final = (b"data: not json at all\r\n\r\n", gemini_frame("after"))
    provider: Final = gemini_provider(Reply(content_type="text/event-stream", chunks=frames))
    with presidio_rig(gateway, tmp_path, provider) as rig:
        received: Final = rig.stream(rig.gemini_path(), rig.gemini_body())
        assert received.status == 200, received.text
        assert received.text.replace("\r\n", "\n") == b"".join(frames).decode().replace("\r\n", "\n")


def test_native_gemini_empty_stream_returns_200_with_no_body(gateway: Gateway, tmp_path: Path) -> None:
    provider: Final = gemini_provider(Reply(content_type="text/event-stream", chunks=()))
    with presidio_rig(gateway, tmp_path, provider) as rig:
        received: Final = rig.stream(rig.gemini_path(), rig.gemini_body())
        assert received.status == 200, received.text
        assert received.text == ""


def test_native_gemini_streams_while_presidio_analyzer_is_down(gateway: Gateway, tmp_path: Path) -> None:
    frames: Final = (gemini_frame(f"{PERSON} one. "), gemini_frame("two."))
    provider: Final = gemini_provider(Reply(content_type="text/event-stream", chunks=frames))
    with presidio_rig(gateway, tmp_path, provider, analyze=broken) as rig:
        received: Final = rig.stream(rig.gemini_path(), rig.gemini_body())
        assert received.status == 200, received.text
        assert gemini_texts(b"".join(received.frames)) == (f"{PERSON} one. ", "two.")
        assert rig.analyzer.drain() == ()


def test_native_gemini_unauthenticated_request_is_rejected_before_upstream(gateway: Gateway, tmp_path: Path) -> None:
    provider: Final = gemini_provider(Reply(content_type="text/event-stream", chunks=(gemini_frame("never"),)))
    with presidio_rig(gateway, tmp_path, provider) as rig:
        received: Final = rig.stream(rig.gemini_path(), rig.gemini_body(), key="sk-not-a-key")
        assert received.status == 401, received.text
        assert rig.upstream.drain() == ()


def anthropic_provider(chunks: tuple[bytes, ...], *, pause_between_chunks: float = 0) -> Callable[[Request], Reply]:
    def provider(request: Request) -> Reply:
        assert request.target == "/v1/messages", request.target
        return Reply(content_type="text/event-stream", chunks=chunks, pause_between_chunks=pause_between_chunks)

    return provider


def test_anthropic_messages_stream_masks_person_in_text_delta(gateway: Gateway, tmp_path: Path) -> None:
    identity: Final = "msg_" + uuid.uuid4().hex
    provider: Final = anthropic_provider(anthropic_stream(identity, f"{PERSON} designed it."))
    with presidio_rig(gateway, tmp_path, provider) as rig:
        received: Final = rig.stream("/v1/messages", rig.messages_body())
        assert received.status == 200, received.text
        assert anthropic_text(received) == f"{MASK} designed it."
        assert PERSON not in received.text
        assert identity in received.text
        analyzed: Final = rig.analyzer.drain()
        anonymized: Final = rig.anonymizer.drain()
        assert len(analyzed) == len(anonymized) == 1
        assert json.loads(analyzed[0].body)["text"] == f"{PERSON} designed it."


def test_anthropic_messages_first_frame_split_across_transport_chunks_is_still_masked(
    gateway: Gateway, tmp_path: Path
) -> None:
    identity: Final = "msg_" + uuid.uuid4().hex
    whole: Final = anthropic_stream(identity, f"{PERSON} designed it.")
    split_at: Final = whole[0].index(b'"message_') + len(b'"message_')
    chunks: Final = (whole[0][:split_at], whole[0][split_at:], *whole[1:])
    with presidio_rig(gateway, tmp_path, anthropic_provider(chunks)) as rig:
        received: Final = rig.stream("/v1/messages", rig.messages_body())
        assert received.status == 200, received.text
        assert anthropic_text(received) == f"{MASK} designed it."
        assert received.text.count("event: message_start") == 1


def test_anthropic_messages_first_frame_split_inside_a_utf8_character_is_still_masked(
    gateway: Gateway, tmp_path: Path
) -> None:
    identity: Final = "msg_" + uuid.uuid4().hex
    whole: Final = anthropic_stream(identity, f"{PERSON} designed the caf\u00e9.")
    delta: Final = whole[2].replace("\\u00e9".encode(), "\u00e9".encode())
    split_at: Final = delta.index("\u00e9".encode()) + 1
    assert delta[split_at - 1 : split_at] == b"\xc3", delta
    chunks: Final = (whole[0] + whole[1] + delta[:split_at], delta[split_at:], *whole[3:])
    with presidio_rig(gateway, tmp_path, anthropic_provider(chunks, pause_between_chunks=0.5)) as rig:
        received: Final = rig.stream("/v1/messages", rig.messages_body())
        assert received.status == 200, received.text
        assert anthropic_text(received) == f"{MASK} designed the caf\u00e9."
        assert PERSON not in received.text, received.text
        assert anthropic_message_id(received) == identity, received.text


def test_anthropic_messages_stream_led_by_sse_comment_keepalive_is_still_masked(
    gateway: Gateway, tmp_path: Path
) -> None:
    identity: Final = "msg_" + uuid.uuid4().hex
    chunks: Final = (b": keepalive\n\n", *anthropic_stream(identity, f"{PERSON} designed it."))
    with presidio_rig(gateway, tmp_path, anthropic_provider(chunks, pause_between_chunks=0.5)) as rig:
        received: Final = rig.stream("/v1/messages", rig.messages_body())
        assert received.status == 200, received.text
        assert anthropic_text(received) == f"{MASK} designed it."
        assert PERSON not in received.text, received.text
        assert identity in received.text


def test_anthropic_messages_stream_led_by_data_less_ping_event_is_still_masked(
    gateway: Gateway, tmp_path: Path
) -> None:
    identity: Final = "msg_" + uuid.uuid4().hex
    chunks: Final = (b"event: ping\n\n", *anthropic_stream(identity, f"{PERSON} designed it."))
    with presidio_rig(gateway, tmp_path, anthropic_provider(chunks, pause_between_chunks=0.5)) as rig:
        received: Final = rig.stream("/v1/messages", rig.messages_body())
        assert received.status == 200, received.text
        assert anthropic_text(received) == f"{MASK} designed it."
        assert PERSON not in received.text, received.text
        assert identity in received.text


def test_anthropic_messages_stream_fails_closed_when_analyzer_is_down(gateway: Gateway, tmp_path: Path) -> None:
    identity: Final = "msg_" + uuid.uuid4().hex
    provider: Final = anthropic_provider(anthropic_stream(identity, f"{PERSON} designed it."))
    with presidio_rig(gateway, tmp_path, provider, analyze=broken) as rig:
        received: Final = rig.stream("/v1/messages", rig.messages_body())
        assert PERSON not in received.text, received.text
        assert "Presidio analyzer" in received.text, received.text
        assert rig.anonymizer.drain() == ()


def test_anthropic_messages_per_request_guardrails_selects_masking(gateway: Gateway, tmp_path: Path) -> None:
    identity: Final = "msg_" + uuid.uuid4().hex
    provider: Final = anthropic_provider(anthropic_stream(identity, f"{PERSON} designed it."))
    with presidio_rig(gateway, tmp_path, provider, default_on=False) as rig:
        unguarded: Final = rig.stream("/v1/messages", rig.messages_body())
        assert unguarded.status == 200, unguarded.text
        assert anthropic_text(unguarded) == f"{PERSON} designed it."
        assert rig.analyzer.drain() == ()
        guarded: Final = rig.stream("/v1/messages", rig.messages_body(guardrails=(rig.guardrail,)))
        assert guarded.status == 200, guarded.text
        assert anthropic_text(guarded) == f"{MASK} designed it."
        assert len(rig.analyzer.drain()) == 1


def openai_provider(identity: str) -> Callable[[Request], Reply]:
    def provider(request: Request) -> Reply:
        assert request.target == "/v1/chat/completions", request.target
        if json.loads(request.body).get("stream"):
            return Reply(
                content_type="text/event-stream",
                chunks=(
                    openai_frame(identity, {"role": "assistant", "content": ""}),
                    openai_frame(identity, {"content": f"{PERSON} designed"}),
                    openai_frame(identity, {"content": " it."}, "stop"),
                    b"data: [DONE]\n\n",
                ),
            )
        return Reply(
            body=json.dumps(
                {
                    "id": identity,
                    "object": "chat.completion",
                    "created": 1,
                    "model": "gpt-4o-mini",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": f"{PERSON} designed it."},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 11, "completion_tokens": 4, "total_tokens": 15},
                }
            ).encode()
        )

    return provider


def test_chat_completions_openai_sdk_stream_and_non_stream_are_masked(gateway: Gateway, tmp_path: Path) -> None:
    identity: Final = "chatcmpl-" + uuid.uuid4().hex
    with presidio_rig(gateway, tmp_path, openai_provider(identity)) as rig:
        client: Final = OpenAI(api_key=rig.gateway.key, base_url=f"{rig.gateway.client.base_url}/v1", max_retries=0)
        streamed: Final = client.chat.completions.create(
            model=rig.openai, messages=[{"role": "user", "content": "who designed it"}], stream=True
        )
        pieces: Final = tuple(
            chunk.choices[0].delta.content for chunk in streamed if chunk.choices and chunk.choices[0].delta.content
        )
        assert "".join(pieces) == f"{MASK} designed it.", pieces
        whole: Final = client.chat.completions.create(
            model=rig.openai, messages=[{"role": "user", "content": "who designed it"}]
        )
        assert whole.id == identity
        assert whole.choices[0].message.content == f"{MASK} designed it."
        assert len(rig.upstream.drain()) == 2
        assert len(rig.analyzer.drain()) == len(rig.anonymizer.drain()) == 2


def test_mixed_burst_survives_anonymizer_outage_and_recovers(gateway: Gateway, tmp_path: Path) -> None:
    outage: Final = threading.Event()

    def flaky_anonymizer(request: Request) -> Reply:
        return Reply(status=503, body=b'{"error": "scripted outage"}') if outage.is_set() else anonymizer(request)

    def provider(request: Request) -> Reply:
        if request.target == "/v1/messages":
            identity: Final = "msg_" + json.loads(request.body)["messages"][0]["content"]
            return Reply(content_type="text/event-stream", chunks=anthropic_stream(identity, f"{PERSON} designed it."))
        return Reply(
            content_type="text/event-stream",
            chunks=(gemini_frame(f"{PERSON} "), gemini_frame("designed it.")),
            pause_between_chunks=0.5,
        )

    with presidio_rig(gateway, tmp_path, provider, anonymize=flaky_anonymizer) as rig:

        def gemini_call(index: int) -> tuple[str, str, int]:
            received: Final = rig.stream(rig.gemini_path(), rig.gemini_body())
            return (
                "gemini",
                f"g{index}",
                received.status if gemini_texts(b"".join(received.frames)) == (f"{PERSON} ", "designed it.") else -1,
            )

        def anthropic_call(index: int) -> tuple[str, str, int]:
            body: Final = {**rig.messages_body(), "messages": [{"role": "user", "content": f"a{index}"}]}
            received: Final = rig.stream("/v1/messages", body)
            leaked: Final = PERSON in received.text
            return ("anthropic", f"a{index}", -1 if leaked else (1 if MASK in received.text else 0))

        def phase(offset: int) -> tuple[tuple[str, str, int], ...]:
            with ThreadPoolExecutor(max_workers=12) as pool:
                futures: Final = tuple(
                    pool.submit(gemini_call if index % 2 == 0 else anthropic_call, offset + index)
                    for index in range(12)
                )
                return tuple(future.result() for future in futures)

        healthy_before: Final = phase(0)
        outage.set()
        during: Final = phase(100)
        outage.clear()
        healthy_after: Final = phase(200)

    for name, results in (("before", healthy_before), ("during", during), ("after", healthy_after)):
        assert all(status == 200 for kind, _, status in results if kind == "gemini"), (name, results)
    assert all(status == 1 for kind, _, status in healthy_before + healthy_after if kind == "anthropic"), (
        healthy_before,
        healthy_after,
    )
    assert all(status == 0 for kind, _, status in during if kind == "anthropic"), during
    identities: Final = tuple(identity for _, identity, _ in healthy_before + during + healthy_after)
    assert len(identities) == len(set(identities)) == 36


def test_native_gemini_keeps_streaming_after_one_worker_is_killed(gateway: Gateway, tmp_path: Path) -> None:
    frames: Final = (gemini_frame("alive "), gemini_frame("still."))
    provider: Final = gemini_provider(Reply(content_type="text/event-stream", chunks=frames, pause_between_chunks=0.5))
    with presidio_rig(gateway, tmp_path, provider) as rig:
        workers: Final = eventually(
            lambda: tuple(
                member for member in group_members(rig.proxy.process.pid) if member.pid != rig.proxy.process.pid
            ),
            lambda members: len(members) >= 2,
            seconds=30,
        )
        victim: Final = workers[0]
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures: Final = tuple(pool.submit(rig.stream, rig.gemini_path(), rig.gemini_body()) for _ in range(8))
            victim.send_signal(signal.SIGKILL)
            psutil.wait_procs((victim,), timeout=10)
            first_wave: Final = tuple(future.result() for future in futures)
        survivors: Final = tuple(received for received in first_wave if received.status == 200)
        assert survivors, [received.text[:200] for received in first_wave]
        assert all(gemini_texts(b"".join(received.frames)) == ("alive ", "still.") for received in survivors)
        second_wave: Final = tuple(rig.stream(rig.gemini_path(), rig.gemini_body()) for _ in range(6))
        assert all(received.status == 200 for received in second_wave), [r.text[:200] for r in second_wave]
        assert all(gemini_texts(b"".join(received.frames)) == ("alive ", "still.") for received in second_wave)
        assert rig.proxy.process.poll() is None
