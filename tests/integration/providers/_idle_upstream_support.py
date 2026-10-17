from __future__ import annotations

import asyncio
import io
import json
import uuid
import wave
from collections.abc import Callable, Generator, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from types import MappingProxyType
from typing import Final

import anthropic
import httpx
import openai
import yaml
from anthropic.types import RawContentBlockDeltaEvent, RawMessageStartEvent, TextBlock, TextDelta
from integration._support.anthropic_sse import message_json, message_stream, stream_reply
from integration._support.client import Gateway, eventually, held, list_value, object_value, string_value
from integration._support.database import read_rows
from integration._support.idle_gateway import MARKER, Arrival, Drop, IdleGateway, idle_gateway
from integration._support.openai_wire import chat_reply, chat_stream, responses_reply
from integration._support.process import OwnedProxy, WorkerHealthcheck, owned_proxy_process
from integration._support.wire import Reply, Request
from openai.types.chat import ChatCompletionChunk
from openai.types.responses import ResponseCompletedEvent, ResponseTextDeltaEvent
from pydantic import JsonValue, TypeAdapter

IDLE_SECONDS: Final = 17.0
DROP_AFTER_SECONDS: Final = 16.0
REUSE_IDLE_SECONDS: Final = 5.0
QUIET_STREAM_SECONDS: Final = 16.0
PROVIDER_KEY: Final = "sk-idle-upstream-provider"
CALL_ID: Final = "x-litellm-call-id"
RUST_SERVED: Final = "x-litellm-rust"
GUARDRAIL: Final = "idle-upstream-guardrail"
PASS_THROUGH: Final = "/idle-upstream-pass-through"
EMBEDDING: Final = (0.25, 0.5)
FRESH_CONNECTIONS: Final = httpx.Limits(max_keepalive_connections=0)
_JSON: Final = TypeAdapter(dict[str, JsonValue])
_CONFIG: Final = Path(__file__).resolve().parents[1] / "proxy_config.yaml"


def new_marker() -> str:
    return f"idle-{uuid.uuid4().hex[:16]}"


def expected_text(marker: str) -> str:
    return f"answer {marker}"


def expected_embedding(_marker: str) -> str:
    return json.dumps(list(EMBEDDING))


def _marker_of(request: Request) -> str:
    found: Final = MARKER.search(request.body)
    return found.group().decode() if found else ""


def _converse(text: str) -> bytes:
    return json.dumps(
        {
            "output": {"message": {"role": "assistant", "content": [{"text": text}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 5, "outputTokens": 3, "totalTokens": 8},
            "metrics": {"latencyMs": 1},
        }
    ).encode()


def _ocr(text: str) -> bytes:
    return json.dumps(
        {
            "model": "mistral-ocr-latest",
            "pages": [{"index": 0, "markdown": text}],
            "usage_info": {"pages_processed": 1, "doc_size_bytes": 1024},
        }
    ).encode()


def _embeddings() -> bytes:
    return json.dumps(
        {
            "object": "list",
            "data": [{"object": "embedding", "index": 0, "embedding": list(EMBEDDING)}],
            "model": "text-embedding-idle-upstream",
            "usage": {"prompt_tokens": 1, "total_tokens": 1},
        }
    ).encode()


def _model_reply(target: str, marker: str, body: Mapping[str, JsonValue]) -> Reply:
    model: Final = str(body.get("model", "idle-upstream"))
    stream: Final = body.get("stream") is True
    if target.endswith("/v1/messages"):
        if stream:
            return stream_reply(message_stream(f"msg_{marker}", model, expected_text(marker)))
        return Reply(body=message_json(f"msg_{marker}", model, expected_text(marker)))
    if target.endswith("/responses"):
        return responses_reply(f"resp_{marker}", model, expected_text(marker), stream=stream)
    if target.endswith("/chat/completions"):
        return chat_reply(f"chatcmpl-{marker}", model, expected_text(marker), stream=stream)
    return Reply(status=404, body=b'{"error": "unscripted route"}')


def answer(request: Request) -> Reply:
    marker: Final = _marker_of(request)
    target: Final = request.target.split("?", 1)[0]
    if target.endswith("/beta/litellm_basic_guardrail_api"):
        return Reply(body=json.dumps({"action": "NONE"}).encode())
    if target.startswith("/passthrough"):
        return Reply(body=json.dumps({"echo": expected_text(marker)}).encode())
    if target.endswith("/converse"):
        return Reply(body=_converse(expected_text(marker)))
    if target.endswith("/ocr"):
        return Reply(body=_ocr(expected_text(marker)))
    if target.endswith("/embeddings"):
        return Reply(body=_embeddings())
    return _model_reply(target, marker, _JSON.validate_json(request.body))


def quiet_stream_answer(request: Request) -> Reply:
    marker: Final = _marker_of(request)
    body: Final = _JSON.validate_json(request.body)
    model: Final = str(body.get("model", "idle-upstream"))
    frames: Final = (
        message_stream(f"msg_{marker}", model, expected_text(marker))
        if request.target.endswith("/v1/messages")
        else chat_stream(f"chatcmpl-{marker}", model, expected_text(marker))
    )
    return Reply(
        content_type="text/event-stream",
        chunks=(frames[0], b"".join(frames[1:])),
        pause_between_chunks=QUIET_STREAM_SECONDS,
    )


@dataclass(frozen=True, slots=True)
class ProxyTarget:
    url: str
    key: str


@dataclass(frozen=True, slots=True)
class Answer:
    status: int
    text: str
    call_id: str
    response_id: str
    rust: bool = False


@dataclass(frozen=True, slots=True)
class Sent:
    marker: str
    answer: Answer


def _call_id(headers: httpx.Headers) -> str:
    return next(iter(headers.get_list(CALL_ID)), "")


def _rust(headers: httpx.Headers) -> bool:
    return "true" in headers.get_list(RUST_SERVED)


def _refused(status: int, response: httpx.Response) -> Answer:
    return Answer(status, response.text, _call_id(response.headers), "")


def _auth(target: ProxyTarget) -> Mapping[str, str]:
    return {"Authorization": f"Bearer {target.key}"}


def _openai(target: ProxyTarget) -> openai.OpenAI:
    return openai.OpenAI(
        base_url=f"{target.url}/v1",
        api_key=target.key,
        max_retries=0,
        http_client=httpx.Client(limits=FRESH_CONNECTIONS, timeout=60, trust_env=False),
    )


def _async_openai(target: ProxyTarget) -> openai.AsyncOpenAI:
    return openai.AsyncOpenAI(
        base_url=f"{target.url}/v1",
        api_key=target.key,
        max_retries=0,
        http_client=httpx.AsyncClient(limits=FRESH_CONNECTIONS, timeout=60, trust_env=False),
    )


def _anthropic(target: ProxyTarget) -> anthropic.Anthropic:
    return anthropic.Anthropic(
        base_url=target.url,
        api_key=target.key,
        max_retries=0,
        http_client=httpx.Client(limits=FRESH_CONNECTIONS, timeout=60, trust_env=False),
    )


def _async_anthropic(target: ProxyTarget) -> anthropic.AsyncAnthropic:
    return anthropic.AsyncAnthropic(
        base_url=target.url,
        api_key=target.key,
        max_retries=0,
        http_client=httpx.AsyncClient(limits=FRESH_CONNECTIONS, timeout=60, trust_env=False),
    )


def _chat(target: ProxyTarget, model: str, marker: str, extra: Mapping[str, JsonValue]) -> Answer:
    with _openai(target) as client:
        try:
            raw: Final = client.chat.completions.with_raw_response.create(
                model=model, messages=[{"role": "user", "content": marker}], extra_body=dict(extra)
            )
        except openai.APIStatusError as error:
            return _refused(error.status_code, error.response)
        completion: Final = raw.parse()
        return Answer(200, completion.choices[0].message.content or "", _call_id(raw.headers), completion.id)


def chat_sdk(target: ProxyTarget, model: str, marker: str) -> Answer:
    return _chat(target, model, marker, {})


def chat_sdk_with_default_retries(target: ProxyTarget, model: str, marker: str) -> Answer:
    return _chat(target, model, marker, {"num_retries": 2})


def _chunk_text(chunk: ChatCompletionChunk) -> str:
    return "".join(choice.delta.content or "" for choice in chunk.choices)


async def _chat_stream(target: ProxyTarget, model: str, marker: str) -> Answer:
    async with _async_openai(target) as client:
        try:
            raw: Final = await client.chat.completions.with_raw_response.create(
                model=model, messages=[{"role": "user", "content": marker}], stream=True
            )
        except openai.APIStatusError as error:
            return _refused(error.status_code, error.response)
        try:
            chunks: Final = tuple([chunk async for chunk in raw.parse()])
        except openai.APIError as error:
            return Answer(200, f"stream broke: {error}", _call_id(raw.headers), "")
        return Answer(
            200,
            "".join(_chunk_text(chunk) for chunk in chunks),
            _call_id(raw.headers),
            chunks[0].id if chunks else "",
        )


def chat_sdk_stream(target: ProxyTarget, model: str, marker: str) -> Answer:
    return asyncio.run(_chat_stream(target, model, marker))


def _post(
    target: ProxyTarget,
    path: str,
    body: Mapping[str, JsonValue],
    read: Callable[[Mapping[str, JsonValue]], tuple[str, str]],
) -> Answer:
    with httpx.Client(limits=FRESH_CONNECTIONS, timeout=60, trust_env=False) as client:
        response: Final = client.post(f"{target.url}{path}", json=dict(body), headers=dict(_auth(target)))
    if response.status_code != 200:
        return _refused(response.status_code, response)
    text, identity = read(_JSON.validate_json(response.content))
    return Answer(200, text, _call_id(response.headers), identity, _rust(response.headers))


def _chat_body(body: Mapping[str, JsonValue]) -> tuple[str, str]:
    message: Final = object_value(object_value(list_value(body["choices"])[0])["message"])
    return string_value(message["content"]), string_value(body["id"])


def _echo_body(body: Mapping[str, JsonValue]) -> tuple[str, str]:
    return string_value(body["echo"]), ""


def _ocr_body(body: Mapping[str, JsonValue]) -> tuple[str, str]:
    return string_value(object_value(list_value(body["pages"])[0])["markdown"]), ""


def chat_httpx(target: ProxyTarget, model: str, marker: str) -> Answer:
    return _post(
        target, "/v1/chat/completions", {"model": model, "messages": [{"role": "user", "content": marker}]}, _chat_body
    )


def chat_with_guardrail(target: ProxyTarget, model: str, marker: str) -> Answer:
    return _post(
        target,
        "/v1/chat/completions",
        {"model": model, "messages": [{"role": "user", "content": marker}], "guardrails": [GUARDRAIL]},
        _chat_body,
    )


def pass_through(target: ProxyTarget, _model: str, marker: str) -> Answer:
    return _post(target, PASS_THROUGH, {"input": marker}, _echo_body)


def ocr(target: ProxyTarget, model: str, marker: str) -> Answer:
    return _post(
        target,
        "/v1/ocr",
        {
            "model": model,
            "document": {"type": "document_url", "document_url": f"https://documents.invalid/{marker}.pdf"},
        },
        _ocr_body,
    )


def _wav() -> bytes:
    buffer: Final = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(8000)
        writer.writeframes(b"\x00\x00" * 800)
    return buffer.getvalue()


def transcription(target: ProxyTarget, model: str, marker: str) -> Answer:
    with httpx.Client(limits=FRESH_CONNECTIONS, timeout=60, trust_env=False) as client:
        response: Final = client.post(
            f"{target.url}/v1/audio/transcriptions",
            files={"file": ("idle.wav", _wav(), "audio/wav")},
            data={"model": model, "prompt": marker},
            headers=dict(_auth(target)),
        )
    if response.status_code != 200:
        return _refused(response.status_code, response)
    return Answer(
        200,
        string_value(_JSON.validate_json(response.content)["text"]),
        _call_id(response.headers),
        "",
        _rust(response.headers),
    )


def embeddings_sdk(target: ProxyTarget, model: str, marker: str) -> Answer:
    with _openai(target) as client:
        try:
            raw: Final = client.embeddings.with_raw_response.create(model=model, input=marker, encoding_format="float")
        except openai.APIStatusError as error:
            return _refused(error.status_code, error.response)
        result: Final = raw.parse()
        return Answer(200, json.dumps(result.data[0].embedding), _call_id(raw.headers), "")


def responses_sdk(target: ProxyTarget, model: str, marker: str) -> Answer:
    with _openai(target) as client:
        try:
            raw: Final = client.responses.with_raw_response.create(model=model, input=marker)
        except openai.APIStatusError as error:
            return _refused(error.status_code, error.response)
        response: Final = raw.parse()
        return Answer(200, response.output_text, _call_id(raw.headers), response.id)


async def _responses_stream(target: ProxyTarget, model: str, marker: str) -> Answer:
    async with _async_openai(target) as client:
        try:
            raw: Final = await client.responses.with_raw_response.create(model=model, input=marker, stream=True)
        except openai.APIStatusError as error:
            return _refused(error.status_code, error.response)
        try:
            events: Final = tuple([event async for event in raw.parse()])
        except openai.APIError as error:
            return Answer(200, f"stream broke: {error}", _call_id(raw.headers), "")
        return Answer(
            200,
            "".join(event.delta for event in events if isinstance(event, ResponseTextDeltaEvent)),
            _call_id(raw.headers),
            next((event.response.id for event in events if isinstance(event, ResponseCompletedEvent)), ""),
        )


def responses_sdk_stream(target: ProxyTarget, model: str, marker: str) -> Answer:
    return asyncio.run(_responses_stream(target, model, marker))


def messages_sdk(target: ProxyTarget, model: str, marker: str) -> Answer:
    with _anthropic(target) as client:
        try:
            raw: Final = client.messages.with_raw_response.create(
                model=model, max_tokens=32, messages=[{"role": "user", "content": marker}]
            )
        except anthropic.APIStatusError as error:
            return _refused(error.status_code, error.response)
        message: Final = raw.parse()
        return Answer(
            200,
            "".join(block.text for block in message.content if isinstance(block, TextBlock)),
            _call_id(raw.headers),
            message.id,
            _rust(raw.headers),
        )


def _delta_text(event: object) -> str:
    if isinstance(event, RawContentBlockDeltaEvent) and isinstance(event.delta, TextDelta):
        return event.delta.text
    return ""


async def _messages_stream(target: ProxyTarget, model: str, marker: str) -> Answer:
    async with _async_anthropic(target) as client:
        try:
            raw: Final = await client.messages.with_raw_response.create(
                model=model, max_tokens=32, messages=[{"role": "user", "content": marker}], stream=True
            )
        except anthropic.APIStatusError as error:
            return _refused(error.status_code, error.response)
        try:
            events: Final = tuple([event async for event in raw.parse()])
        except anthropic.APIError as error:
            return Answer(200, f"stream broke: {error}", _call_id(raw.headers), "")
        return Answer(
            200,
            "".join(_delta_text(event) for event in events),
            _call_id(raw.headers),
            next((event.message.id for event in events if isinstance(event, RawMessageStartEvent)), ""),
            _rust(raw.headers),
        )


def messages_sdk_stream(target: ProxyTarget, model: str, marker: str) -> Answer:
    return asyncio.run(_messages_stream(target, model, marker))


@dataclass(frozen=True, slots=True)
class ConfigPart:
    models: tuple[Mapping[str, JsonValue], ...] = ()
    pass_through: tuple[Mapping[str, JsonValue], ...] = ()
    guardrails: tuple[Mapping[str, JsonValue], ...] = ()


def openai_deployment(model: str, url: str, _steady: str) -> ConfigPart:
    return ConfigPart(
        models=(
            {
                "model_name": model,
                "litellm_params": {"model": "openai/idle-upstream", "api_base": url, "api_key": PROVIDER_KEY},
            },
        )
    )


def embedding_deployment(model: str, url: str, _steady: str) -> ConfigPart:
    return ConfigPart(
        models=(
            {
                "model_name": model,
                "litellm_params": {
                    "model": "openai/text-embedding-idle-upstream",
                    "api_base": url,
                    "api_key": PROVIDER_KEY,
                },
                "model_info": {"mode": "embedding"},
            },
        )
    )


def anthropic_deployment(model: str, url: str, _steady: str) -> ConfigPart:
    return ConfigPart(
        models=(
            {
                "model_name": model,
                "litellm_params": {"model": "anthropic/claude-idle-upstream", "api_base": url, "api_key": PROVIDER_KEY},
            },
        )
    )


def ocr_deployment(model: str, url: str, _steady: str) -> ConfigPart:
    return ConfigPart(
        models=(
            {
                "model_name": model,
                "litellm_params": {"model": "mistral/mistral-ocr-latest", "api_base": url, "api_key": PROVIDER_KEY},
            },
        )
    )


def transcription_deployment(model: str, url: str, _steady: str) -> ConfigPart:
    return ConfigPart(
        models=(
            {
                "model_name": model,
                "litellm_params": {
                    "model": "bedrock/mistral.voxtral-mini-3b-2507",
                    "api_base": url,
                    "aws_access_key_id": "AKIAIDLEUPSTREAM0000",
                    "aws_secret_access_key": "idle-upstream-synthetic-secret",
                    "aws_region_name": "us-east-1",
                },
                "model_info": {"mode": "audio_transcription"},
            },
        )
    )


def pass_through_deployment(_model: str, url: str, _steady: str) -> ConfigPart:
    return ConfigPart(pass_through=({"path": PASS_THROUGH, "target": f"{url}/passthrough"},))


def guardrail_deployment(model: str, url: str, steady: str) -> ConfigPart:
    return ConfigPart(
        models=openai_deployment(model, steady, steady).models,
        guardrails=(
            {
                "guardrail_name": GUARDRAIL,
                "litellm_params": {
                    "guardrail": "generic_guardrail_api",
                    "mode": "pre_call",
                    "default_on": False,
                    "api_base": url,
                    "api_key": PROVIDER_KEY,
                },
            },
        ),
    )


@dataclass(frozen=True, slots=True)
class Row:
    name: str
    deploy: Callable[[str, str, str], ConfigPart]
    send: Callable[[ProxyTarget, str, str], Answer]
    expected: Callable[[str], str] = expected_text
    drop: Drop = Drop.CLOSE
    rust: bool = False
    respond: Callable[[Request], Reply] = answer

    @property
    def model(self) -> str:
        return f"idle-{self.name}"


def send(row: Row, target: ProxyTarget) -> Sent:
    marker: Final = new_marker()
    return Sent(marker, row.send(target, row.model, marker))


def proxy_config(directory: Path, parts: tuple[ConfigPart, ...]) -> Path:
    base: Final = _JSON.validate_python(yaml.safe_load(_CONFIG.read_text()))
    config: Final = {
        **base,
        "model_list": list(chain.from_iterable(part.models for part in parts)),
        "guardrails": list(chain.from_iterable(part.guardrails for part in parts)),
        "general_settings": {
            **object_value(base["general_settings"]),
            "pass_through_endpoints": list(chain.from_iterable(part.pass_through for part in parts)),
        },
        "router_settings": {**object_value(base["router_settings"]), "num_retries": 0},
    }
    path: Final = directory / f"idle-upstream-{uuid.uuid4().hex}.yaml"
    path.write_text(yaml.safe_dump(json.loads(json.dumps(config))))
    return path


def proxy_target(owned: OwnedProxy) -> ProxyTarget:
    return ProxyTarget(str(owned.gateway.client.base_url).rstrip("/"), owned.gateway.key)


def two_workers(owned: OwnedProxy) -> tuple[int, int]:
    first, second = sorted(eventually(owned.worker_pids, lambda pids: len(pids) == 2, seconds=120))
    return first, second


def on_each_worker(
    owned: OwnedProxy, workers: tuple[int, int], call: Callable[[], tuple[Sent, ...]]
) -> tuple[Sent, ...]:
    first, second = workers
    with owned.suspended(second):
        on_first: Final = call()
    with owned.suspended(first):
        on_second: Final = call()
    return (*on_first, *on_second)


def stay_idle(gateways: tuple[IdleGateway, ...], seconds: float) -> None:
    held(
        lambda: sum(gateway.pending() for gateway in gateways),
        lambda arrived: arrived == 0,
        holding=seconds,
        seconds=seconds + 30,
    )


def answered(sents: tuple[Sent, ...]) -> tuple[tuple[int, str], ...]:
    return tuple((sent.answer.status, sent.answer.text) for sent in sents)


def expected_answers(row: Row, sents: tuple[Sent, ...]) -> tuple[tuple[int, str], ...]:
    return tuple((200, row.expected(sent.marker)) for sent in sents)


def served_by_rust(sents: tuple[Sent, ...]) -> tuple[bool, ...]:
    return tuple(sent.answer.rust for sent in sents)


def connections(arrivals: tuple[Arrival, ...]) -> tuple[int, ...]:
    return tuple(sorted(arrival.connection for arrival in arrivals))


def landed(arrivals: tuple[Arrival, ...]) -> tuple[tuple[str, bool], ...]:
    return tuple(sorted((arrival.marker, arrival.dropped) for arrival in arrivals))


def landed_once_undropped(sents: tuple[Sent, ...]) -> tuple[tuple[str, bool], ...]:
    return tuple(sorted((sent.marker, False) for sent in sents))


def spend_statuses(sents: tuple[Sent, ...]) -> tuple[str, ...]:
    call_ids: Final = ",".join(sorted(sent.answer.call_id for sent in sents if sent.answer.call_id))
    rows: Final = eventually(
        lambda: read_rows(
            'SELECT request_id, status FROM "LiteLLM_SpendLogs" WHERE litellm_call_id = ANY(string_to_array(%s, %s))',
            (call_ids, ","),
        ),
        lambda found: len(found) >= len(sents),
        seconds=90,
        return_last_on_timeout=True,
    )
    return tuple(sorted(string_value(row["status"]) for row in rows))


@dataclass(frozen=True, slots=True)
class UpstreamProxy:
    owned: OwnedProxy
    workers: tuple[int, int]
    rows: Mapping[str, Row]
    gateways: Mapping[str, IdleGateway]

    @property
    def target(self) -> ProxyTarget:
        return proxy_target(self.owned)

    def probe(self, name: str) -> tuple[tuple[Sent, ...], tuple[Arrival, ...]]:
        row: Final = self.rows[name]
        target: Final = self.target
        sents: Final = on_each_worker(self.owned, self.workers, lambda: (send(row, target),))
        return sents, self.gateways[name].drain()

    def burst(self, names: tuple[str, ...], each: int) -> tuple[tuple[Row, Sent], ...]:
        rows: Final = tuple(chain.from_iterable((self.rows[name],) * each for name in names))
        target: Final = self.target
        with ThreadPoolExecutor(max_workers=len(rows)) as pool:
            return tuple(zip(rows, pool.map(send, rows, (target,) * len(rows)), strict=True))


@dataclass(frozen=True, slots=True)
class StalePools:
    proxy: UpstreamProxy
    warmed: Mapping[str, tuple[Arrival, ...]]

    @property
    def rows(self) -> Mapping[str, Row]:
        return self.proxy.rows

    def probe(self, name: str) -> tuple[tuple[Sent, ...], tuple[Arrival, ...]]:
        return self.proxy.probe(name)


def burst_answers(sent: tuple[tuple[Row, Sent], ...]) -> tuple[tuple[int, str], ...]:
    return tuple((one.answer.status, one.answer.text) for _, one in sent)


def burst_expected(sent: tuple[tuple[Row, Sent], ...]) -> tuple[tuple[int, str], ...]:
    return tuple((200, row.expected(one.marker)) for row, one in sent)


def drain_all(gateways: tuple[IdleGateway, ...]) -> tuple[Arrival, ...]:
    return tuple(chain.from_iterable(gateway.drain() for gateway in gateways))


@contextmanager
def proxy_for(
    gateway: Gateway,
    directory: Path,
    rows: tuple[Row, ...],
    gateways: Mapping[str, IdleGateway],
    steady: str,
    overrides: Mapping[str, str],
) -> Generator[UpstreamProxy]:
    config: Final = proxy_config(
        directory, tuple(row.deploy(row.model, gateways[row.name].url, steady) for row in rows)
    )
    with owned_proxy_process(
        gateway, directory, overrides, config=config, workers=2, healthcheck=WorkerHealthcheck.LENIENT
    ) as owned:
        yield UpstreamProxy(owned, two_workers(owned), MappingProxyType({row.name: row for row in rows}), gateways)


@contextmanager
def upstream_proxy(
    gateway: Gateway,
    directory: Path,
    rows: tuple[Row, ...],
    overrides: Mapping[str, str],
    drop_after: float | None = DROP_AFTER_SECONDS,
) -> Generator[UpstreamProxy]:
    with ExitStack() as stack:
        steady: Final = stack.enter_context(idle_gateway(answer, drop_after=None))
        gateways: Final = MappingProxyType(
            {
                row.name: stack.enter_context(idle_gateway(row.respond, drop_after=drop_after, drop=row.drop))
                for row in rows
            }
        )
        yield stack.enter_context(proxy_for(gateway, directory, rows, gateways, steady.url, overrides))


def _warm(rows: tuple[Row, ...], target: ProxyTarget) -> tuple[Sent, ...]:
    return tuple(send(row, target) for row in rows)


def warm_and_idle(proxy: UpstreamProxy) -> StalePools:
    """Gives every row one pooled upstream connection per worker, then leaves each idle for IDLE_SECONDS: past the
    15 s default keep-alive, inside the 120 s one it replaced."""
    target: Final = proxy.target
    warm: Final = on_each_worker(proxy.owned, proxy.workers, lambda: _warm(tuple(proxy.rows.values()), target))
    refused: Final = tuple(sent for sent in warm if sent.answer.status != 200)
    assert refused == (), refused
    warmed: Final = MappingProxyType({name: upstream.drain() for name, upstream in proxy.gateways.items()})
    stay_idle(tuple(proxy.gateways.values()), IDLE_SECONDS)
    return StalePools(proxy, warmed)


@contextmanager
def stale_pools(
    gateway: Gateway,
    directory: Path,
    rows: tuple[Row, ...],
    overrides: Mapping[str, str],
    drop_after: float | None = DROP_AFTER_SECONDS,
) -> Generator[StalePools]:
    with upstream_proxy(gateway, directory, rows, overrides, drop_after) as proxy:
        yield warm_and_idle(proxy)
