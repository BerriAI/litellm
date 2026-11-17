import json
import threading
import uuid
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Final

import pytest
from integration._support.client import Gateway, Scenario, eventually, gateway_from_environment
from integration._support.process import OwnedProxy, graceful_stop_seconds, owned_proxy_process
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter
from tokenizers import Regex, Tokenizer, models, pre_tokenizers

pytestmark: Final = pytest.mark.timeout(2 * graceful_stop_seconds() + 120)

_LLAMA_3_BACKEND: Final = "databricks-meta-llama-3-3-70b-instruct"
_DEEPSEEK_V4: Final = "azure_ai/DeepSeek-V4-Flash-0731"
_API_KEY: Final = "synthetic-databricks-key"
_PROMPT: Final = "Count this message before the tokenizer comes back"
_COUNTS_AFTER_FAILURE: Final = 4
_TOKENIZER_FILE: Final = "/resolve/main/tokenizer.json"
_TOKENIZER_HEADERS: Final = MappingProxyType({"X-Repo-Commit": "0" * 40, "ETag": '"scripted-char-tokenizer"'})
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])


class _Usage(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")
    prompt_tokens: int


class _Chunk(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")
    id: str
    usage: _Usage | None = None


def _hf_environment(directory: Path) -> dict[str, str]:
    return {
        "HF_HOME": str(directory / "hf-home"),
        "HF_HUB_CACHE": str(directory / "hf-cache"),
        "HF_HUB_OFFLINE": "0",
    }


def _missing(request: Request) -> Reply:
    return Reply(status=404, body=json.dumps({"error": f"unscripted target {request.target}"}).encode())


@pytest.fixture(scope="module")
def unreachable_hub_proxy(tmp_path_factory: pytest.TempPathFactory) -> Iterator[tuple[OwnedProxy, Wire]]:
    directory: Final = tmp_path_factory.mktemp("hf-tokenizer-unreachable")
    with (
        gateway_from_environment() as gateway,
        wire_server(_missing) as hub,
        owned_proxy_process(gateway, directory, {**_hf_environment(directory), "HF_ENDPOINT": hub.url}) as owned,
    ):
        yield owned, hub


def _llama_3(scenario: Scenario, api_base: str) -> str:
    return scenario.model(model=f"databricks/{_LLAMA_3_BACKEND}", api_base=api_base, api_key=_API_KEY)


def _local_count(gateway: Gateway, model: str) -> tuple[JsonValue, JsonValue]:
    response: Final = gateway.request(
        "POST",
        "/utils/token_counter",
        {"model": model, "messages": [{"role": "user", "content": _PROMPT}]},
        params={"call_endpoint": "false"},
    )
    assert response.status_code == 200, response.text
    payload: Final = _JSON_OBJECT.validate_json(response.content)
    return payload["tokenizer_type"], payload["total_tokens"]


def _frame(identity: str, delta: Mapping[str, JsonValue], finish_reason: str | None) -> bytes:
    chunk: Final = {
        "id": identity,
        "object": "chat.completion.chunk",
        "created": 1,
        "model": _LLAMA_3_BACKEND,
        "choices": [{"index": 0, "delta": dict(delta), "finish_reason": finish_reason}],
    }
    return b"data: " + json.dumps(chunk).encode() + b"\n\n"


def _stream_without_usage(identity: str) -> Callable[[Request], Reply]:
    frames: Final = (
        _frame(identity, {"role": "assistant", "content": "Counted "}, None),
        _frame(identity, {"content": "locally."}, None),
        _frame(identity, {}, "stop"),
        b"data: [DONE]\n\n",
    )

    def respond(_request: Request) -> Reply:
        return Reply(content_type="text/event-stream", chunks=frames)

    return respond


def test_counts_after_a_failed_llama_3_tokenizer_download_never_reach_the_hub_again(
    unreachable_hub_proxy: tuple[OwnedProxy, Wire],
) -> None:
    owned, hub = unreachable_hub_proxy
    with owned.gateway.scenario() as scenario:
        model: Final = _llama_3(scenario, owned.gateway.upstream_url)
        first: Final = _local_count(owned.gateway, model)
        attempted: Final = hub.received.qsize()
        later: Final = tuple(_local_count(owned.gateway, model) for _ in range(_COUNTS_AFTER_FAILURE))
        retried: Final = hub.received.qsize() - attempted
    assert attempted > 0, "the proxy never asked the scripted hub for the llama-3 tokenizer"
    assert first[0] == "openai_tokenizer", first
    assert later == (first,) * _COUNTS_AFTER_FAILURE, later
    assert retried == 0, f"{retried} hub requests after the download already failed"


def test_streamed_llama_3_chat_after_a_failed_download_counts_usage_without_reaching_the_hub(
    unreachable_hub_proxy: tuple[OwnedProxy, Wire],
) -> None:
    owned, hub = unreachable_hub_proxy
    identity: Final = f"chatcmpl-{uuid.uuid4().hex}"
    with wire_server(_stream_without_usage(identity)) as upstream, owned.gateway.scenario() as scenario:
        model: Final = _llama_3(scenario, upstream.url)
        counted: Final = _local_count(owned.gateway, model)
        attempted: Final = hub.received.qsize()
        with owned.gateway.client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": _PROMPT}],
                "stream": True,
                "stream_options": {"include_usage": True},
            },
            headers={"Authorization": f"Bearer {owned.gateway.key}"},
        ) as response:
            assert response.status_code == 200, response.read()
            lines: Final = tuple(line for line in response.iter_lines() if line.startswith("data: "))
        retried: Final = hub.received.qsize() - attempted
        sent: Final = tuple((request.method, request.target) for request in upstream.drain())
    assert counted[0] == "openai_tokenizer", counted
    assert lines[-1] == "data: [DONE]", lines
    chunks: Final = tuple(_Chunk.model_validate_json(line.removeprefix("data: ")) for line in lines[:-1])
    assert {chunk.id for chunk in chunks} == {identity}, lines
    usages: Final = tuple(chunk.usage for chunk in chunks if chunk.usage is not None)
    assert tuple(usage.prompt_tokens for usage in usages) == (counted[1],), (counted, lines)
    assert sent == (("POST", "/chat/completions"),), sent
    assert retried == 0, f"{retried} hub requests while streaming after the download already failed"


def test_a_failed_llama_3_download_leaves_the_deepseek_tokenizer_counting(
    unreachable_hub_proxy: tuple[OwnedProxy, Wire],
) -> None:
    owned, _hub = unreachable_hub_proxy
    with owned.gateway.scenario() as scenario:
        llama: Final = _llama_3(scenario, owned.gateway.upstream_url)
        deepseek: Final = scenario.model(model=_DEEPSEEK_V4, api_base=owned.gateway.upstream_url, api_key=_API_KEY)
        assert _local_count(owned.gateway, llama)[0] == "openai_tokenizer"
        counted: Final = _local_count(owned.gateway, deepseek)
    assert counted[0] == "huggingface_tokenizer", counted


def _char_tokenizer() -> bytes:
    tokenizer: Final = Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Split(Regex("."), behavior="isolated")
    return tokenizer.to_str().encode()


def _recovering_hub(recovered: threading.Event, tokenizer: bytes) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        if recovered.is_set() and request.target.endswith(_TOKENIZER_FILE):
            return Reply(body=tokenizer, headers=_TOKENIZER_HEADERS)
        return _missing(request)

    return respond


def test_a_failed_llama_3_download_is_retried_once_the_retry_window_passes(gateway: Gateway, tmp_path: Path) -> None:
    recovered: Final = threading.Event()
    with (
        wire_server(_recovering_hub(recovered, _char_tokenizer())) as hub,
        owned_proxy_process(
            gateway,
            tmp_path,
            {**_hf_environment(tmp_path), "HF_ENDPOINT": hub.url, "HUGGINGFACE_TOKENIZER_LOAD_RETRY_SECONDS": "1"},
        ) as owned,
        owned.gateway.scenario() as scenario,
    ):
        model: Final = _llama_3(scenario, owned.gateway.upstream_url)
        failed: Final = _local_count(owned.gateway, model)
        hub.drain()
        recovered.set()
        counted: Final = eventually(
            lambda: _local_count(owned.gateway, model), lambda count: count[0] == "huggingface_tokenizer", seconds=30
        )
        downloads: Final = tuple(request.target for request in hub.drain() if request.target.endswith(_TOKENIZER_FILE))
    assert failed[0] == "openai_tokenizer", failed
    assert counted[1] != failed[1], (failed, counted)
    assert downloads, "the tokenizer came back without a new download"
