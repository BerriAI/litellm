"""Tests for the Rust input token counter bridge.

The native factory is dependency-injected through ``TOKEN_COUNTER.override``
so the fallback cases run without the compiled extension present. The parity
cases need the extension and are skipped when it is not built.
"""

from __future__ import annotations

import json
from types import MappingProxyType
from typing import Final

import pytest
import tiktoken
from tokenizers import Tokenizer

import litellm
from litellm.constants import TIKTOKEN_ENCODE_CHUNK_SIZE_CHARS
from litellm.litellm_core_utils.token_counter import openai_tokenizer_encoding
from litellm.proxy.spend_tracking.budget_reservation import _count_input_tokens
from litellm.rust_bridge import bindings, configuration
from litellm.rust_bridge import token_counter as bridge
from litellm.utils import claude_json_str

MODEL: Final = "claude-sonnet-4-5-20250929"
CL100K_MODEL: Final = "gpt-4"
O200K_MODEL: Final = "gpt-4o"
TOKENIZERS: Final[tuple[bridge.RustTokenizer, ...]] = ("anthropic", "cl100k_base", "o200k_base")
RANK_FILE_LINES: Final = MappingProxyType({"cl100k_base": 100_256, "o200k_base": 199_998})
BODY: Final = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": "hello"}]}).encode()


class _FakeDeclined(Exception):
    pass


class _FakeUpstream(Exception):
    pass


class _FakeNative:
    RustBridgeDeclined = _FakeDeclined
    RustUpstreamError = _FakeUpstream


class _RecordingCounter:
    def __init__(self, tokenizer_json: str) -> None:
        self.tokenizer_json = tokenizer_json
        self.bodies: list[bytes] = []

    async def acount_request(self, body: bytes) -> object:
        self.bodies.append(body)
        return {"model": MODEL, "input_tokens": 42}


class _RecordingFactory:
    """Stands in for the native `TokenCounter` class: callable for tokenizer JSON, `from_*_ranks` for rank files."""

    def __init__(self) -> None:
        self.counters: list[_RecordingCounter] = []
        self.rank_files: list[str] = []

    def __call__(self, tokenizer_json: str) -> _RecordingCounter:
        counter = _RecordingCounter(tokenizer_json)
        self.counters.append(counter)
        return counter

    def from_cl100k_ranks(self, rank_file: str) -> _RecordingCounter:
        self.rank_files.append(rank_file)
        return self("cl100k_base")

    def from_o200k_ranks(self, rank_file: str) -> _RecordingCounter:
        self.rank_files.append(rank_file)
        return self("o200k_base")


class _RaisingCounter:
    def __init__(self, error: Exception) -> None:
        self.error = error

    async def acount_request(self, body: bytes) -> object:
        raise self.error


class _RaisingFactory:
    """Every counter it builds, for either tokenizer, raises `error` on count."""

    def __init__(self, error: Exception) -> None:
        self.error = error

    def __call__(self, tokenizer_json: str) -> _RaisingCounter:
        return _RaisingCounter(self.error)

    def from_cl100k_ranks(self, rank_file: str) -> _RaisingCounter:
        return _RaisingCounter(self.error)

    def from_o200k_ranks(self, rank_file: str) -> _RaisingCounter:
        return _RaisingCounter(self.error)


@pytest.fixture(autouse=True)
def _reset_bridge(monkeypatch: pytest.MonkeyPatch):
    bridge.TOKEN_COUNTER.reset()
    bridge._counter.cache_clear()
    configuration.reset_rust_configuration()
    monkeypatch.setattr(bindings, "get_native_bridge", lambda: _FakeNative())
    yield
    bridge.TOKEN_COUNTER.reset()
    bridge._counter.cache_clear()
    configuration.reset_rust_configuration()


@pytest.mark.asyncio
@pytest.mark.parametrize("tokenizer", TOKENIZERS)
async def test_disabled_bridge_never_constructs_a_counter(tokenizer: bridge.RustTokenizer) -> None:
    factory: Final = _RecordingFactory()
    litellm.rust(False)
    bridge.TOKEN_COUNTER.override(factory)

    assert await bridge.count_input_tokens(BODY, tokenizer) is None
    assert factory.counters == []


@pytest.mark.asyncio
async def test_enabled_bridge_returns_typed_count_and_reuses_one_counter() -> None:
    factory: Final = _RecordingFactory()
    litellm.rust(True)
    bridge.TOKEN_COUNTER.override(factory)

    first: Final = await bridge.count_input_tokens(BODY, "anthropic")
    second: Final = await bridge.count_input_tokens(BODY, "anthropic")

    assert first == bridge.InputTokenCount(model=MODEL, input_tokens=42)
    assert second == first
    assert len(factory.counters) == 1
    assert factory.counters[0].bodies == [BODY, BODY]
    assert json.loads(factory.counters[0].tokenizer_json)["model"]["type"] == "BPE"


@pytest.mark.asyncio
@pytest.mark.parametrize("tokenizer", ("cl100k_base", "o200k_base"))
async def test_tiktoken_counter_is_built_from_the_vendored_rank_file_once(tokenizer: bridge.RustTokenizer) -> None:
    factory: Final = _RecordingFactory()
    litellm.rust(True)
    bridge.TOKEN_COUNTER.override(factory)

    first: Final = await bridge.count_input_tokens(BODY, tokenizer)
    second: Final = await bridge.count_input_tokens(BODY, tokenizer)

    assert first == second == bridge.InputTokenCount(model=MODEL, input_tokens=42)
    assert len(factory.rank_files) == 1
    assert factory.rank_files[0].startswith("IQ== 0\n")
    assert factory.rank_files[0].count("\n") == RANK_FILE_LINES[tokenizer]
    assert factory.counters[0].tokenizer_json == tokenizer
    assert factory.counters[0].bodies == [BODY, BODY]


@pytest.mark.asyncio
async def test_each_tokenizer_gets_its_own_cached_counter() -> None:
    factory: Final = _RecordingFactory()
    litellm.rust(True)
    bridge.TOKEN_COUNTER.override(factory)

    await bridge.count_input_tokens(BODY, "anthropic")
    await bridge.count_input_tokens(BODY, "cl100k_base")
    await bridge.count_input_tokens(BODY, "o200k_base")
    await bridge.count_input_tokens(BODY, "anthropic")
    await bridge.count_input_tokens(BODY, "o200k_base")

    assert [counter.tokenizer_json for counter in factory.counters][1:] == ["cl100k_base", "o200k_base"]
    assert [len(counter.bodies) for counter in factory.counters] == [2, 1, 2]


@pytest.mark.asyncio
async def test_missing_native_module_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    litellm.rust(True)
    monkeypatch.setattr(bindings, "get_native_bridge", lambda: None)

    assert [await bridge.count_input_tokens(BODY, tokenizer) for tokenizer in TOKENIZERS] == [None, None, None]


@pytest.mark.asyncio
@pytest.mark.parametrize("tokenizer", TOKENIZERS)
async def test_declined_request_falls_back(tokenizer: bridge.RustTokenizer) -> None:
    litellm.rust(True)
    bridge.TOKEN_COUNTER.override(_RaisingFactory(_FakeDeclined("request has no messages")))

    assert await bridge.count_input_tokens(BODY, tokenizer) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("tokenizer", TOKENIZERS)
async def test_runtime_failure_falls_back(tokenizer: bridge.RustTokenizer) -> None:
    litellm.rust(True)
    bridge.TOKEN_COUNTER.override(_RaisingFactory(RuntimeError("encode failed")))

    assert await bridge.count_input_tokens(BODY, tokenizer) is None


@pytest.mark.parametrize(
    ("model", "expected"),
    (
        (MODEL, "anthropic"),
        ("claude-3-5-sonnet-20241022", "cl100k_base"),
        ("gpt-4", "cl100k_base"),
        ("gpt-4-turbo", "cl100k_base"),
        ("gpt-3.5-turbo", "cl100k_base"),
        ("azure/gpt-35-turbo", "cl100k_base"),
        ("gemini/gemini-2.5-pro", "cl100k_base"),
        ("mistral/mistral-large-latest", "cl100k_base"),
        ("my-router-alias", "cl100k_base"),
        ("azure/gpt-4o", "cl100k_base"),
        ("command-r-plus", "cl100k_base"),
        ("gpt-4o", "o200k_base"),
        ("gpt-4o-mini", "o200k_base"),
        ("gpt-4o-2024-08-06", "o200k_base"),
        ("chatgpt-4o-latest", "o200k_base"),
        ("gpt-4.1", "o200k_base"),
        ("gpt-5", "o200k_base"),
        ("gpt-5-mini", "o200k_base"),
        ("o1", "o200k_base"),
        ("o3", "o200k_base"),
        ("o3-mini", "o200k_base"),
        ("o4-mini", "o200k_base"),
        ("replicate/meta/llama-2-70b-chat", None),
        ("meta-llama/Llama-3-8b", None),
    ),
)
def test_rust_tokenizer_mirrors_python_tokenizer_selection(model: str, expected: bridge.RustTokenizer | None) -> None:
    assert bridge.rust_tokenizer(model) == expected


@pytest.mark.parametrize(
    ("model", "python_encoding"),
    (("text-davinci-003", "p50k_base"), ("gpt-oss-120b", "o200k_harmony")),
)
def test_rust_tokenizer_declines_tiktoken_encodings_rust_does_not_have(
    monkeypatch: pytest.MonkeyPatch, model: str, python_encoding: str
) -> None:
    monkeypatch.setattr(litellm, "open_ai_chat_completion_models", litellm.open_ai_chat_completion_models | {model})

    assert openai_tokenizer_encoding(model).name == python_encoding
    assert bridge.rust_tokenizer(model) is None


def test_rust_tokenizer_declines_the_cohere_tokenizer_download(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "cohere_models", litellm.cohere_models | {"command-r-plus"})

    assert bridge.rust_tokenizer("command-r-plus") is None


@pytest.mark.parametrize("legacy_model", ("gpt-3.5-turbo-0301", "gpt-35-turbo-0301"))
def test_rust_tokenizer_declines_legacy_message_accounting_python_prices_differently(
    monkeypatch: pytest.MonkeyPatch, legacy_model: str
) -> None:
    monkeypatch.setattr(
        litellm, "open_ai_chat_completion_models", litellm.open_ai_chat_completion_models | {"gpt-3.5-turbo-0301"}
    )
    monkeypatch.setattr(litellm, "azure_llms", {**litellm.azure_llms, "gpt-35-turbo-0301": "azure"})
    messages: Final = [{"role": "user", "name": "bob", "content": "hello there"}]

    assert litellm.token_counter(model=legacy_model, messages=messages) != litellm.token_counter(
        model=CL100K_MODEL, messages=messages
    )
    assert bridge.rust_tokenizer(legacy_model) is None
    assert bridge.rust_tokenizer(CL100K_MODEL) == "cl100k_base"


@pytest.mark.parametrize("model", (MODEL, CL100K_MODEL, O200K_MODEL, "gpt-5", "o3"))
def test_rust_tokenizer_names_the_encoding_python_actually_counts_with(model: str) -> None:
    text: Final = (
        "Hello, world! camelCase ABCdef \u00e9\u00e8 12345 \u3053\u3093\u306b\u3061\u306f <|endoftext|>\r\n" * 9
    )
    python_count: Final = litellm.token_counter(model=model, text=text)
    cl100k_count: Final = len(tiktoken.get_encoding("cl100k_base").encode(text, disallowed_special=()))
    o200k_count: Final = len(tiktoken.get_encoding("o200k_base").encode(text, disallowed_special=()))
    assert cl100k_count != o200k_count
    match bridge.rust_tokenizer(model):
        case "cl100k_base":
            assert python_count == cl100k_count
        case "o200k_base":
            assert python_count == o200k_count
        case "anthropic":
            assert python_count == len(Tokenizer.from_str(claude_json_str).encode(text).ids)
            assert python_count not in {cl100k_count, o200k_count}
        case None:
            pytest.fail(f"{model} must have a Rust tokenizer")


def test_disabled_hf_download_routes_anthropic_models_to_cl100k_like_python(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "disable_hf_tokenizer_download", True)

    assert bridge.rust_tokenizer(MODEL) == "cl100k_base"
    assert bridge.rust_tokenizer("meta-llama/Llama-3-8b") == "cl100k_base"
    assert bridge.rust_tokenizer(O200K_MODEL) == "o200k_base"


def test_disabled_token_counter_declines_every_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "disable_token_counter", True)

    assert bridge.rust_tokenizer(MODEL) is None
    assert bridge.rust_tokenizer(CL100K_MODEL) is None
    assert bridge.rust_tokenizer(O200K_MODEL) is None


PARITY_REQUESTS: Final[tuple[dict[str, object], ...]] = (
    {"model": MODEL, "messages": [{"role": "user", "content": "Hello, how are you today?"}]},
    {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You are terse."},
            {"role": "user", "name": "bob", "content": [{"type": "text", "text": "Summarize this."}]},
            {"role": "assistant", "content": "Sure."},
        ],
    },
    {
        "model": MODEL,
        "messages": [{"role": "user", "content": "weather in sf?"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string", "description": "City"},
                            "unit": {"type": "string", "enum": ["c", "f"]},
                        },
                        "required": ["city"],
                    },
                },
            }
        ],
        "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
    },
    {
        "model": MODEL,
        "messages": [{"role": "user", "content": "x " * 500}],
    },
    {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": "I'VE got 1234567 things; it's \"fine\"...\r\n\r\n  caf\u00e9 \u0645\u0631\u062d\u0628\u0627 \U0001f600 <|endoftext|>",
            }
        ],
    },
    {"model": MODEL, "prompt": "Write a haiku about ships.", "max_tokens": 20},
    {"model": MODEL, "prompt": ["first prompt", "second prompt"]},
    {
        "model": MODEL,
        "instructions": "be terse",
        "input": [
            {"role": "user", "content": [{"type": "input_text", "text": 'Summarise caf\u00e9 menus \u2014 "ok"?\n'}]},
            {"role": "assistant", "content": "Sure."},
        ],
    },
    {"model": MODEL, "input": "a single embedding string"},
    {"model": MODEL, "input": [[101, 2023, 5], [7]], "encoding_format": "float"},
    {"model": MODEL, "query": "best harbour", "documents": ["doc one", {"text": "doc two", "title": "T", "n": 3}]},
    {"model": MODEL, "messages": None, "prompt": "messages key wins even when null"},
    {"prompt": "model comes from the route"},
)


PARITY_MODELS: Final[tuple[tuple[str, bridge.RustTokenizer], ...]] = (
    (MODEL, "anthropic"),
    (CL100K_MODEL, "cl100k_base"),
    (O200K_MODEL, "o200k_base"),
    ("gpt-5", "o200k_base"),
)


@pytest.mark.asyncio
@pytest.mark.parametrize(("model", "tokenizer"), PARITY_MODELS)
@pytest.mark.parametrize("request_body", PARITY_REQUESTS)
async def test_native_count_matches_python_budget_counter(
    monkeypatch: pytest.MonkeyPatch, request_body: dict[str, object], model: str, tokenizer: bridge.RustTokenizer
) -> None:
    native: Final = pytest.importorskip("litellm.rust_bridge._native")
    monkeypatch.setattr(bindings, "get_native_bridge", lambda: native)
    litellm.rust(True)
    body: Final = json.dumps(request_body).replace(MODEL, model)

    rust_count: Final = await bridge.count_input_tokens(body.encode(), tokenizer)
    python_count: Final = _count_input_tokens(request_body=json.loads(body), model=model)

    assert rust_count is not None
    assert rust_count.model == json.loads(body).get("model")
    assert rust_count.input_tokens == python_count


@pytest.mark.asyncio
@pytest.mark.parametrize(("model", "tokenizer"), ((CL100K_MODEL, "cl100k_base"), (O200K_MODEL, "o200k_base")))
async def test_tiktoken_counts_long_text_exactly_where_python_chunks(
    monkeypatch: pytest.MonkeyPatch, model: str, tokenizer: bridge.RustTokenizer
) -> None:
    """Python encodes tiktoken text in fixed-size chunks (drift of up to one token per chunk boundary); Rust does not."""
    native: Final = pytest.importorskip("litellm.rust_bridge._native")
    monkeypatch.setattr(bindings, "get_native_bridge", lambda: native)
    litellm.rust(True)
    text: Final = "x " * 20_000
    body: Final = {"model": model, "messages": [{"role": "user", "content": text}]}
    encoding: Final = tiktoken.get_encoding(tokenizer)
    exact: Final = 3 + len(encoding.encode("user")) + len(encoding.encode(text)) + 3
    chunks: Final = -(-len(text) // TIKTOKEN_ENCODE_CHUNK_SIZE_CHARS)

    rust_count: Final = await bridge.count_input_tokens(json.dumps(body).encode(), tokenizer)
    python_count: Final = _count_input_tokens(request_body=body, model=model)

    assert rust_count is not None
    assert rust_count.input_tokens == exact
    assert python_count is not None
    assert exact < python_count <= exact + chunks


DECLINED_REQUESTS: Final[tuple[dict[str, object], ...]] = (
    {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}}]}
        ],
    },
    {"model": MODEL, "prompt": 1.5},
    {"model": MODEL, "documents": [{"score": 0.5}]},
    {"model": MODEL, "file": "audio.mp3"},
)


@pytest.mark.asyncio
@pytest.mark.parametrize("tokenizer", TOKENIZERS)
@pytest.mark.parametrize("request_body", DECLINED_REQUESTS)
async def test_native_declines_shapes_python_prices_differently(
    monkeypatch: pytest.MonkeyPatch, request_body: dict[str, object], tokenizer: bridge.RustTokenizer
) -> None:
    native: Final = pytest.importorskip("litellm.rust_bridge._native")
    monkeypatch.setattr(bindings, "get_native_bridge", lambda: native)
    litellm.rust(True)

    assert await bridge.count_input_tokens(json.dumps(request_body).encode(), tokenizer) is None
