"""Tests for the Rust input token counter bridge, called directly rather than through the route catalog.

The factory is passed into ``native_count`` so the caching cases run without the compiled extension
present. The parity cases need the extension and are skipped when it is not built.
"""

from __future__ import annotations

import json
from types import MappingProxyType
from typing import Final

import pytest

import litellm
from litellm.constants import TIKTOKEN_ENCODE_CHUNK_SIZE_CHARS
from litellm.litellm_core_utils.token_counter import openai_tokenizer_encoding
from litellm.proxy.spend_tracking.input_tokens import count_input_tokens_for_model
from litellm.rust_bridge import token_counter as bridge
from litellm.rust_bridge import tokenizer as tokenizer_dispatch
from litellm.rust_bridge._native import Tokenizer
from litellm.utils import claude_json_str

MODEL: Final = "claude-sonnet-4-5-20250929"
CL100K_MODEL: Final = "gpt-4"
O200K_MODEL: Final = "gpt-4o"
MODEL_BY_TOKENIZER: Final[MappingProxyType[bridge.RustTokenizer, str]] = MappingProxyType(
    {"anthropic": MODEL, "cl100k_base": CL100K_MODEL, "o200k_base": O200K_MODEL}
)
TOKENIZERS: Final[tuple[bridge.RustTokenizer, ...]] = ("anthropic", "cl100k_base", "o200k_base")
BODY: Final = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": "hello"}]}).encode()


def _counted(body: dict[str, object], model: str) -> tuple[bytes, dict[str, object]]:
    raw: Final = json.dumps({**body, "model": model}).encode()
    return raw, json.loads(raw)


class _FakeTokenizer:
    """Stands in for one shared native `Tokenizer`; only its name identifies it."""

    def __init__(self, name: str, json: str | None = None) -> None:
        self.name = name
        self.json = json


class _RecordingCounter:
    def __init__(self, tokenizer: _FakeTokenizer, fast: bool) -> None:
        self.tokenizer = tokenizer
        self.fast = fast
        self.bodies: list[bytes] = []

    async def acount_request(self, body: bytes) -> object:
        self.bodies.append(body)
        return {"model": MODEL, "input_tokens": 42}


class _RecordingFactory:
    """Stands in for the native `TokenCounter` class, built over a loaded `Tokenizer`."""

    def __init__(self) -> None:
        self.counters: list[_RecordingCounter] = []

    def from_tokenizer(self, tokenizer: _FakeTokenizer, fast: bool = False) -> _RecordingCounter:
        counter = _RecordingCounter(tokenizer, fast)
        self.counters.append(counter)
        return counter


@pytest.fixture(autouse=True)
def _reset_counters():
    bridge._counter.cache_clear()
    yield
    bridge._counter.cache_clear()


@pytest.fixture
def fake_tokenizers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the counter's tokenizer lookups at fakes so a recording factory sees which one it was built over."""
    fakes: Final = {name: _FakeTokenizer(name) for name in ("cl100k_base", "o200k_base")}
    anthropic: Final = _FakeTokenizer("anthropic", claude_json_str)
    monkeypatch.setattr(tokenizer_dispatch, "native_encoding", fakes.__getitem__)
    monkeypatch.setattr(tokenizer_dispatch, "native_anthropic", lambda: anthropic)


@pytest.mark.asyncio
async def test_native_count_returns_typed_count_and_reuses_one_counter(fake_tokenizers: None) -> None:
    factory: Final = _RecordingFactory()

    first: Final = await bridge.native_count(factory, "anthropic", BODY)
    second: Final = await bridge.native_count(factory, "anthropic", BODY)

    assert first == bridge.InputTokenCount(model=MODEL, input_tokens=42)
    assert second == first
    assert len(factory.counters) == 1
    assert factory.counters[0].bodies == [BODY, BODY]
    assert factory.counters[0].fast is False
    assert factory.counters[0].tokenizer is tokenizer_dispatch.native_anthropic()
    assert json.loads(factory.counters[0].tokenizer.json or "")["model"]["type"] == "BPE"


@pytest.mark.asyncio
@pytest.mark.parametrize("tokenizer", ("cl100k_base", "o200k_base"))
async def test_tiktoken_counter_is_built_over_the_shared_encoding_once(
    fake_tokenizers: None, tokenizer: bridge.RustTokenizer
) -> None:
    factory: Final = _RecordingFactory()

    first: Final = await bridge.native_count(factory, tokenizer, BODY)
    second: Final = await bridge.native_count(factory, tokenizer, BODY)

    assert first == second == bridge.InputTokenCount(model=MODEL, input_tokens=42)
    assert len(factory.counters) == 1
    assert factory.counters[0].tokenizer.name == tokenizer
    assert factory.counters[0].tokenizer is tokenizer_dispatch.native_encoding(tokenizer)
    assert factory.counters[0].fast is False
    assert factory.counters[0].bodies == [BODY, BODY]


@pytest.mark.asyncio
async def test_each_tokenizer_gets_its_own_cached_counter(fake_tokenizers: None) -> None:
    factory: Final = _RecordingFactory()

    await bridge.native_count(factory, "anthropic", BODY)
    await bridge.native_count(factory, "cl100k_base", BODY)
    await bridge.native_count(factory, "o200k_base", BODY)
    await bridge.native_count(factory, "anthropic", BODY)
    await bridge.native_count(factory, "o200k_base", BODY)

    assert [counter.tokenizer.name for counter in factory.counters] == ["anthropic", "cl100k_base", "o200k_base"]
    assert [len(counter.bodies) for counter in factory.counters] == [2, 1, 2]


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
    cl100k_count: Final = Tokenizer.from_tiktoken("cl100k_base").count(text)
    o200k_count: Final = Tokenizer.from_tiktoken("o200k_base").count(text)
    assert cl100k_count != o200k_count
    match bridge.rust_tokenizer(model):
        case "cl100k_base":
            assert python_count == cl100k_count
        case "o200k_base":
            assert python_count == o200k_count
        case "anthropic":
            assert python_count == Tokenizer.from_json(claude_json_str).count(text)
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
    request_body: dict[str, object], model: str, tokenizer: bridge.RustTokenizer
) -> None:
    native: Final = pytest.importorskip("litellm.rust_bridge._native")
    body: Final = json.dumps(request_body).replace(MODEL, model)
    parsed: Final = json.loads(body)

    counted: Final = await bridge.native_count(native.TokenCounter, tokenizer, body.encode())

    assert counted.input_tokens == count_input_tokens_for_model(request_body=parsed, model=model)


@pytest.mark.asyncio
@pytest.mark.parametrize(("model", "tokenizer"), ((CL100K_MODEL, "cl100k_base"), (O200K_MODEL, "o200k_base")))
async def test_tiktoken_counts_long_text_exactly_where_python_chunks(
    model: str, tokenizer: bridge.RustTokenizer
) -> None:
    """Python encodes tiktoken text in fixed-size chunks (drift of up to one token per chunk boundary); Rust does not."""
    native: Final = pytest.importorskip("litellm.rust_bridge._native")
    text: Final = "x " * 20_000
    body: Final = {"model": model, "messages": [{"role": "user", "content": text}]}
    encoding: Final = Tokenizer.from_tiktoken(tokenizer)
    exact: Final = 3 + encoding.count("user") + encoding.count(text) + 3
    chunks: Final = -(-len(text) // TIKTOKEN_ENCODE_CHUNK_SIZE_CHARS)

    counted: Final = await bridge.native_count(native.TokenCounter, tokenizer, json.dumps(body).encode())
    python_count: Final = count_input_tokens_for_model(request_body=body, model=model)

    assert counted.input_tokens == exact
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
    request_body: dict[str, object], tokenizer: bridge.RustTokenizer
) -> None:
    native: Final = pytest.importorskip("litellm.rust_bridge._native")
    raw, _ = _counted(request_body, MODEL_BY_TOKENIZER[tokenizer])

    with pytest.raises(native.RustBridgeDeclined):
        await bridge.native_count(native.TokenCounter, tokenizer, raw)
