import asyncio
import base64
import contextlib
import hashlib
import json
import logging
import re
import threading
from collections.abc import Iterable
from itertools import product
from pathlib import Path
from typing import Final

import httpx
import pytest
import respx
import tiktoken

import litellm
from litellm._logging import verbose_logger
from litellm.llms.bedrock.count_tokens.kimi_k3 import (
    HUGGING_FACE_SOURCE,
    KIMI_K3_RANKS_FILE,
    KIMI_K3_REVISION,
    KimiK3Encodings,
    KimiK3TokenizerSource,
    kimi_k3_anthropic_tokens,
    kimi_k3_tokens,
    load_kimi_k3_encoding,
)
from litellm.responses.litellm_completion_transformation.transformation import LiteLLMCompletionResponsesConfig

MODEL: Final = "us.moonshotai.kimi-k3"
_CONTROL: Final = re.compile(r"<\|(?:open|close|sep|end_of_msg)\|>")
_WEATHER: Final = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the weather in °C",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
            "required": ["city"],
        },
    },
}


def _call(call_id: str, arguments: dict[str, object]) -> dict[str, object]:
    return {"id": call_id, "type": "function", "function": {"name": "get_weather", "arguments": json.dumps(arguments)}}


def _written_ranks(path: Path, tokens: Iterable[bytes]) -> KimiK3TokenizerSource:
    path.write_bytes(b"\n".join(base64.b64encode(token) + b" %d" % rank for rank, token in enumerate(tokens)))
    return KimiK3TokenizerSource(ranks_path=lambda: str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def _single_bytes() -> tuple[bytes, ...]:
    return tuple(bytes((value,)) for value in range(256))


@pytest.fixture(scope="module")
def byte_source(tmp_path_factory: pytest.TempPathFactory) -> KimiK3TokenizerSource:
    return _written_ranks(tmp_path_factory.mktemp("bytes") / "tiktoken.model", _single_bytes())


@pytest.fixture(scope="module")
def byte_encoding(byte_source: KimiK3TokenizerSource) -> tiktoken.Encoding:
    loaded: Final = load_kimi_k3_encoding(byte_source)
    assert loaded is not None
    return loaded


@pytest.fixture(scope="module")
def pair_encoding(tmp_path_factory: pytest.TempPathFactory) -> tiktoken.Encoding:
    pairs: Final = (bytes(pair) for pair in product(range(256), repeat=2))
    source: Final = _written_ranks(tmp_path_factory.mktemp("pairs") / "tiktoken.model", (*_single_bytes(), *pairs))
    loaded: Final = load_kimi_k3_encoding(source)
    assert loaded is not None
    return loaded


def _template_count(encoding: tiktoken.Encoding, rendered: str) -> int:
    texts: Final = _CONTROL.split(rendered)
    return len(_CONTROL.findall(rendered)) + sum(len(encoding.encode_ordinary(text)) for text in texts)


_GENERATION_PROMPT: Final = '<|open|>message role="assistant"<|sep|><|open|>think<|sep|>'


# Expected renders follow the Kimi K3 chat template in moonshotai/Kimi-K3 at revision f831ab6, which matched
# Bedrock's prompt_tokens for us.moonshotai.kimi-k3 in 19 live calls on 2026-10-16
@pytest.mark.parametrize(
    ("messages", "tools", "rendered"),
    [
        pytest.param(
            [{"role": "user", "content": "Hello there"}],
            None,
            '<|open|>message role="user"<|sep|>Hello there<|close|>message<|sep|><|end_of_msg|>' + _GENERATION_PROMPT,
            id="user",
        ),
        pytest.param(
            [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "Be terse."}, {"type": "text", "text": "Answer in French."}],
                },
                {"role": "user", "content": [{"type": "text", "text": "Hi"}, {"type": "text", "text": " there"}]},
            ],
            None,
            '<|open|>message role="system"<|sep|>Be terse.Answer in French.<|close|>message<|sep|><|end_of_msg|>'
            '<|open|>message role="user"<|sep|>Hi there<|close|>message<|sep|><|end_of_msg|>' + _GENERATION_PROMPT,
            id="text_parts_join",
        ),
        pytest.param(
            [
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "4"},
                {"role": "user", "content": "And 3+3?"},
            ],
            None,
            '<|open|>message role="user"<|sep|>What is 2+2?<|close|>message<|sep|><|end_of_msg|>'
            '<|open|>message role="assistant"<|sep|><|open|>think<|sep|><|close|>think<|sep|>'
            "<|open|>response<|sep|>4<|close|>response<|sep|><|close|>message<|sep|><|end_of_msg|>"
            '<|open|>message role="user"<|sep|>And 3+3?<|close|>message<|sep|><|end_of_msg|>' + _GENERATION_PROMPT,
            id="assistant_turn",
        ),
        pytest.param(
            [
                {"role": "user", "content": "Weather in Paris and Oslo?"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [_call("c1", {"city": "Paris", "days": 3}), _call("c2", {"city": "Oslo"})],
                },
                {"role": "tool", "tool_call_id": "c2", "content": "Snow"},
                {"role": "tool", "tool_call_id": "c1", "content": "Sun"},
            ],
            [_WEATHER],
            '<|open|>message role="system" type="tool-declare"<|sep|># Tools\n'
            "Here are the available tools, described in JSONSchema.\n\n```json\n"
            '[{"function":{"description":"Get the weather in °C","name":"get_weather","parameters":{"properties":'
            '{"city":{"type":"string"},"days":{"type":"integer"}},"required":["city"],"type":"object"}},'
            '"type":"function"}]\n```<|close|>message<|sep|><|end_of_msg|>'
            '<|open|>message role="user"<|sep|>Weather in Paris and Oslo?<|close|>message<|sep|><|end_of_msg|>'
            '<|open|>message role="assistant"<|sep|><|open|>think<|sep|><|close|>think<|sep|>'
            "<|open|>response<|sep|><|close|>response<|sep|><|open|>tools<|sep|>"
            '<|open|>call tool="get_weather" index="1"<|sep|>'
            '<|open|>argument key="city" type="string"<|sep|>Paris<|close|>argument<|sep|>'
            '<|open|>argument key="days" type="number"<|sep|>3<|close|>argument<|sep|><|close|>call<|sep|>'
            '<|open|>call tool="get_weather" index="2"<|sep|>'
            '<|open|>argument key="city" type="string"<|sep|>Oslo<|close|>argument<|sep|><|close|>call<|sep|>'
            "<|close|>tools<|sep|><|close|>message<|sep|><|end_of_msg|>"
            '<|open|>message role="tool" tool="get_weather" index="1"<|sep|>Sun<|close|>message<|sep|><|end_of_msg|>'
            '<|open|>message role="tool" tool="get_weather" index="2"<|sep|>Snow<|close|>message<|sep|><|end_of_msg|>'
            + _GENERATION_PROMPT,
            id="tool_results_out_of_call_order",
        ),
        pytest.param(
            [
                {"role": "user", "content": "Weather?"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        _call(
                            "c1", {"a&b": 'x"y', "flag": True, "none": None, "list": [1, "zé"], "object": {"k": True}}
                        )
                    ],
                },
                {"role": "tool", "tool_call_id": "c1", "content": "Rain"},
            ],
            [_WEATHER],
            '<|open|>message role="system" type="tool-declare"<|sep|># Tools\n'
            "Here are the available tools, described in JSONSchema.\n\n```json\n"
            '[{"function":{"description":"Get the weather in °C","name":"get_weather","parameters":{"properties":'
            '{"city":{"type":"string"},"days":{"type":"integer"}},"required":["city"],"type":"object"}},'
            '"type":"function"}]\n```<|close|>message<|sep|><|end_of_msg|>'
            '<|open|>message role="user"<|sep|>Weather?<|close|>message<|sep|><|end_of_msg|>'
            '<|open|>message role="assistant"<|sep|><|open|>think<|sep|><|close|>think<|sep|>'
            "<|open|>response<|sep|><|close|>response<|sep|><|open|>tools<|sep|>"
            '<|open|>call tool="get_weather" index="1"<|sep|>'
            '<|open|>argument key="a&amp;b" type="string"<|sep|>x"y<|close|>argument<|sep|>'
            '<|open|>argument key="flag" type="boolean"<|sep|>true<|close|>argument<|sep|>'
            '<|open|>argument key="none" type="null"<|sep|>null<|close|>argument<|sep|>'
            '<|open|>argument key="list" type="array"<|sep|>[1, "zé"]<|close|>argument<|sep|>'
            '<|open|>argument key="object" type="object"<|sep|>{"k": true}<|close|>argument<|sep|>'
            "<|close|>call<|sep|><|close|>tools<|sep|><|close|>message<|sep|><|end_of_msg|>"
            '<|open|>message role="tool" tool="get_weather" index="1"<|sep|>Rain<|close|>message<|sep|><|end_of_msg|>'
            + _GENERATION_PROMPT,
            id="argument_types_and_escaping",
        ),
    ],
)
@pytest.mark.parametrize("encoding_fixture", ["byte_encoding", "pair_encoding"])
def test_count_is_the_kimi_k3_chat_template_render_of_the_bedrock_request(
    request: pytest.FixtureRequest,
    encoding_fixture: str,
    messages: list[dict[str, object]],
    tools: list[dict[str, object]] | None,
    rendered: str,
):
    encoding: Final[tiktoken.Encoding] = request.getfixturevalue(encoding_fixture)

    assert kimi_k3_tokens(encoding, MODEL, messages, tools) == _template_count(encoding, rendered)


def _parallel_calls(count: int) -> list[dict[str, object]]:
    ids: Final = tuple(f"c{number}" for number in range(1, count + 1))
    return [
        {"role": "user", "content": "Weather?"},
        {"role": "assistant", "content": None, "tool_calls": [_call(call_id, {"city": "Paris"}) for call_id in ids]},
        *({"role": "tool", "tool_call_id": call_id, "content": "Sun"} for call_id in ids),
    ]


def test_tenth_parallel_call_and_its_result_are_numbered_ten(byte_encoding: tiktoken.Encoding):
    nine: Final = kimi_k3_tokens(byte_encoding, MODEL, _parallel_calls(9), [_WEATHER])
    ten: Final = kimi_k3_tokens(byte_encoding, MODEL, _parallel_calls(10), [_WEATHER])
    assert nine is not None and ten is not None

    assert ten - nine == _template_count(
        byte_encoding,
        '<|open|>call tool="get_weather" index="10"<|sep|>'
        '<|open|>argument key="city" type="string"<|sep|>Paris<|close|>argument<|sep|><|close|>call<|sep|>'
        '<|open|>message role="tool" tool="get_weather" index="10"<|sep|>Sun<|close|>message<|sep|><|end_of_msg|>',
    )


@pytest.mark.parametrize(
    "text",
    [
        pytest.param("The quick brown fox, jumps over 12345 lazy dogs. " * 60, id="prose"),
        pytest.param("a" + "1234567890" * 250 + "1", id="long_digit_run"),
        pytest.param("--- " * 600, id="symbols"),
        pytest.param("def f(x):\n    return x  # it's 42\n\n" * 80, id="code"),
        pytest.param("word   next\t\ttab\n \n" * 120, id="whitespace"),
        pytest.param("naïve café, “quotes” & <tags> 中文汉字，日本語のテキスト 😀🎉\r\n" * 40, id="unicode"),
        pytest.param("We're sure they'd say it's DON'T or I'M " * 60, id="contractions"),
        pytest.param("abc123def4567ghi89 " * 120, id="letters_and_digits"),
    ],
)
def test_long_text_counts_what_the_tokenizer_counts_for_the_whole_text(pair_encoding: tiktoken.Encoding, text: str):
    def counted(content: str) -> int | None:
        return kimi_k3_tokens(pair_encoding, MODEL, [{"role": "user", "content": content}], None)

    short: Final = counted("x")
    assert short is not None

    assert counted(text) == short - len(pair_encoding.encode_ordinary("x")) + len(pair_encoding.encode_ordinary(text))


@pytest.mark.parametrize(
    ("messages", "tools"),
    [
        pytest.param(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is this?"},
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}},
                    ],
                }
            ],
            None,
            id="image",
        ),
        pytest.param(
            [{"role": "user", "content": "Weather?"}, {"role": "tool", "tool_call_id": "c1", "content": "Sun"}],
            [_WEATHER],
            id="tool_result_without_its_call",
        ),
    ],
)
def test_request_the_template_cannot_render_is_not_counted(
    pair_encoding: tiktoken.Encoding, messages: list[dict[str, object]], tools: list[dict[str, object]] | None
):
    assert kimi_k3_tokens(pair_encoding, MODEL, messages, tools) is None


@pytest.mark.parametrize(
    "tool",
    [
        pytest.param(
            {
                "type": "function",
                "name": "get_weather",
                "description": "Get the weather in °C",
                "parameters": _WEATHER["function"]["parameters"],
            },
            id="responses_api",
        ),
        pytest.param(
            {
                "name": "get_weather",
                "description": "Get the weather in °C",
                "input_schema": _WEATHER["function"]["parameters"],
            },
            id="anthropic_messages",
        ),
    ],
)
def test_tool_in_another_api_shape_counts_like_the_same_chat_completions_tool(
    pair_encoding: tiktoken.Encoding, tool: dict[str, object]
):
    messages: Final = [{"role": "user", "content": "Weather in Paris?"}]

    counted: Final = kimi_k3_tokens(pair_encoding, MODEL, messages, [tool])

    assert counted is not None
    assert counted == kimi_k3_tokens(pair_encoding, MODEL, messages, [_WEATHER])


@pytest.mark.parametrize(
    "tool",
    [
        pytest.param({"type": "web_search_20250305", "name": "web_search"}, id="anthropic_web_search"),
        pytest.param({"type": "web_search"}, id="responses_web_search"),
        pytest.param(
            {
                "type": "mcp",
                "server_label": "docs",
                "server_url": "https://mcp.example/mcp",
                "require_approval": "never",
            },
            id="mcp",
        ),
    ],
)
def test_tool_bedrock_inference_drops_counts_like_the_function_tools_alone(
    pair_encoding: tiktoken.Encoding, tool: dict[str, object]
):
    messages: Final = [{"role": "user", "content": "Weather in Paris?"}]

    counted: Final = kimi_k3_tokens(pair_encoding, MODEL, messages, [_WEATHER, tool])

    assert counted is not None
    assert counted == kimi_k3_tokens(pair_encoding, MODEL, messages, [_WEATHER])


def test_responses_custom_tool_counts_like_the_function_tool_inference_sends(pair_encoding: tiktoken.Encoding):
    messages: Final = [{"role": "user", "content": "Weather in Paris?"}]
    custom_tool: Final = {"type": "custom", "name": "grammar", "description": "Free-form SQL"}
    sent_to_inference, _ = LiteLLMCompletionResponsesConfig.transform_responses_api_tools_to_chat_completion_tools(
        [dict(custom_tool)]
    )

    without_it: Final = kimi_k3_tokens(pair_encoding, MODEL, messages, [_WEATHER])

    counted: Final = kimi_k3_tokens(pair_encoding, MODEL, messages, [_WEATHER, custom_tool])

    assert counted is not None and without_it is not None
    assert counted > without_it
    assert counted == kimi_k3_tokens(pair_encoding, MODEL, messages, [_WEATHER, *sent_to_inference])


def test_counting_a_request_with_an_image_url_never_downloads_the_image(
    pair_encoding: tiktoken.Encoding, respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(litellm, "user_url_validation", False)
    image: Final = respx_mock.get("https://images.example/cat.png").mock(return_value=httpx.Response(200))
    messages: Final = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is this?"},
                {"type": "image_url", "image_url": {"url": "https://images.example/cat.png"}},
            ],
        }
    ]

    assert kimi_k3_tokens(pair_encoding, MODEL, messages, None) is None
    assert not image.called


def test_anthropic_request_counts_like_the_same_openai_request(pair_encoding: tiktoken.Encoding):
    anthropic: Final = kimi_k3_anthropic_tokens(
        pair_encoding,
        MODEL,
        {
            "system": "Be terse.",
            "messages": [
                {"role": "user", "content": "Weather in Paris?"},
                {
                    "role": "assistant",
                    "content": [{"type": "tool_use", "id": "c1", "name": "get_weather", "input": {"city": "Paris"}}],
                },
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "c1", "content": "Sun"}]},
            ],
            "tools": [
                {
                    "name": "get_weather",
                    "description": "Get the weather in °C",
                    "input_schema": _WEATHER["function"]["parameters"],
                }
            ],
        },
    )
    openai: Final = kimi_k3_tokens(
        pair_encoding,
        MODEL,
        [
            {"role": "system", "content": "Be terse."},
            {"role": "user", "content": "Weather in Paris?"},
            {"role": "assistant", "content": None, "tool_calls": [_call("c1", {"city": "Paris"})]},
            {"role": "tool", "tool_call_id": "c1", "content": "Sun"},
        ],
        [_WEATHER],
    )

    assert openai is not None
    assert anthropic == openai


@pytest.mark.parametrize("failure", ["checksum", "missing_file"])
def test_tokenizer_that_fails_its_checksum_or_cannot_be_read_is_not_loaded(
    byte_source: KimiK3TokenizerSource, tmp_path: Path, caplog: pytest.LogCaptureFixture, failure: str
):
    source: Final = (
        KimiK3TokenizerSource(ranks_path=byte_source.ranks_path, sha256=hashlib.sha256(b"other").hexdigest())
        if failure == "checksum"
        else KimiK3TokenizerSource(ranks_path=lambda: str(tmp_path / "missing"), sha256=byte_source.sha256)
    )
    verbose_logger.addHandler(caplog.handler)
    try:
        with caplog.at_level(logging.WARNING, logger=verbose_logger.name):
            loaded: Final = load_kimi_k3_encoding(source)
    finally:
        verbose_logger.removeHandler(caplog.handler)

    assert loaded is None
    assert any("Kimi K3 tokenizer unavailable" in record.getMessage() for record in caplog.records)


@pytest.mark.asyncio
async def test_slow_tokenizer_load_is_not_waited_on_past_the_wait_and_is_shared_once_done(
    byte_source: KimiK3TokenizerSource,
):
    released: Final = threading.Event()

    def slow_ranks_path() -> str:
        released.wait(10)
        return byte_source.ranks_path()

    encodings: Final = KimiK3Encodings(
        source=KimiK3TokenizerSource(ranks_path=slow_ranks_path, sha256=byte_source.sha256),
        retry_after_seconds=60,
        clock=lambda: 0.0,
    )

    assert await encodings.encoding(wait_seconds=0) is None
    released.set()
    loaded: Final = await encodings.encoding(wait_seconds=5)
    assert loaded is not None
    assert await encodings.encoding(wait_seconds=0) is loaded


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


async def _loaded_eventually(encodings: KimiK3Encodings) -> tiktoken.Encoding | None:
    for _ in range(500):
        if (loaded := await encodings.encoding(wait_seconds=0)) is not None:
            return loaded
        await asyncio.sleep(0.01)
    return None


@pytest.mark.asyncio
async def test_failed_tokenizer_load_is_retried_in_the_background_only_after_the_retry_window(
    byte_source: KimiK3TokenizerSource, tmp_path: Path
):
    clock: Final = _Clock()
    reads: Final[list[float]] = []
    released: Final = threading.Event()

    def ranks_path() -> str:
        reads.append(clock.now)
        if len(reads) == 1:
            return str(tmp_path / "missing")
        released.wait(10)
        return byte_source.ranks_path()

    encodings: Final = KimiK3Encodings(
        source=KimiK3TokenizerSource(ranks_path=ranks_path, sha256=byte_source.sha256),
        retry_after_seconds=60,
        clock=clock,
    )

    assert await encodings.encoding(wait_seconds=5) is None
    clock.now = 59.0
    assert await encodings.encoding(wait_seconds=5) is None
    clock.now = 60.0
    assert await asyncio.wait_for(encodings.encoding(wait_seconds=5), timeout=1) is None
    released.set()
    assert await _loaded_eventually(encodings) is not None
    assert reads == [0.0, 60.0]


@pytest.mark.asyncio
async def test_tokenizer_load_still_running_after_the_wait_is_logged(
    byte_source: KimiK3TokenizerSource, caplog: pytest.LogCaptureFixture
):
    released: Final = threading.Event()

    def hung_ranks_path() -> str:
        released.wait(10)
        return byte_source.ranks_path()

    encodings: Final = KimiK3Encodings(
        source=KimiK3TokenizerSource(ranks_path=hung_ranks_path, sha256=byte_source.sha256), retry_after_seconds=60
    )
    verbose_logger.addHandler(caplog.handler)
    try:
        with caplog.at_level(logging.WARNING, logger=verbose_logger.name):
            loaded: Final = await encodings.encoding(wait_seconds=0.05)
    finally:
        verbose_logger.removeHandler(caplog.handler)
        released.set()

    assert loaded is None
    assert any("Kimi K3 tokenizer still loading" in record.getMessage() for record in caplog.records)


@pytest.mark.parametrize(
    ("offline_mode", "disable_download", "asks_hub"),
    [("true", None, False), ("false", True, False), ("false", None, True)],
)
def test_offline_mode_and_disabled_downloads_never_ask_hugging_face_for_the_tokenizer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, offline_mode: str, disable_download: bool | None, asks_hub: bool
):
    import huggingface_hub.constants
    import huggingface_hub.utils._detect_agent

    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(tmp_path / "hub"))
    monkeypatch.setattr(huggingface_hub.constants, "AGENT_HARNESSES_PATH", str(tmp_path / "agent_harnesses.json"))
    monkeypatch.setattr(huggingface_hub.utils._detect_agent, "_registry", None)
    monkeypatch.setenv("LITELLM_OFFLINE_MODE", offline_mode)
    monkeypatch.setattr(litellm, "disable_hf_tokenizer_download", disable_download)
    with respx.mock(assert_all_called=False) as router:
        hub: Final = router.route(host="huggingface.co").mock(return_value=httpx.Response(503))
        with contextlib.suppress(Exception):
            HUGGING_FACE_SOURCE.ranks_path()
    assert hub.called is asks_hub


def test_offline_mode_reads_the_tokenizer_from_the_hugging_face_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    import huggingface_hub.constants

    cached: Final = tmp_path / "models--moonshotai--Kimi-K3" / "snapshots" / KIMI_K3_REVISION / KIMI_K3_RANKS_FILE
    cached.parent.mkdir(parents=True)
    cached.write_bytes(b"cached ranks")
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(tmp_path))
    monkeypatch.setenv("LITELLM_OFFLINE_MODE", "true")
    with respx.mock(assert_all_called=False) as router:
        hub: Final = router.route(host="huggingface.co").mock(return_value=httpx.Response(503))
        path: Final = HUGGING_FACE_SOURCE.ranks_path()
    assert Path(path).read_bytes() == b"cached ranks"
    assert not hub.called
