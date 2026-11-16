import itertools
import json
import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Final
import pytest
from pydantic import TypeAdapter
import litellm
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
from litellm.types.utils import Delta, ModelResponseStream, StreamingChoices, Usage
from litellm.utils import ModelResponseListIterator
from litellm import completion


bedrock_chunks = [
    ModelResponseStream(
        id="chatcmpl-d249def8-a78b-464c-87b5-3a6f43565292",
        created=1742056047,
        model=None,
        object="chat.completion.chunk",
        system_fingerprint=None,
        choices=[
            StreamingChoices(
                finish_reason=None,
                index=0,
                delta=Delta(
                    provider_specific_fields=None,
                    content="I'm Claude",
                    role="assistant",
                    function_call=None,
                    tool_calls=None,
                    audio=None,
                ),
                logprobs=None,
            )
        ],
        provider_specific_fields={},
        usage=None,
    ),
    ModelResponseStream(
        id="chatcmpl-fe559823-b383-4249-ab87-52f6ad9d08c2",
        created=1742056047,
        model=None,
        object="chat.completion.chunk",
        system_fingerprint=None,
        choices=[
            StreamingChoices(
                finish_reason=None,
                index=0,
                delta=Delta(
                    provider_specific_fields=None,
                    content=", an AI",
                    role="assistant",
                    function_call=None,
                    tool_calls=None,
                    audio=None,
                ),
                logprobs=None,
            )
        ],
        provider_specific_fields={},
        usage=None,
    ),
    ModelResponseStream(
        id="chatcmpl-c1c6cc2f-75b9-4a24-88b9-4e5aacd0268b",
        created=1742056047,
        model=None,
        object="chat.completion.chunk",
        system_fingerprint=None,
        choices=[
            StreamingChoices(
                finish_reason="stop",
                index=0,
                delta=Delta(
                    provider_specific_fields=None,
                    content="",
                    role="assistant",
                    function_call=None,
                    tool_calls=None,
                    audio=None,
                ),
                logprobs=None,
            )
        ],
        provider_specific_fields={},
        usage=None,
    ),
]


FINISH_ONLY_LOGPROBS_STREAM_SCRIPT: Final = """
import json
import sys

sys.path.insert(0, sys.argv[1])

import httpx
from openai import OpenAI

import litellm

upstream_choices = (
    {"index": 0, "delta": {"role": "assistant", "content": "hi"}, "finish_reason": None},
    {
        "index": 0,
        "delta": {},
        "logprobs": {"content": [{"token": "<|return|>", "logprob": -0.5, "top_logprobs": []}]},
        "finish_reason": "stop",
    },
)
body = "".join(
    "data: "
    + json.dumps({"id": "c", "object": "chat.completion.chunk", "created": 1, "model": "m", "choices": [choice]})
    + "\\n\\n"
    for choice in upstream_choices
) + "data: [DONE]\\n\\n"
client = OpenAI(
    api_key="unused",
    base_url="https://provider.example/v1",
    http_client=httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})
        )
    ),
)
stream = litellm.completion(
    model="openai/m", messages=[{"role": "user", "content": "hi"}], stream=True, logprobs=True, client=client
)
print(json.dumps([json.loads(chunk.model_dump_json(exclude_none=True)) for chunk in stream]))
"""


def test_stream_with_logprobs_only_on_finish_chunk_serializes_in_a_fresh_process() -> None:
    result: Final = subprocess.run(
        [sys.executable, "-I", "-c", FINISH_ONLY_LOGPROBS_STREAM_SCRIPT, str(Path(litellm.__file__).parent.parent)],
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, "LITELLM_LOCAL_MODEL_COST_MAP": "True"},
    )

    assert result.returncode == 0, result.stdout + result.stderr
    wire_choices: Final = tuple(
        itertools.chain.from_iterable(chunk["choices"] for chunk in json.loads(result.stdout.strip().splitlines()[-1]))
    )
    finish_choices: Final = [choice for choice in wire_choices if choice.get("finish_reason")]
    assert [choice["finish_reason"] for choice in finish_choices] == ["stop"]
    assert [entry["token"] for entry in finish_choices[0]["logprobs"]["content"]] == ["<|return|>"]


_REPETITION_TEST_CASES = [
    # Basic cases
    pytest.param(
        ["same"] * litellm.REPEATED_STREAMING_CHUNK_LIMIT,
        True,
        id="all_identical_raises",
    ),
    pytest.param(
        ["same"] * (litellm.REPEATED_STREAMING_CHUNK_LIMIT - 1),
        False,
        id="below_threshold_no_raise",
    ),
    pytest.param(
        [None] * litellm.REPEATED_STREAMING_CHUNK_LIMIT,
        False,
        id="none_content_no_raise",
    ),
    pytest.param(
        [""] * litellm.REPEATED_STREAMING_CHUNK_LIMIT,
        False,
        id="empty_content_no_raise",
    ),
    # Short content (len <= 2) should not raise
    pytest.param(
        ["##"] * litellm.REPEATED_STREAMING_CHUNK_LIMIT,
        False,
        id="short_content_2chars_no_raise",
    ),
    pytest.param(
        ["{"] * litellm.REPEATED_STREAMING_CHUNK_LIMIT,
        False,
        id="short_content_1char_no_raise",
    ),
    pytest.param(
        ["ab"] * litellm.REPEATED_STREAMING_CHUNK_LIMIT,
        False,
        id="short_content_2chars_ab_no_raise",
    ),
    # All different chunks
    pytest.param(
        ["diff"] * litellm.REPEATED_STREAMING_CHUNK_LIMIT,
        False,
        id="all_different_no_raise",
    ),
    # One chunk different at various positions
    pytest.param(
        ["different_first"] + ["same"] * (litellm.REPEATED_STREAMING_CHUNK_LIMIT - 1),
        False,
        id="first_chunk_different_no_raise",
    ),
    pytest.param(
        ["same"] * (litellm.REPEATED_STREAMING_CHUNK_LIMIT - 1) + ["different_last"],
        False,
        id="last_chunk_different_no_raise",
    ),
    pytest.param(
        ["same"] * (litellm.REPEATED_STREAMING_CHUNK_LIMIT // 2 + 1)
        + ["different_mid"]
        + ["same"]
        * (
            litellm.REPEATED_STREAMING_CHUNK_LIMIT
            - litellm.REPEATED_STREAMING_CHUNK_LIMIT // 2
            + 1
        ),
        False,
        id="middle_chunk_different_no_raise",
    ),
    pytest.param(
        ["same"] * (litellm.REPEATED_STREAMING_CHUNK_LIMIT - 2) + ["diff", "diff"],
        False,
        id="last_two_different_no_raise",
    ),
    pytest.param(
        ["diff"] * litellm.REPEATED_STREAMING_CHUNK_LIMIT
        + ["same"] * litellm.REPEATED_STREAMING_CHUNK_LIMIT
        + ["diff"],
        True,
        id="in_between_same_and_diff_raise",
    ),
]


# Azure streaming chunks that reproduce issue #24221:
# Azure sends an initial chunk with prompt_filter_results and choices=[],
# then a chunk with role='assistant' and content='', then content chunks.
# With stream_options.include_usage=True, the empty-choices chunk was
# forwarded with an inflated default choice, consuming the sent_first_chunk
# flag and causing strip_role_from_delta to strip the role from the real
# first chunk.
_AZURE_CHUNKS_WITH_PROMPT_FILTER = [
    # Chunk 1: prompt_filter_results, no choices (Azure-specific)
    ModelResponseStream(
        id="chatcmpl-abc123",
        created=1742056047,
        model=None,
        object="chat.completion.chunk",
        choices=[],
        usage=None,
    ),
    # Chunk 2: first real chunk with role='assistant' and empty content
    ModelResponseStream(
        id="chatcmpl-abc123",
        created=1742056047,
        model=None,
        object="chat.completion.chunk",
        choices=[
            StreamingChoices(
                finish_reason=None,
                index=0,
                delta=Delta(content="", role="assistant"),
            )
        ],
        usage=None,
    ),
    # Chunk 3: content
    ModelResponseStream(
        id="chatcmpl-abc123",
        created=1742056047,
        model=None,
        object="chat.completion.chunk",
        choices=[
            StreamingChoices(
                finish_reason=None,
                index=0,
                delta=Delta(content="Hello!"),
            )
        ],
        usage=None,
    ),
    # Chunk 4: finish_reason
    ModelResponseStream(
        id="chatcmpl-abc123",
        created=1742056047,
        model=None,
        object="chat.completion.chunk",
        choices=[
            StreamingChoices(
                finish_reason="stop",
                index=0,
                delta=Delta(),
            )
        ],
        usage=None,
    ),
    # Chunk 5: final usage chunk, no choices
    ModelResponseStream(
        id="chatcmpl-abc123",
        created=1742056047,
        model=None,
        object="chat.completion.chunk",
        choices=[],
        usage=Usage(
            completion_tokens=10,
            prompt_tokens=20,
            total_tokens=30,
        ),
    ),
]


_AZURE_CHOICE_EXTRAS: Final = MappingProxyType(
    {"content_filter_results": {"hate": {"filtered": False, "severity": "safe"}}}
)
_OPENAI_AND_AZURE_CHOICE_EXTRAS: Final = (
    pytest.param("openai", MappingProxyType({}), id="openai"),
    pytest.param("azure", _AZURE_CHOICE_EXTRAS, id="azure"),
)


_TOKEN_IDS: Final = TypeAdapter(list[int] | None)
_CHOICE_INDEX: Final = TypeAdapter(int)
_DELTA: Final = TypeAdapter(Mapping[str, object])


def _streaming_wrapper_for_unit_test(
    model: str,
    custom_llm_provider: str,
    chunks: list[ModelResponseStream],
) -> CustomStreamWrapper:
    return CustomStreamWrapper(
        completion_stream=ModelResponseListIterator(model_responses=chunks),
        model=model,
        custom_llm_provider=custom_llm_provider,
        logging_obj=Logging(
            model=model,
            messages=[{"role": "user", "content": "unit test"}],
            stream=True,
            call_type="completion",
            start_time=0.0,
            litellm_call_id="unit-test-call",
            function_id="unit-test-function",
        ),
    )


def _unit_test_streaming_chunk(
    content: str | None,
    finish_reason: str | None = None,
    role: str | None = "assistant",
) -> ModelResponseStream:
    return ModelResponseStream(
        id="unit-test-stream",
        created=1,
        model="unit-test-model",
        choices=[
            StreamingChoices(
                index=0,
                delta=Delta(content=content, role=role),
                finish_reason=finish_reason,
            )
        ],
    )
