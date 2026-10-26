from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Final

_REPOSITORY: Final = Path(__file__).resolve().parents[3]
_CONCURRENT_FIRST_USE: Final = """
import sys
import threading

from anthropic.types import Message, RawContentBlockDeltaEvent
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types.responses import Response, ResponseTextDeltaEvent

from tests.integration._support.sdk_models import build_sdk_models

USAGE = {"input_tokens": 1, "output_tokens": 1}
PAYLOADS = (
    (
        Response,
        {
            "id": "resp_1",
            "object": "response",
            "created_at": 1,
            "model": "gpt-4.1-mini",
            "status": "completed",
            "output": [
                {
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": "hi", "annotations": []}],
                }
            ],
            "usage": {**USAGE, "total_tokens": 2},
        },
    ),
    (
        ChatCompletion,
        {
            "id": "chatcmpl_1",
            "object": "chat.completion",
            "created": 1,
            "model": "gpt-4o-mini",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "hi"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    ),
    (
        Message,
        {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-5",
            "content": [{"type": "text", "text": "hi"}],
            "stop_reason": "end_turn",
            "usage": USAGE,
        },
    ),
    (
        ResponseTextDeltaEvent,
        {
            "type": "response.output_text.delta",
            "item_id": "msg_1",
            "output_index": 0,
            "content_index": 0,
            "sequence_number": 1,
            "delta": "hi",
            "logprobs": [],
        },
    ),
    (
        ChatCompletionChunk,
        {
            "id": "chatcmpl_1",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "gpt-4o-mini",
            "choices": [{"index": 0, "finish_reason": None, "delta": {"role": "assistant", "content": "hi"}}],
        },
    ),
    (
        RawContentBlockDeltaEvent,
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "hi"}},
    ),
)
THREADS = 30

sys.setswitchinterval(1e-6)
build_sdk_models()
gate = threading.Barrier(THREADS)
failures = []


def first_use():
    gate.wait()
    for model, payload in PAYLOADS:
        try:
            model.model_construct(**payload)
        except Exception as error:
            failures.append(f"{model.__name__}: {type(error).__name__}: {error}")


threads = [threading.Thread(target=first_use) for _ in range(THREADS)]
for thread in threads:
    thread.start()
for thread in threads:
    thread.join()
print(sorted(set(failures)))
"""


def test_sdk_models_built_up_front_survive_concurrent_first_use_from_many_threads() -> None:
    completed: Final = subprocess.run(
        (sys.executable, "-P", "-c", _CONCURRENT_FIRST_USE),
        env={**os.environ, "PYTHONPATH": str(_REPOSITORY)},
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "[]", completed.stdout
