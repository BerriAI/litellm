import asyncio
import os
import types
import pytest

import litellm
from litellm.router import Router
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper


async def _fake_stream_chunks(texts):
    for i, t in enumerate(texts):
        yield {
            "text": t,
            "is_finished": i == len(texts) - 1,
            "finish_reason": "stop" if i == len(texts) - 1 else None,
            "usage": None,
        }


class _DummyLog:
    def __init__(self):
        # minimal fields CustomStreamWrapper expects
        self.model_call_details = {"litellm_params": {}}
        self.messages = []
        self.stream_options = None
        self.optional_params = {}
        self.completion_start_time = None

    # no-op handlers used by streaming wrapper
    async def async_success_handler(self, *args, **kwargs):
        return None

    def success_handler(self, *args, **kwargs):
        return None

    async def async_failure_handler(self, *args, **kwargs):
        return None

    def failure_handler(self, *args, **kwargs):
        return None

    def _update_completion_start_time(self, completion_start_time=None, **kwargs):
        self.completion_start_time = completion_start_time


@pytest.mark.parametrize("mode", ["legacy", "extracted"])
@pytest.mark.asyncio
async def test_router_streaming_smoke(monkeypatch, mode):
    monkeypatch.setenv("LITELLM_ROUTER_CORE", mode)

    async def fake_acompletion(**kwargs):
        return CustomStreamWrapper(
            completion_stream=_fake_stream_chunks(["a", "b"]),
            model="dummy",
            logging_obj=_DummyLog(),
        )

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

    router = Router(
        model_list=[
            {
                "model_name": "alias-a",
                "litellm_params": {"model": "openai/gpt-3.5-turbo", "api_key": "sk-test"},
            }
        ]
    )

    stream = await router.acompletion(
        model="alias-a",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
    )
    assert isinstance(stream, CustomStreamWrapper)

    collected = []
    async for chunk in stream:
        if chunk and getattr(chunk.choices[0].delta, "content", None):
            collected.append(chunk.choices[0].delta.content)
    assert collected == ["a", "b"]
