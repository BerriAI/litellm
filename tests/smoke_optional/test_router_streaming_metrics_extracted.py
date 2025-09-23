import asyncio
import pytest
from litellm.router import Router
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper


async def _fake_stream(texts):
    import asyncio as aio
    for i, t in enumerate(texts):
        if i == 0:
            await aio.sleep(0.01)
        yield {"text": t, "is_finished": i == len(texts) - 1, "finish_reason": "stop" if i == len(texts) - 1 else None, "usage": None}


class _DummyLog:
    def __init__(self):
        self.model_call_details = {}
        self.messages = []
        self.stream_options = None
        self.optional_params = {}
        self.completion_start_time = None
    async def async_success_handler(self,*a,**k): return None
    def success_handler(self,*a,**k): return None
    async def async_failure_handler(self,*a,**k): return None
    def failure_handler(self,*a,**k): return None
    def _update_completion_start_time(self, completion_start_time=None, **k):
        self.completion_start_time = completion_start_time


@pytest.mark.smoke
def test_router_streaming_metrics_extracted(monkeypatch):
    import os, litellm

    os.environ["LITELLM_ROUTER_CORE"] = "extracted"

    async def fake_acompletion(**kwargs):
        return CustomStreamWrapper(
            completion_stream=_fake_stream(["a", "b"]),
            model="dummy",
            logging_obj=_DummyLog(),
        )

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

    r = Router(model_list=[{"model_name": "a", "litellm_params": {"model": "openai/gpt-4o-mini", "api_key": "sk-test"}}])

    stream = asyncio.get_event_loop().run_until_complete(
        r.acompletion(model="a", messages=[{"role": "user", "content": "hi"}], stream=True)
    )

    async def drain():
        async for _ in stream:
            pass

    asyncio.get_event_loop().run_until_complete(drain())

    metrics = getattr(stream, "_hidden_params", {}).get("metrics", {}) or getattr(stream, "_litellm_metrics", {})
    assert "ttft_ms" in metrics and "total_ms" in metrics and metrics["total_ms"] >= metrics["ttft_ms"]
