import asyncio
import pytest
from litellm.router import Router
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper


async def _fake_stream(texts):
    for i, t in enumerate(texts):
        yield {"text": t, "is_finished": i == len(texts) - 1, "finish_reason": "stop" if i == len(texts) - 1 else None, "usage": None}


class _DummyLog:
    def __init__(self):
        self.model_call_details = {}
        self.messages = []
        self.stream_options = None
        self.optional_params = {}
        self.completion_start_time = None
    async def async_success_handler(self,*a,**k):
        return None
    def success_handler(self,*a,**k):
        return None
    async def async_failure_handler(self,*a,**k):
        return None
    def failure_handler(self,*a,**k):
        return None
    def _update_completion_start_time(self, completion_start_time=None, **k):
        self.completion_start_time = completion_start_time


@pytest.mark.smoke
def test_router_streaming_parity(monkeypatch):
    import litellm

    async def fake_acompletion(**kwargs):
        return CustomStreamWrapper(
            completion_stream=_fake_stream(["x", "y"]),
            model="dummy",
            logging_obj=_DummyLog(),
        )

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

    def run(mode):
        import os
        os.environ["LITELLM_ROUTER_CORE"] = mode
        r = Router(model_list=[{"model_name": "a", "litellm_params": {"model": "openai/gpt-4o-mini", "api_key": "sk-test"}}])
        stream = asyncio.get_event_loop().run_until_complete(
            r.acompletion(model="a", messages=[{"role": "user", "content": "hi"}], stream=True)
        )
        out = []

        async def collect():
            async for c in stream:
                if isinstance(c, dict):
                    out.append(c.get("text", ""))
                else:
                    try:
                        delta = getattr(c.choices[0], "delta", None)
                        content = getattr(delta, "content", None)
                        if content:
                                out.append(content)
                    except Exception:
                        pass

        asyncio.get_event_loop().run_until_complete(collect())
        return out

    legacy = run("legacy")
    extracted = run("extracted")
    assert legacy == extracted == ["x", "y"]
