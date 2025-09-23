import asyncio
import pytest

from litellm.router import Router
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper


class _DummyLog:
    def __init__(self):
        self.model_call_details = {"litellm_params": {}}
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


async def _gen_with_midstream_fallback():
    # Yield one token, then raise MidStreamFallbackError
    yield {"text": "X", "is_finished": False, "finish_reason": None, "usage": None}
    from litellm.exceptions import MidStreamFallbackError
    raise MidStreamFallbackError(generated_content="X")


async def _fallback_stream():
    yield {"text": "Y", "is_finished": False, "finish_reason": None, "usage": None}
    yield {"text": "Z", "is_finished": True, "finish_reason": "stop", "usage": None}


@pytest.mark.smoke
def test_router_streaming_midstream_fallback_parity(monkeypatch):
    import os, litellm

    # Fake primary acompletion returns a stream that mid-stream-falls back
    async def fake_acompletion(**kwargs):
        return CustomStreamWrapper(
            completion_stream=_gen_with_midstream_fallback(),
            model="dummy",
            logging_obj=_DummyLog(),
        )

    # When router performs fallback, return a streaming response of Y then Z
    async def fake_async_function_with_fallbacks_common_utils(*args, **kwargs):
        return CustomStreamWrapper(
            completion_stream=_fallback_stream(),
            model="dummy",
            logging_obj=_DummyLog(),
        )

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

    # Patch method on Router instance; we'll bind via monkeypatch.setattr on class
    monkeypatch.setattr(Router, "async_function_with_fallbacks_common_utils", staticmethod(fake_async_function_with_fallbacks_common_utils))

    def run(mode):
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
    def _norm(xs):
        return xs[1:] if xs and xs[0] == "X" else xs
    assert _norm(legacy) == _norm(extracted) == ["Y", "Z"]
