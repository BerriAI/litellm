import asyncio
import time
import pytest


@pytest.mark.smoke
def test_parallel_preserve_order(monkeypatch):
    from litellm import Router

    async def fake_acompletion(*, model, messages, **kw):
        txt = messages[0]["content"]
        # Introduce delay based on content marker
        if "one" in txt:
            await asyncio.sleep(0.05)
            return {"choices": [{"message": {"role": "assistant", "content": "ONE"}}]}
        else:
            await asyncio.sleep(0.01)
            return {"choices": [{"message": {"role": "assistant", "content": "TWO"}}]}

    # Monkeypatch Router.acompletion on the instance
    import os, importlib
    os.environ['LITELLM_ENABLE_PARALLEL_ACOMPLETIONS'] = '1'
    import litellm.experimental_flags as flags
    importlib.reload(flags)
    import litellm.router as router_mod
    importlib.reload(router_mod)
    r = Router()
    r.acompletion = fake_acompletion  # type: ignore

    from litellm.router_utils.parallel_acompletion import RouterParallelRequest, run_parallel_requests

    reqs = [
        RouterParallelRequest("noop", [{"role": "user", "content": "say one"}], {}),
        RouterParallelRequest("noop", [{"role": "user", "content": "say two"}], {}),
    ]

    async def run():
        results = await run_parallel_requests(r, reqs, preserve_order=True)
        outs = []
        for _, data, err in results:
            assert err is None
            choices = getattr(data, "choices", None) or data.get("choices", [])
            first = choices[0] if choices else {}
            msg = getattr(first, "message", None) or first.get("message", {})
            content = getattr(msg, "content", None) or msg.get("content")
            outs.append(content)
        return outs

    outs = asyncio.run(run())
    assert outs == ["ONE", "TWO"]  # submission order


@pytest.mark.smoke
def test_parallel_fail_fast_cancel(monkeypatch):
    from litellm import Router

    async def fake_acompletion(*, model, messages, **kw):
        txt = messages[0]["content"]
        if "err" in txt:
            await asyncio.sleep(0.01)
            raise RuntimeError("boom")
        await asyncio.sleep(0.1)
        return {"choices": [{"message": {"role": "assistant", "content": txt.upper()}}]}

    import os, importlib
    os.environ['LITELLM_ENABLE_PARALLEL_ACOMPLETIONS'] = '1'
    import litellm.experimental_flags as flags
    importlib.reload(flags)
    import litellm.router as router_mod
    importlib.reload(router_mod)
    r = Router()
    r.acompletion = fake_acompletion  # type: ignore
    from litellm.router_utils.parallel_acompletion import RouterParallelRequest, run_parallel_requests

    reqs = [
        RouterParallelRequest("noop", [{"role": "user", "content": "ok"}], {}),
        RouterParallelRequest("noop", [{"role": "user", "content": "err"}], {}),
    ]

    async def run():
        with pytest.raises(Exception):
            await run_parallel_requests(r, reqs, preserve_order=True, return_exceptions=False)

    asyncio.run(run())
