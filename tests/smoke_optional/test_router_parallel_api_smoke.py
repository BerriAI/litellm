import asyncio
import pytest


class _Msg:
    def __init__(self, content: str):
        self.role = "assistant"
        self.content = content


class _Choice:
    def __init__(self, content: str):
        self.message = _Msg(content)


class _Resp:
    def __init__(self, content: str):
        self.choices = [_Choice(content)]


@pytest.mark.smoke
def test_parallel_as_completed_and_acompletions(monkeypatch):
    from litellm.router import Router
    from litellm.router_utils.parallel_acompletion import RouterParallelRequest

    r = Router()

    async def fake_acompletion(self, *, model: str, messages, **kwargs):  # type: ignore[no-redef]
        # Delay varies by prompt length to create different completion order
        user_text = messages[0].get("content") if messages else ""
        await asyncio.sleep(0.01 * (len(user_text) % 3))
        return _Resp(f"ok:{user_text}")

    monkeypatch.setattr(Router, "acompletion", fake_acompletion, raising=True)

    prompts = ["a", "bbbb", "cc"]
    reqs = [
        RouterParallelRequest(model="m", messages=[{"role": "user", "content": p}], kwargs={})
        for p in prompts
    ]

    # As-completed iterator should potentially yield out of submission order
    seen = []
    async def run_as_completed():
        async for res in r.parallel_as_completed(reqs):
            seen.append((res.index, res.content))
    asyncio.run(run_as_completed())
    assert sorted(i for i, _ in seen) == [0, 1, 2]
    # Contents match prompts
    assert {c for _, c in seen} == {"ok:a", "ok:bbbb", "ok:cc"}

    # All-at-once path with preserve_order
    results = asyncio.run(r.parallel_acompletions(reqs, preserve_order=True, return_exceptions=True))
    assert [res.index for res in results] == [0, 1, 2]
    assert [res.content for res in results] == ["ok:a", "ok:bbbb", "ok:cc"]
