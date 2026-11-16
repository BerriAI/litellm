"""Verification-only repro of three review findings on PR2 0f89cd086b5. Not part of the repository.
Real Redis (REDIS_HOST/REDIS_PORT) runs the shipped STREAK_SCRIPT. Callbacks are awaited one by one in the stated
arrival order, so ordering is explicit and no sleep is used. No provider call is made (respx answers case 3).
usage: PYTHONPATH=<worktree> REDIS_HOST=127.0.0.1 REDIS_PORT=6392 python greptile_repro.py"""
import asyncio, json, os, uuid
import respx, litellm
from redis.exceptions import ConnectionError as RedisConnectionError
from litellm.caching.redis_cache import RedisCache
from litellm.integrations.prompt_cache_alerts.detector import STREAK_SCRIPT, PromptCacheAlertLogger
from litellm.litellm_core_utils.litellm_logging import Logging
from tests.unit.integrations.prompt_cache_alerts import test_detector as t

S = t._START
OUT = []

def kw(dep, call, at, read, write):
    k = t._kwargs(call_id=call, at=S + at, raw=t._raw(read, write))
    return {**k, "standard_logging_object": {**k["standard_logging_object"], "model_id": dep}}

EV = {"P": (0, 0, 2048), "A": (10, 0, 0), "H": (20, 2048, 0), "B": (30, 0, 0), "C": (40, 0, 0)}

class FailOn:
    def __init__(self, run): self.run, self.fail_next, self.calls = run, False, []
    async def __call__(self, keys, args):
        if self.fail_next:
            self.fail_next = False
            raise RedisConnectionError("injected")
        self.calls.append(args[0]); return await self.run(keys, args)

async def deliver(loggers, dep, order, fail=None):
    """order: sequence of (event name, worker index). Returns alerts recorded after each callback."""
    trace = []
    for name, w in order:
        logger, script, alerts = loggers[w]
        if fail == (name, w): script.fail_next = True
        await t._log(logger, kw(dep, name, *EV[name]))
        trace.append(f"{name}@w{w}:{sum(len(a.recorded) for _, _, a in loggers)}")
    return trace

def worker(run):
    script, alerts = FailOn(run), t._Alerts()
    return (PromptCacheAlertLogger(t._KEY, lambda: script, alerts, clock=lambda: t._CLOCK_MS / 1000), script, alerts)

def report(case, expected_by_review, trace, loggers):
    total = sum(len(a.recorded) for _, _, a in loggers)
    OUT.append({"case": case, "arrival": trace, "incidents_saved": total,
                "evidence": [d.evidence.model_dump(mode="json") for _, _, a in loggers for d in a.recorded][:1],
                "lost_left": [dict(l._lost) for l, _, _ in loggers]})
    print(f"{case}: arrival={' '.join(trace)} incidents={total}")

async def main():
    cache = RedisCache(host=os.environ["REDIS_HOST"], port=int(os.environ["REDIS_PORT"]))
    run = cache.async_register_script(STREAK_SCRIPT)
    def dep(): return f"greptile-{uuid.uuid4()}"

    w = [worker(run)]
    report("1 control: chronological P A H B C", 0, await deliver(w, dep(), [(n, 0) for n in "PAHBC"]), w)
    w = [worker(run)]
    report("1 finding: requests P A H B C, callbacks arrive P A B C H", 1, await deliver(w, dep(), [(n, 0) for n in "PABCH"]), w)

    w = [worker(run), worker(run)]
    report("2 control: hit H on w0 reaches Redis, then C on w1", 0,
           await deliver(w, dep(), [("P", 0), ("A", 0), ("B", 1), ("H", 0), ("C", 1)]), w)
    w = [worker(run), worker(run)]
    report("2 control: H on w0 fails Redis, third miss C also on w0", 0,
           await deliver(w, dep(), [("P", 0), ("A", 0), ("B", 1), ("H", 0), ("C", 0)], fail=("H", 0)), w)
    w = [worker(run), worker(run)]
    report("2 finding: H on w0 fails Redis, third miss C on w1", 1,
           await deliver(w, dep(), [("P", 0), ("A", 0), ("B", 1), ("H", 0), ("C", 1)], fail=("H", 0)), w)

    # case 3: a full acompletion through the real callback pipeline, with and without no-log
    litellm.disable_aiohttp_transport = True
    counts = {}
    for label, extra in (("plain", {}), ("no-log", {"no-log": True})):
        script = t._Script(); litellm.callbacks = [PromptCacheAlertLogger(t._KEY, lambda s=script: s, t._Alerts())]
        with respx.mock(assert_all_called=False) as m:
            m.post(t._API_BASE).mock(return_value=t._anthropic_reply(t._usage(2048, 0)))
            await litellm.acompletion(model="anthropic/claude-sonnet-5-5", api_key="sk-ant-test",
                messages=[{"role": "system", "content": [{"type": "text", "text": "shared policy", "cache_control": {"type": "ephemeral"}}]}, {"role": "user", "content": "q"}],
                metadata={"model_info": {"id": "claude-east"}}, **extra)
        await t.drain_logging_tasks(); await asyncio.wait_for(t.GLOBAL_LOGGING_WORKER.flush(), timeout=5)
        counts[label] = script.events()
    print("3 healthy hit completion, detector script calls after logging drained ->", counts)
    OUT.append({"case": "3 no-log", "script_events_after_drain": counts})
    await cache.disconnect()
    json.dump(OUT, open(os.environ["OUT"], "w"), indent=1, default=str)

asyncio.run(main())
