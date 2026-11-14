import asyncio
import contextvars
import sys
import time
import traceback
import types
from pathlib import Path

repo = Path(globals()["repo"])
rust = globals()["rust"]
dropped = globals()["dropped"]

for name, path in (("litellm", repo / "litellm"), ("litellm.integrations", repo / "litellm" / "integrations")):
    if name not in sys.modules:
        stub = types.ModuleType(name)
        stub.__path__ = [str(path)]
        sys.modules[name] = stub

from litellm.integrations.custom_logger_v2 import (
    CallbackRegistry,
    Dispatcher,
    Field,
    Hook,
    InProcessSink,
    RequestSelection,
    build_manifest,
    plan_hook,
)
from litellm.integrations.custom_logger_v2.host import bind_scope, current_scope
from litellm.integrations.custom_logger_v2.rust import rust_logger
from litellm.integrations.custom_logger_v2.planner import Entry
from litellm.rust_bridge.execution import Complete, run

VAR = contextvars.ContextVar("host_calls_var", default="unset")


class Boom(Exception):
    pass


class Host:
    def __init__(self):
        self.asked = []
        self.tasks = []
        self.raised = []
        self.cancelled = []
        self.entered = asyncio.Event()
        self.expected_scope = None
        self.delivered = []

    def lookup(self, request):
        scenario = request["scenario"]
        step = request["step"]
        self.asked.append((scenario, step))
        return getattr(self, scenario)(step)

    def plain(self, step):
        return "plain"

    async def coroutine(self, step):
        await asyncio.sleep(0)
        return "coroutine"

    async def reads(self, step):
        await asyncio.sleep(0)
        return VAR.get()

    async def writes(self, step):
        await asyncio.sleep(0)
        VAR.set("written")
        return "wrote"

    def writes_sync(self, step):
        VAR.set("written sync")
        return "wrote sync"

    async def sequential(self, step):
        await asyncio.sleep(0)
        return f"step{step}"

    async def task(self, step):
        self.tasks.append(asyncio.current_task())
        return "task"

    async def scope(self, step):
        return "scoped" if current_scope() is self.expected_scope and current_scope() is not None else "unscoped"

    async def boom(self, step):
        await asyncio.sleep(0)
        error = Boom("no spend for sk-1")
        self.raised.append(error)
        raise error

    def boom_sync(self, step):
        error = Boom("sync failure")
        self.raised.append(error)
        raise error

    def handled(self, step):
        raise Boom("handled failure")

    async def cancel_host(self, step):
        self.entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled.append(step)
            raise

    def cancel_rust(self, step):
        if step == 1:
            self.entered.set()
        return "entered"

    async def attempt(self, step):
        await asyncio.sleep(0)
        self.delivered.append(VAR.get())
        return "delivered"


def plan(logger, hook=Hook.pre_call):
    manifest = build_manifest(logger)
    return plan_hook((Entry(0, manifest, manifest.subscriptions[hook]),))


def ignore(logger, kwargs):
    return None


def fresh_scope():
    return Dispatcher(CallbackRegistry(), sink=InProcessSink()).open_scope(RequestSelection())


async def pre_call(model, host, scope=None):
    scope = scope or fresh_scope()
    logger = rust_logger(rust, host=host)
    result = await scope.run_plan(Hook.pre_call, plan(logger), {Field.model: model}, None, ignore)
    return result.fields[Field.model]


def model_of(returned):
    return returned["patch"]["ops"][0]["value"]


async def until(predicate):
    for _ in range(400):
        if predicate():
            return
        await asyncio.sleep(0.005)
    raise AssertionError("timed out")


def plain_value():
    host = Host()
    assert asyncio.run(pre_call("plain", host)) == "plain"
    assert host.asked == [("plain", 1)]


def coroutine():
    assert asyncio.run(pre_call("coroutine", Host())) == "coroutine"


def reads_context_var():
    async def main():
        VAR.set("set by the caller")
        return await pre_call("reads", Host())

    assert asyncio.run(main()) == "set by the caller"


def writes_context_var():
    async def main():
        assert await pre_call("writes", Host()) == "wrote"
        after_async = VAR.get()
        assert await pre_call("writes_sync", Host()) == "wrote sync"
        return after_async, VAR.get()

    assert asyncio.run(main()) == ("written", "written sync")


def sequential_calls():
    host = Host()
    assert asyncio.run(pre_call("sequential", host)) == "step1+step2"
    assert host.asked == [("sequential", 1), ("sequential", 2)]


def same_task():
    async def main():
        host = Host()
        await pre_call("task", host)
        return host.tasks, asyncio.current_task()

    tasks, caller = asyncio.run(main())
    assert len(tasks) == 1
    assert tasks[0] is caller


def request_scope():
    async def main():
        host = Host()
        scope = fresh_scope()
        host.expected_scope = scope
        with bind_scope(scope):
            return await pre_call("scope", host, scope)

    assert asyncio.run(main()) == "scoped"


def original_exception():
    for scenario, frame in (("boom", "boom"), ("boom_sync", "boom_sync")):
        host = Host()
        try:
            asyncio.run(pre_call(scenario, host))
        except Boom as error:
            assert error is host.raised[0], scenario
            assert frame in [entry.name for entry in traceback.extract_tb(error.__traceback__)], scenario
        else:
            raise AssertionError(f"{scenario} did not raise")


def handled_exception():
    assert asyncio.run(pre_call("handled", Host())) == "handled Boom: handled failure"


def panic_contained():
    try:
        asyncio.run(pre_call("panic", Host()))
    except RuntimeError as error:
        assert str(error) == "the rust callback panicked: host calls panicked", str(error)
    else:
        raise AssertionError("the panic did not raise")


def cancelled_in_host_call():
    async def main():
        host = Host()
        task = asyncio.ensure_future(pre_call("cancel_host", host))
        await host.entered.wait()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        else:
            raise AssertionError("the hook was not cancelled")
        assert host.cancelled == [1]
        await until(lambda: dropped("cancel_host"))

    asyncio.run(main())


def cancelled_in_rust():
    async def main():
        host = Host()
        task = asyncio.ensure_future(pre_call("cancel_rust", host))
        await host.entered.wait()
        await asyncio.sleep(0.02)
        assert not dropped("cancel_rust")
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        else:
            raise AssertionError("the hook was not cancelled")
        await until(lambda: dropped("cancel_rust"))
        await asyncio.sleep(0.02)
        assert host.asked == [("cancel_rust", 1)]

    asyncio.run(main())


def no_host_call():
    async def main():
        hosted = await pre_call("none", Host())
        unhosted = await pre_call("none", None)
        step = rust.start("pre_call", {"model": "none"}, None, None, Host()).send(None)
        return hosted, unhosted, step

    hosted, unhosted, step = asyncio.run(main())
    assert hosted == unhosted == "rust/none"
    assert isinstance(step, Complete)
    assert model_of(step.value) == "rust/none"


def detached():
    async def main():
        return await rust.handle("pre_call", {"model": "plain"}, None, None)

    try:
        asyncio.run(main())
    except RuntimeError as error:
        assert str(error) == "no host serves the lookup host call", str(error)
    else:
        raise AssertionError("a detached host call did not raise")


def sync_line():
    host = Host()
    VAR.set("set before the sync line")
    logger = rust_logger(rust, host=host)
    for scenario, expected in (("coroutine", "coroutine"), ("reads", "set before the sync line"), ("writes", "wrote")):
        fields = fresh_scope().run_plan_sync(Hook.pre_call, plan(logger), {Field.model: scenario}, None, ignore)
        assert fields[Field.model] == expected, scenario
    assert VAR.get() == "set before the sync line"


def off_path_without_a_loop():
    host = Host()
    VAR.set("set before the off-path line")
    logger = rust_logger(rust, host=host)
    scope = fresh_scope()
    scope.run_plan_sync(Hook.attempt_ended, plan(logger, Hook.attempt_ended), {Field.model: "attempt"}, None, ignore)
    for _ in range(400):
        if host.delivered:
            break
        time.sleep(0.005)
    assert host.delivered == ["unset"]


def sync_driver():
    def streamless(execution, head):
        raise AssertionError("no stream")

    assert model_of(run(rust.start("pre_call", {"model": "plain"}, None, None, Host()), streamless)) == "plain"
    host = AwaitingHost()
    try:
        run(rust.start("pre_call", {"model": "plain"}, None, None, host), streamless)
    except RuntimeError as error:
        assert str(error) == "sync call suspended"
    else:
        raise AssertionError("a sync driver awaited a host call")
    finally:
        for awaitable in host.returned:
            awaitable.close()


class AwaitingHost:
    def __init__(self):
        self.returned = []

    def lookup(self, request):
        awaitable = asyncio.sleep(0, result="never awaited")
        self.returned.append(awaitable)
        return awaitable
