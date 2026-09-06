import asyncio
import contextvars
import copy
import gc
import json
import threading
import weakref


async def checkpoint():
    ready = asyncio.Event()
    asyncio.get_running_loop().call_soon(ready.set)
    await ready.wait()


class Value:
    pass


class ReferenceFactory:
    def __init__(self):
        self.live = 0

    def prepare(self, callable, positional, keywords=None, awaited=False):
        return ReferenceOwner(self, callable, positional, keywords, awaited)


class ReferenceOwner:
    def __init__(self, factory, callable, positional, keywords, awaited):
        self.factory = factory
        self.call = (callable, positional, keywords, awaited)
        factory.live += 1

    def invoke(self):
        if self.call is None:
            raise RuntimeError("invocation owner released")
        callable, positional, keywords, awaited = self.call
        if not awaited:
            return callable(*positional, **(keywords if keywords is not None else {}))

        async def run():
            return await callable(*positional, **(keywords if keywords is not None else {}))

        return run()

    def clone_owner(self):
        return ReferenceOwner(self.factory, *self.call)

    def close(self):
        if self.call is not None:
            released, self.call = self.call, None
            self.factory.live -= 1
            del released

    def __del__(self):
        self.close()


async def awaitable_kinds(owners):
    payload = Value()
    calls = []

    async def coroutine(value, *, alias):
        assert value is alias
        calls.append("called")
        return value

    class CustomAwaitable:
        def __await__(self):
            return coroutine(payload, alias=payload).__await__()

    for kind in ("async", "sync_coroutine", "custom", "future"):
        future = asyncio.get_running_loop().create_future()
        future.set_result(payload)
        callback = {
            "async": coroutine,
            "sync_coroutine": lambda value, *, alias: coroutine(value, alias=alias),
            "custom": lambda value, *, alias: CustomAwaitable(),
            "future": lambda value, *, alias: future,
        }[kind]
        owner = owners.prepare(callback, (payload,), {"alias": payload}, awaited=True)
        pending = owner.invoke()
        before = len(calls)
        owner.close()
        assert await pending is payload
        assert len(calls) == before + (kind != "future")

    owner = owners.prepare(lambda: payload, (), awaited=True)
    try:
        await owner.invoke()
        assert False, "non-awaitable result accepted"
    except TypeError:
        pass
    owner.close()

    inner = coroutine(payload, alias=payload)

    async def returns_coroutine():
        return inner

    owner = owners.prepare(returns_coroutine, (), awaited=True)
    assert await owner.invoke() is inner
    assert inner.cr_frame is not None
    inner.close()
    owner.close()

    direct = owners.prepare(coroutine, (payload,), {"alias": payload})
    untouched = direct.invoke()
    assert untouched.cr_frame is not None
    assert untouched.cr_await is None
    untouched.close()
    direct.close()


async def identity_and_context(owners):
    context = contextvars.ContextVar("retained_context", default="outside")
    task = asyncio.current_task()
    loop = asyncio.get_running_loop()
    thread = threading.get_ident()
    payload = {"nested": {}}
    alias = payload["nested"]
    saved = []
    gate = asyncio.Event()

    async def nested(value):
        assert asyncio.current_task() is task
        assert context.get() == "inside"
        value["nested"]["nested_call"] = True
        context.set("nested")
        return value

    async def callback(value, *, shared):
        assert value is payload and shared is alias
        assert asyncio.current_task() is task
        assert asyncio.get_running_loop() is loop
        assert threading.get_ident() == thread
        assert context.get() == "at_await"
        owner.close()
        payload["closure_mutation"] = True
        context.set("inside")
        saved.append(value)
        shared["before"] = True
        loop.call_soon(gate.set)
        await gate.wait()
        inner = owners.prepare(nested, (value,), awaited=True)
        try:
            assert await inner.invoke() is value
        finally:
            inner.close()
        return value

    owner = owners.prepare(callback, (payload,), {"shared": alias}, awaited=True)
    pending = owner.invoke()
    context.set("at_await")
    assert await pending is payload
    assert context.get() == "nested"
    assert payload["closure_mutation"] is True
    assert owners.live == 0
    alias["after"] = True
    assert saved[0]["nested"] == {"before": True, "nested_call": True, "after": True}


async def exceptions(owners):
    for error in (RuntimeError("original"), KeyboardInterrupt("original"), asyncio.CancelledError("original")):
        cause = ValueError("cause")
        payload = {}

        async def callback():
            payload["changed"] = True
            raise error from cause

        owner = owners.prepare(callback, (), awaited=True)
        try:
            await owner.invoke()
            assert False, "exception lost"
        except BaseException as caught:
            assert caught is error and caught.__cause__ is cause
            frames = []
            tb = caught.__traceback__
            while tb:
                frames.append(tb.tb_frame.f_code.co_name)
                tb = tb.tb_next
            assert "callback" in frames
        finally:
            owner.close()
        assert payload["changed"] is True


async def exception_ownership(owners):
    class Callback:
        def __init__(self, error):
            self.error = error

        async def __call__(self, value):
            value.changed = True
            raise self.error

    value = Value()
    error = RuntimeError("retained exception")
    callback = Callback(error)
    value_ref, callback_ref = weakref.ref(value), weakref.ref(callback)
    owner = owners.prepare(callback, (value,), awaited=True)
    del value, callback
    try:
        await owner.invoke()
        assert False, "expected callback failure"
    except RuntimeError as caught:
        assert caught is error
    owner.close()
    assert value_ref().changed and callback_ref() is not None
    del error
    gc.collect()
    assert value_ref() is None and callback_ref() is None


async def cancellation_before_start(owners):
    started = []

    async def callback(value):
        started.append(value)

    for operation in ("close", "cancel"):
        value = Value()
        ref = weakref.ref(value)
        owner = owners.prepare(callback, (value,), awaited=True)
        pending = owner.invoke()
        owner.close()
        del value
        assert ref() is not None
        if operation == "close":
            pending.close()
        else:
            task = asyncio.create_task(pending)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            del task
        del pending
        await checkpoint()
        gc.collect()
        assert ref() is None
    assert started == []


async def cancellation_case(owners, repeated=False, suppress=False):
    started, cleaning, finish = asyncio.Event(), asyncio.Event(), asyncio.Event()
    value = Value()
    ref = weakref.ref(value)
    observed = []

    async def callback(argument):
        try:
            started.set()
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cleaning.set()
            try:
                await finish.wait()
            except asyncio.CancelledError:
                observed.append("second cancellation")
                await finish.wait()
            argument.cleaned = True
            if suppress:
                return argument
            raise

    owner = owners.prepare(callback, (value,), awaited=True)
    task = asyncio.create_task(owner.invoke())
    owner.close()
    del value
    try:
        await started.wait()
        task.cancel()
        await cleaning.wait()
        assert ref() is not None and not task.done()
        if repeated:
            task.cancel()
            barrier = asyncio.Event()
            asyncio.get_running_loop().call_soon(barrier.set)
            await barrier.wait()
            assert observed == ["second cancellation"] and not task.done()
        finish.set()
        try:
            result = await task
            assert suppress and result is ref() and result.cleaned
            del result
        except asyncio.CancelledError:
            assert not suppress
            assert ref().cleaned
    finally:
        finish.set()
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
    del task
    await checkpoint()
    gc.collect()
    assert ref() is None


async def cancellation_unwinds(owners):
    await cancellation_case(owners)


async def cancellation_during_cleanup(owners):
    await cancellation_case(owners, repeated=True)


async def cancellation_suppressed(owners):
    await cancellation_case(owners, suppress=True)


async def registration_and_gc(owners):
    original = Value()
    original_ref = weakref.ref(original)
    registered = owners.prepare(lambda value: value, (original,))
    active = registered.clone_owner()
    registered.close()
    registered = owners.prepare(lambda: "replacement", ())
    del original
    assert active.invoke() is original_ref()
    active.close()
    gc.collect()
    assert original_ref() is None
    assert registered.invoke() == "replacement"
    registered.close()

    for edge in ("callable", "positional", "keywords"):

        class Callback:
            def __call__(self, *args, **kwargs):
                pass

        value = Callback()
        owner = owners.prepare(
            value if edge == "callable" else lambda *a, **k: None,
            (value,) if edge == "positional" else (),
            {"value": value} if edge == "keywords" else None,
        )
        value.owner = owner
        value_ref, owner_ref = weakref.ref(value), weakref.ref(owner)
        del value, owner
        gc.collect()
        assert value_ref() is None and owner_ref() is None
        assert owners.live == 0

    finalized = []

    class Reenter:
        def __del__(self):
            try:
                reentrant.close()
                another = owners.prepare(lambda: 42, ())
                finalized.append(another.invoke())
                another.close()
            except BaseException as error:
                finalized.append(type(error).__name__)

    value = Reenter()
    reentrant = owners.prepare(lambda value: None, (value,))
    del value
    reentrant.close()
    reentrant.close()
    assert finalized == [42]
    assert owners.live == 0


async def background_and_session(owners):
    context = contextvars.ContextVar("background_context", default="initial")
    start, finish = asyncio.Event(), asyncio.Event()
    payload = {"nested": {"value": "queued"}}
    saved = []

    async def upload(value):
        assert context.get() == "submission"
        start.set()
        await finish.wait()
        saved.append(json.dumps(value))

    session = owners.prepare(lambda value: value, (payload,))
    first_response, second_response = session.clone_owner(), session.clone_owner()
    upload_owner = owners.prepare(upload, (payload,), awaited=True)
    context.set("submission")
    task = asyncio.create_task(upload_owner.invoke())
    context.set("consumer")
    upload_owner.close()
    first_response.close()
    await start.wait()
    assert second_response.invoke() is payload
    payload["nested"]["value"] = "later"
    consumed = json.dumps(payload)
    payload["nested"]["after_consumption"] = True
    assert "after_consumption" not in consumed
    second_response.close()
    session.close()
    assert owners.live == 0 and not task.done()
    finish.set()
    await task
    assert json.loads(saved[0]) == payload
    assert context.get() == "consumer"


async def stream_lifecycle(owners):
    for terminal in ("exhaustion", "failure", "close"):
        nested = {"usage": 0}
        item = {"nested": nested}
        closed = []

        async def source():
            try:
                yield item
                nested["usage"] = 12
                if terminal == "failure":
                    raise ValueError("stream failure")
            finally:
                closed.append(True)

        stream = source()
        pull = owners.prepare(stream.__anext__, (), awaited=True)
        yielded = await pull.invoke()
        assert yielded is item
        shallow, deep = copy.copy(yielded), copy.deepcopy(yielded)
        retained = owners.prepare(lambda value: value, (yielded["nested"],))
        yielded["nested"] = {"replacement": True}
        if terminal == "close":
            close = owners.prepare(stream.aclose, (), awaited=True)
            await close.invoke()
            await close.invoke()
            close.close()
        else:
            try:
                await pull.invoke()
                assert False, "expected stream termination"
            except (StopAsyncIteration, ValueError) as error:
                assert isinstance(error, ValueError) == (terminal == "failure")
        pull.close()
        assert closed == [True]
        assert retained.invoke() is nested
        assert shallow["nested"] is nested and deep["nested"]["usage"] == 0
        assert nested["usage"] == (0 if terminal == "close" else 12)
        retained.close()


async def sync_stream_lifecycle(owners):
    for terminal in ("exhaustion", "failure", "close"):
        value = {"nested": {"usage": 0}}
        closed = []

        def source():
            try:
                yield value
                value["nested"]["usage"] = 12
                if terminal == "failure":
                    raise ValueError("stream failure")
            finally:
                closed.append(True)

        stream = source()
        pull = owners.prepare(stream.__next__, ())
        assert pull.invoke() is value
        saved = owners.prepare(lambda value: value, (value,))
        if terminal == "close":
            close = owners.prepare(stream.close, ())
            close.invoke()
            close.invoke()
            close.close()
        else:
            try:
                pull.invoke()
                assert False, "stream did not terminate"
            except (StopIteration, ValueError) as error:
                assert isinstance(error, ValueError) == (terminal == "failure")
        pull.close()
        assert saved.invoke() is value
        assert value["nested"]["usage"] == (0 if terminal == "close" else 12)
        assert closed == [True]
        saved.close()


async def repeated_ownership(owners):
    refs = []
    for batch in range(8):
        gate = asyncio.Event()

        async def work(value):
            await gate.wait()
            return None

        tasks = []
        for index in range(4):
            value = Value()
            refs.append(weakref.ref(value))
            owner = owners.prepare(work, (value,), awaited=True)
            tasks.append(asyncio.create_task(owner.invoke()))
            owner.close()
            del value
        gate.set()
        await asyncio.gather(*tasks)
        del tasks
        gc.collect()
        assert owners.live == 0
        assert all(ref() is None for ref in refs)


def run_scenario(name, retained, factory):
    owners = factory if retained else ReferenceFactory()

    async def run():
        async with asyncio.timeout(15):
            await globals()[name](owners)
        assert owners.live == 0
        pending = asyncio.all_tasks() - {asyncio.current_task()}
        assert not pending, f"undrained tasks: {pending}"

    asyncio.run(run())
    gc.collect()
    assert owners.live == 0
