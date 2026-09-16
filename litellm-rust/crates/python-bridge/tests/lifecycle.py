import asyncio
import gc
import threading
import weakref
from contextvars import ContextVar


async def exercise():
    caller = asyncio.current_task()
    thread = threading.get_ident()
    loop = asyncio.get_running_loop()
    marker = ContextVar("driver", default="before")
    entered = asyncio.Event()
    released = asyncio.Event()
    result = object()

    class CustomAwaitable:
        def __await__(self):
            return operation().__await__()

    async def operation():
        assert asyncio.current_task() is caller
        assert threading.get_ident() == thread
        assert asyncio.get_running_loop() is loop
        marker.set("inside")
        entered.set()
        await released.wait()
        assert asyncio.current_task() is caller
        assert marker.get() == "inside"
        return result

    async def release():
        await entered.wait()
        released.set()

    releaser = asyncio.create_task(release())
    execution = await_execution(CustomAwaitable())
    try:
        execution.resume_value(None)
    except RuntimeError:
        pass
    else:
        raise AssertionError("resumed an unstarted execution")
    wrapped = drive(execution)
    try:
        wrapped.send(1)
    except TypeError:
        pass
    else:
        raise AssertionError("accepted initial value")
    assert await wrapped is result
    assert marker.get() == "inside"
    await releaser
    execution.close()
    execution.close()
    try:
        await wrapped
    except RuntimeError:
        pass
    else:
        raise AssertionError("accepted coroutine reuse")

    final_awaitable = CustomAwaitable()
    assert await drive(calling_execution(lambda: final_awaitable)) is final_awaitable

    cause = KeyError("cause")
    failure = ValueError("original")

    async def failing():
        await asyncio.sleep(0)
        raise failure from cause

    try:
        await drive(await_execution(failing()))
    except ValueError as error:
        assert error is failure
        assert error.__cause__ is cause
        names = []
        traceback = error.__traceback__
        while traceback:
            names.append(traceback.tb_frame.f_code.co_name)
            traceback = traceback.tb_next
        assert "failing" in names
    else:
        raise AssertionError("lost original exception")

    for suppress in (False, True):
        pending = asyncio.Event()
        cleanup_entered = asyncio.Event()
        cleanup_release = asyncio.Event()
        cleaned = []

        async def cancel_operation():
            try:
                pending.set()
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                if suppress:
                    return result
                raise
            finally:
                cleanup_entered.set()
                try:
                    await cleanup_release.wait()
                except asyncio.CancelledError:
                    await cleanup_release.wait()
                cleaned.append(asyncio.current_task())

        task = asyncio.create_task(drive(await_execution(cancel_operation())))
        await pending.wait()
        task.cancel()
        await cleanup_entered.wait()
        assert not task.done()
        task.cancel()
        await asyncio.sleep(0)
        cleanup_release.set()
        if suppress:
            assert await task is result
        else:
            try:
                await task
            except asyncio.CancelledError:
                pass
            else:
                raise AssertionError("lost cancellation")
        assert cleaned == [task]

    observed = []

    def reenter():
        try:
            active.start()
        except RuntimeError as error:
            observed.append(str(error))
        return result

    active = calling_execution(reenter)
    assert await drive(active) is result
    assert observed == ["execution is already running"]

    class Finalizer:
        def __call__(self):
            return result

        def __del__(self):
            self.owner.close()
            observed.append("released")

    def cycle(started):
        callback = Finalizer()
        execution = calling_execution(callback)
        callback.owner = execution
        if started:
            assert execution.start().value is result
        return weakref.ref(callback)

    for started in (False, True):
        reference = cycle(started)
        gc.collect()
        assert reference() is None
    assert observed[-2:] == ["released", "released"]

    class Awaitable:
        def __await__(self):
            try:
                yield self
            finally:
                observed.append("unwound")

    def abandoned(started):
        awaitable = Awaitable()
        coroutine = drive(await_execution(awaitable))
        awaitable.owner = coroutine
        if started:
            assert coroutine.send(None) is awaitable
        coroutine.close()
        return weakref.ref(awaitable)

    for started in (False, True):
        reference = abandoned(started)
        gc.collect()
        assert reference() is None
    assert observed[-1] == "unwound"


asyncio.run(asyncio.wait_for(exercise(), 10))
