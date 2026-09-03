import asyncio
import contextvars
import gc
import inspect
import threading
import weakref
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from typing import Final, Protocol

Payload = dict[str, str]
BeforeSend = dict[str, Payload]


class Harness(Protocol):
    def execute(self, adapter: object, fail: bool = False) -> Awaitable[Payload]: ...
    def sync(self, adapter: object) -> Payload: ...
    def interrupt(self, adapter: object, stop: Awaitable[object]) -> Awaitable[Payload]: ...
    def calls(self) -> int: ...


REQUEST_CONTEXT: Final = contextvars.ContextVar("request_context", default="default")


@dataclass(frozen=True, slots=True)
class Adapter:
    pre_request: Callable[[Payload], object]
    pre_api_call: Callable[[BeforeSend], object]
    post_response: Callable[[Payload], object]
    failure: Callable[[Payload], object]


class Recorder:
    def __init__(self) -> None:
        self.context = REQUEST_CONTEXT.get()
        self.loop = asyncio.get_running_loop()
        self.thread = threading.get_ident()
        self.events: tuple[str, ...] = ()

    def record(self, event: str) -> None:
        assert REQUEST_CONTEXT.get() == self.context
        assert asyncio.get_running_loop() is self.loop
        assert threading.get_ident() == self.thread
        self.events += (event,)

    async def pre_request(self, request: Payload) -> Payload:
        self.record("pre_request")
        await asyncio.sleep(0)
        return {"text": request["text"].upper()}

    def pre_api_call(self, event: BeforeSend) -> None:
        self.record("pre_api_call")
        event["body"]["text"] = "observer mutation"

    async def post_response(self, response: Payload) -> Payload:
        self.record("post_response")
        await asyncio.sleep(0)
        return {"text": response["text"] + "!"}

    def failure(self, error: Payload) -> None:
        self.record("failure")
        assert error == {"message": "provider failed"}

    def adapter(self) -> Adapter:
        return Adapter(self.pre_request, self.pre_api_call, self.post_response, self.failure)


async def transforms_and_context(harness: Harness) -> None:
    async def run_one(index: int) -> None:
        token: Final = REQUEST_CONTEXT.set(f"request-{index}")
        try:
            recorder: Final = Recorder()
            result: Final = await harness.execute(recorder.adapter())
            assert result == {"text": "processed:INPUT!"}
            assert recorder.events == ("pre_request", "pre_api_call", "post_response")
        finally:
            REQUEST_CONTEXT.reset(token)

    await asyncio.gather(*(run_one(index) for index in range(32)))
    assert harness.calls() == 32


class CallbackAbort(BaseException):
    pass


async def callback_errors(harness: Harness) -> None:
    async def check(original: BaseException) -> None:
        async def reject(request: Payload) -> Payload:
            raise original

        try:
            await harness.execute(replace(Recorder().adapter(), pre_request=reject))
        except BaseException as error:
            assert error is original
            assert error.__traceback__ is not None
            frames: Final = inspect.getinnerframes(error.__traceback__)
            assert "reject" in tuple(frame.function for frame in frames)
        else:
            raise AssertionError("callback failure was lost")

    await check(ValueError("rejected"))
    await check(CallbackAbort("abort"))
    await check(asyncio.CancelledError())
    assert harness.calls() == 0


async def retained_callbacks(harness: Harness) -> None:
    def start() -> tuple[Awaitable[Payload], weakref.ReferenceType[Recorder]]:
        recorder: Final = Recorder()
        return harness.execute(recorder.adapter()), weakref.ref(recorder)

    future, reference = start()
    gc.collect()
    assert reference() is not None
    assert await future == {"text": "processed:INPUT!"}
    gc.collect()
    assert reference() is None


async def registration_and_return_contracts(harness: Harness) -> None:
    adapter: Final = Recorder().adapter()
    for invalid in (object(), replace(adapter, pre_api_call=None)):
        try:
            harness.execute(invalid)
        except (AttributeError, TypeError):
            pass
        else:
            raise AssertionError("invalid adapter accepted")

    async def wrong_shape(request: Payload) -> int:
        return 42

    async def extra_field(request: Payload) -> Payload:
        return {"text": "input", "unknown": "rejected"}

    def not_awaitable(request: Payload) -> Payload:
        return request

    async def unfinished() -> None:
        pass

    coroutine: Final = unfinished()

    def wrong_direct(event: BeforeSend) -> object:
        return coroutine

    def wrong_observer(event: BeforeSend) -> int:
        return 42

    for invalid_adapter, expected in (
        (replace(adapter, pre_request=wrong_shape), "typed contract"),
        (replace(adapter, pre_request=extra_field), "typed contract"),
        (replace(adapter, pre_request=not_awaitable), "non-awaitable"),
        (replace(adapter, pre_api_call=wrong_direct), "direct hook returned an awaitable"),
        (replace(adapter, pre_api_call=wrong_observer), "typed contract"),
    ):
        try:
            await harness.execute(invalid_adapter)
        except TypeError as error:
            assert expected in str(error)
        else:
            raise AssertionError("invalid callback result accepted")
    assert inspect.getcoroutinestate(coroutine) == inspect.CORO_CLOSED
    assert harness.calls() == 0

    def future_value(request: Payload) -> asyncio.Future[Payload]:
        result: Final[asyncio.Future[Payload]] = asyncio.get_running_loop().create_future()
        result.set_result(request)
        return result

    assert await harness.execute(replace(adapter, pre_request=future_value)) == {"text": "processed:input!"}
    assert harness.calls() == 1


async def provider_failure(harness: Harness) -> None:
    recorder: Final = Recorder()
    original: Final = LookupError("observer failed")

    def fail_observer(error: Payload) -> None:
        recorder.failure(error)
        raise original

    try:
        await harness.execute(replace(recorder.adapter(), failure=fail_observer), fail=True)
    except RuntimeError as error:
        assert str(error) == "provider failed"
        assert getattr(error, "observer_error") is original
    else:
        raise AssertionError("provider error was lost")
    assert recorder.events == ("pre_request", "pre_api_call", "failure")
    assert harness.calls() == 1


async def interrupted_session(harness: Harness) -> None:
    started: Final = asyncio.Event()
    stopped: Final = asyncio.Event()
    cleaned: Final = asyncio.Event()

    async def block(request: Payload) -> Payload:
        started.set()
        try:
            await asyncio.Future[None]()
        finally:
            cleaned.set()
        return request

    task: Final = asyncio.ensure_future(
        harness.interrupt(replace(Recorder().adapter(), pre_request=block), stopped.wait())
    )
    await started.wait()
    stopped.set()
    try:
        await task
    except RuntimeError as error:
        assert str(error) == "callback session was cancelled"
    else:
        raise AssertionError("interrupted session was reused")
    await cleaned.wait()
    assert harness.calls() == 0


async def cancellation_and_admission(harness: Harness) -> None:
    started: Final = asyncio.Event()
    cleaning: Final = asyncio.Event()
    release: Final = asyncio.Event()
    finished: Final = asyncio.Event()
    recorder: Final = Recorder()

    async def block(request: Payload) -> Payload:
        started.set()
        try:
            await asyncio.Future[None]()
        finally:
            cleaning.set()
            await release.wait()
            finished.set()
        return request

    async def assert_full() -> None:
        try:
            await harness.execute(Recorder().adapter())
        except RuntimeError as error:
            assert str(error) == "callback capacity exhausted"
        else:
            raise AssertionError("capacity released before callback completion")

    task: Final = asyncio.ensure_future(harness.execute(replace(recorder.adapter(), pre_request=block)))
    await started.wait()
    await assert_full()
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    else:
        raise AssertionError("request ignored cancellation")
    await cleaning.wait()
    await assert_full()
    assert harness.calls() == 0
    assert recorder.events == ()
    release.set()
    await finished.wait()
    assert await harness.execute(Recorder().adapter()) == {"text": "processed:INPUT!"}
    assert harness.calls() == 1


@dataclass(frozen=True, slots=True)
class DirectAdapter:
    transform: Callable[[Payload], object]


async def synchronous_callbacks(harness: Harness) -> None:
    def on_caller_thread() -> None:
        token: Final = REQUEST_CONTEXT.set("synchronous")
        caller: Final = threading.get_ident()

        def transform(request: Payload) -> Payload:
            assert REQUEST_CONTEXT.get() == "synchronous"
            assert threading.get_ident() == caller
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                pass
            else:
                raise AssertionError("sync callback invented an event loop")
            return {"text": request["text"].upper()}

        try:
            assert harness.sync(DirectAdapter(transform)) == {"text": "INPUT"}
        finally:
            REQUEST_CONTEXT.reset(token)

    await asyncio.to_thread(on_caller_thread)

    original: Final = ValueError("sync failure")

    def reject(request: Payload) -> Payload:
        raise original

    try:
        harness.sync(DirectAdapter(reject))
    except ValueError as error:
        assert error is original
    else:
        raise AssertionError("sync callback error was lost")

    def reenter(request: Payload) -> Payload:
        return harness.sync(DirectAdapter(lambda value: value))

    try:
        harness.sync(DirectAdapter(reenter))
    except RuntimeError as error:
        assert "Tokio context" in str(error)
    else:
        raise AssertionError("synchronous re-entry was accepted")

    async def unfinished() -> None:
        pass

    coroutine: Final = unfinished()
    try:
        harness.sync(DirectAdapter(lambda value: coroutine))
    except TypeError as error:
        assert str(error) == "direct hook returned an awaitable"
    else:
        raise AssertionError("sync callback accepted an awaitable")
    assert inspect.getcoroutinestate(coroutine) == inspect.CORO_CLOSED


TESTS: Final = (
    transforms_and_context,
    callback_errors,
    retained_callbacks,
    registration_and_return_contracts,
    provider_failure,
    cancellation_and_admission,
    interrupted_session,
    synchronous_callbacks,
)


def run(name: str, harness: Harness) -> None:
    test: Final = next(test for test in TESTS if test.__name__ == name)
    asyncio.run(asyncio.wait_for(test(harness), timeout=10))
