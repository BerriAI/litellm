import asyncio
import json
import sys
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Final

from callback_support import (
    DOCUMENT,
    MODEL,
    REQUEST_CONTEXT,
    RESPONSE,
    CallbackRecorder,
    OcrArguments,
    assert_provider_request,
    call_ocr,
    ocr_upstream,
    verify_installed_package,
)
from pydantic import TypeAdapter

import litellm
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
from litellm.rust_bridge.ocr import supports_callback_adapter
from litellm.rust_bridge.ocr_callbacks import PreCallArguments


@dataclass(frozen=True, slots=True)
class Outcome:
    response: str | None
    error_type: str | None
    status: int | None
    events: tuple[str, ...]
    response_types: tuple[str, ...]
    provider_body: str


async def exercise(asynchronous: bool, rust: bool, case: str, *, native_expected: bool | None = None) -> Outcome:
    litellm.logging_callback_manager._reset_all_callbacks()  # pyright: ignore[reportPrivateUsage]  # isolate SDK cases in this dedicated test process
    native: Final = rust if native_expected is None else native_expected
    if native:
        assert supports_callback_adapter(asynchronous=asynchronous), "native callback bridge must be installed"
    status: Final = int(case) if case.isdigit() else 500 if case in ("raise_failure", "raise_async_failure") else 200
    body: Final = "invalid-json" if case == "malformed" else RESPONSE if status == 200 else '{"error":"rejected"}'
    successful: Final = status == 200 and case not in ("malformed", "timeout")
    raises: Final = case.removeprefix("raise_") if case.startswith("raise_") else ""
    recorder: Final = CallbackRecorder(asynchronous, raises=raises, name=f"first-{asynchronous}-{rust}-{case}")
    follower: Final = CallbackRecorder(asynchronous, name=f"second-{asynchronous}-{rust}-{case}")
    context: Final = f"request-{asynchronous}-{rust}-{case}"
    token: Final = REQUEST_CONTEXT.set(context)
    try:
        with ocr_upstream(status, body, stall=case == "timeout") as upstream:
            params: Final[OcrArguments] = {
                "model": MODEL,
                "document": DOCUMENT,
                "api_key": "test-key",
                "api_base": upstream.api_base,
                "callbacks": [recorder, follower],
                "rust": rust,
                "num_retries": 0,
                "timeout": 0.2 if case == "timeout" else 5,
            }
            result, error = await call_ocr(params, asynchronous)
            assert (error is None) is successful
            if error is not None:
                if case.isdigit():
                    assert getattr(error, "status_code", None) == status
                if case == "timeout":
                    assert isinstance(error, litellm.Timeout)
            if result is not None:
                assert result.pages[0].markdown == "callback-test"
            await asyncio.wait_for(GLOBAL_LOGGING_WORKER.flush(), 5)
            events: Final = await recorder.wait()
            following: Final = await follower.wait()
            names: Final = tuple(event.name for event in events)
            prefix: Final = ("pre", "post") if status == 200 and case != "timeout" else ("pre",)
            assert names[: len(prefix)] == prefix
            terminal: Final = "success" if successful else "failure"
            expected: Final = (
                ("async_success",)
                if asynchronous and successful
                else ((terminal, f"async_{terminal}") if asynchronous else (terminal,))
            )
            assert sorted(names[len(prefix) :]) == sorted(expected)
            assert sorted(event.name for event in following) == sorted(names)
            assert len({event.call_id for event in events}) == 1
            assert isinstance(events[0].call_id, str) and events[0].call_id
            assert all(event.model == "mistral-ocr-4-1" for event in events)
            assert events[0].input == "OCR document processing"
            pre_arguments: Final = TypeAdapter(PreCallArguments).validate_json(events[0].additional_args)
            assert pre_arguments["complete_input_dict"] == {"model": "mistral-ocr-4-1", "document": DOCUMENT}
            assert pre_arguments["api_base"] == upstream.api_base + "/ocr"
            assert {key.lower(): value for key, value in pre_arguments["headers"].items()}[
                "authorization"
            ] == "Bearer test-key"
            for event in events[: len(prefix)]:
                assert event.context == context
                assert event.native_provider_hook is native
            if "post" in prefix:
                assert events[1].original_response == body
                assert events[1].response_type == "NoneType"
                assert events[1].start_time is not None and events[1].end_time is None
            for event in events[len(prefix) :]:
                assert event.start_time is not None and event.end_time is not None
                assert event.end_time >= event.start_time
            assert_provider_request(upstream)
            return Outcome(
                result.model_dump_json() if result is not None else None,
                type(error).__name__ if error is not None else None,
                getattr(error, "status_code", None),
                tuple(sorted(names)),
                tuple(sorted(event.response_type for event in events)),
                json.dumps(json.loads(upstream.requests[0][1]), sort_keys=True),
            )
    finally:
        REQUEST_CONTEXT.reset(token)


async def verify_parity() -> None:
    for asynchronous in (False, True):
        for case in (
            "success",
            "malformed",
            "401",
            "429",
            "500",
            "timeout",
            "raise_pre",
            "raise_post",
            "raise_success",
            "raise_async_success",
            "raise_failure",
            "raise_async_failure",
        ):
            await verify_case(asynchronous, case)
    await verify_concurrency()
    for rust in (False, True):
        await verify_delayed_terminal(rust)
    for rust in (False, True):
        await verify_without_loggers(rust)


async def verify_case(asynchronous: bool, case: str) -> None:
    python: Final = await exercise(asynchronous, False, case)
    native: Final = await exercise(asynchronous, True, case)
    assert python == native, (asynchronous, case, python, native)
    sys.stdout.write(f"PASS async={asynchronous} case={case}\n")
    sys.stdout.flush()


async def verify_concurrency() -> None:
    litellm.logging_callback_manager._reset_all_callbacks()  # pyright: ignore[reportPrivateUsage]  # isolate the concurrent SDK scenario
    recorder: Final = CallbackRecorder(True, name="concurrent", expected_calls=32)
    with ocr_upstream() as upstream:

        async def one(index: int) -> None:
            token: Final = REQUEST_CONTEXT.set(f"concurrent-{index}")
            try:
                result: Final = await litellm.aocr(
                    model=MODEL,
                    document=DOCUMENT,
                    api_key="test-key",
                    api_base=upstream.api_base,
                    callbacks=[recorder],
                    rust=True,
                    num_retries=0,
                    timeout=10,
                )
                assert result.pages[0].markdown == "callback-test"
            finally:
                REQUEST_CONTEXT.reset(token)

        await asyncio.gather(*(one(index) for index in range(32)))
        await asyncio.wait_for(GLOBAL_LOGGING_WORKER.flush(), 5)
        events: Final = await recorder.wait()
        assert len(upstream.requests) == 32
        assert Counter(event.name for event in events) == {"pre": 32, "post": 32, "async_success": 32}
        assert len({event.call_id for event in events}) == 32
        for index in range(32):
            assert tuple(event.name for event in events if event.context == f"concurrent-{index}") == (
                "pre",
                "post",
                "async_success",
            )
        assert all(event.native_provider_hook for event in events if event.name in ("pre", "post"))


class DelayedRecorder(CallbackRecorder):
    def __init__(self) -> None:
        super().__init__(True, name="delayed")
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def async_log_success_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: datetime, end_time: datetime
    ) -> None:
        self.started.set()
        await self.release.wait()
        await super().async_log_success_event(kwargs, response_obj, start_time, end_time)


async def verify_delayed_terminal(rust: bool) -> None:
    litellm.logging_callback_manager._reset_all_callbacks()  # pyright: ignore[reportPrivateUsage]  # isolate delayed delivery
    recorder: Final = DelayedRecorder()
    with ocr_upstream() as upstream:
        result: Final = await asyncio.wait_for(
            litellm.aocr(
                model=MODEL,
                document=DOCUMENT,
                api_key="test-key",
                api_base=upstream.api_base,
                callbacks=[recorder],
                rust=rust,
                num_retries=0,
            ),
            5,
        )
        assert result.pages[0].markdown == "callback-test"
        await asyncio.wait_for(recorder.started.wait(), 5)
        assert tuple(event.name for event in recorder.events) == ("pre", "post")
        recorder.release.set()
        await asyncio.wait_for(GLOBAL_LOGGING_WORKER.flush(), 5)
        events: Final = await recorder.wait()
        assert tuple(event.name for event in events) == ("pre", "post", "async_success"), events


async def verify_without_loggers(rust: bool) -> None:
    litellm.logging_callback_manager._reset_all_callbacks()  # pyright: ignore[reportPrivateUsage]  # exercise the no-logger configuration
    with ocr_upstream() as upstream:
        result: Final = await litellm.aocr(
            model=MODEL, document=DOCUMENT, api_key="test-key", api_base=upstream.api_base, rust=rust, num_retries=0
        )
        assert result.pages[0].markdown == "callback-test"
        assert_provider_request(upstream)
        await asyncio.wait_for(GLOBAL_LOGGING_WORKER.flush(), 5)


async def verify_unavailable() -> None:
    assert not supports_callback_adapter()
    for asynchronous in (False, True):
        for case in ("success", "401"):
            await exercise(asynchronous, True, case, native_expected=False)


if __name__ == "__main__":
    if "--installed" in sys.argv:
        verify_installed_package()
    asyncio.run(verify_unavailable() if "--without-native" in sys.argv else verify_parity())
