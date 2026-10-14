from collections.abc import Iterator, Mapping
from typing import Final

import pytest

from litellm.llms.base_llm.ocr.transformation import OCRResponse
from litellm.ocr.dispatch import AOCR, OCR
from litellm.rust_bridge.catalog import Route, RouteRule, Rules
from litellm.rust_bridge.configuration import Rollout
from litellm.rust_bridge.runtime import NoPythonImplementationError
from tests.test_litellm_rust.ocr.data import OCR_DOCUMENT, OCR_MODEL, OCR_RESPONSE, OCR_TEXT
from tests.test_litellm_rust.support.recording_server import RecordingServer, ResponseSpec, recording_service
from tests.test_litellm_rust.support.routing import assert_served_natively

pytest.importorskip("litellm.rust_bridge._native")

RUST: Final = (RouteRule(Route.OCR, Rollout.RUST_REQUIRED),)
CALLS: Final = pytest.mark.parametrize(
    ("args", "kwargs"),
    (
        pytest.param((OCR_MODEL, dict(OCR_DOCUMENT)), {}, id="positional"),
        pytest.param((), {"model": OCR_MODEL, "document": dict(OCR_DOCUMENT)}, id="keyword"),
    ),
)
REJECTED: Final = pytest.mark.parametrize(
    ("args", "kwargs", "rules", "error", "message"),
    (
        pytest.param(
            (OCR_MODEL, dict(OCR_DOCUMENT)), {"model": "duplicate"}, RUST, TypeError, "multiple values", id="duplicate"
        ),
        pytest.param((OCR_MODEL,), {}, RUST, TypeError, "missing 1 required positional", id="missing-document"),
        pytest.param(
            (OCR_MODEL, dict(OCR_DOCUMENT)),
            {},
            (RouteRule(Route.OCR, Rollout.PYTHON_ONLY),),
            NoPythonImplementationError,
            "RUST_REQUIRED",
            id="python-only-rule",
        ),
    ),
)


@pytest.fixture
def server() -> Iterator[RecordingServer]:
    with recording_service() as recording:
        recording.default_response = ResponseSpec(body=OCR_RESPONSE)
        yield recording


def upstream(server: RecordingServer, kwargs: Mapping[str, object]) -> Mapping[str, object]:
    return {**kwargs, "api_key": "test-key", "api_base": server.base_url}


def assert_served_natively_with_text(server: RecordingServer, response: object) -> None:
    assert isinstance(response, OCRResponse)
    assert response.pages[0].markdown == OCR_TEXT
    assert_served_natively(server)


@CALLS
def test_ocr_reaches_native_from_positional_and_keyword_calls(
    server: RecordingServer, args: tuple[object, ...], kwargs: Mapping[str, object]
) -> None:
    assert_served_natively_with_text(server, OCR.run(args, upstream(server, kwargs), rules=RUST))


@pytest.mark.asyncio
@CALLS
async def test_aocr_reaches_native_from_positional_and_keyword_calls(
    server: RecordingServer, args: tuple[object, ...], kwargs: Mapping[str, object]
) -> None:
    assert_served_natively_with_text(server, await AOCR.arun(args, upstream(server, kwargs), rules=RUST))


@REJECTED
def test_ocr_rejects_calls_before_reaching_the_provider(
    server: RecordingServer,
    args: tuple[object, ...],
    kwargs: Mapping[str, object],
    rules: Rules,
    error: type[Exception],
    message: str,
) -> None:
    server.expected_requests = 0
    with pytest.raises(error, match=message):
        OCR.run(args, upstream(server, kwargs), rules=rules)


@pytest.mark.asyncio
@REJECTED
async def test_aocr_rejects_calls_before_reaching_the_provider(
    server: RecordingServer,
    args: tuple[object, ...],
    kwargs: Mapping[str, object],
    rules: Rules,
    error: type[Exception],
    message: str,
) -> None:
    server.expected_requests = 0
    with pytest.raises(error, match=message):
        await AOCR.arun(args, upstream(server, kwargs), rules=rules)


def test_inner_ocr_call_of_aocr_cannot_reach_native(server: RecordingServer) -> None:
    server.expected_requests = 0
    with pytest.raises(NoPythonImplementationError, match="bypass"):
        OCR.run((OCR_MODEL, dict(OCR_DOCUMENT)), upstream(server, {"aocr": True}), rules=RUST)
