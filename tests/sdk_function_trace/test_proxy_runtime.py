from tests.sdk_function_trace.proxy_runtime import run_python_ocr_proxy_trace
from tests.sdk_function_trace.proxy_steps import (
    python_ocr_proxy_issues,
    python_ocr_proxy_steps,
)


def test_should_trace_fastapi_ocr_through_provider_and_proxy_response() -> None:
    steps = python_ocr_proxy_steps(run_python_ocr_proxy_trace())

    assert python_ocr_proxy_issues(steps) == ()
