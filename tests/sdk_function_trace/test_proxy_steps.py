from tests.sdk_function_trace.profiler import FunctionTraceEvent
from tests.sdk_function_trace.proxy_steps import (
    python_ocr_proxy_issues,
    python_ocr_proxy_steps,
)


def test_should_project_python_ocr_from_http_endpoint() -> None:
    events = (
        FunctionTraceEvent("proxy/auth/user_api_key_auth.py:2846 user_api_key_auth", 6),
        FunctionTraceEvent("proxy/ocr_endpoints/endpoints.py:252 ocr", 6),
        FunctionTraceEvent(
            "proxy/ocr_endpoints/endpoints.py:157 _parse_ocr_request", 7
        ),
        FunctionTraceEvent(
            "proxy/ocr_endpoints/endpoints.py:162 _parse_ocr_request_body", 8
        ),
        FunctionTraceEvent(
            "proxy/ocr_endpoints/endpoints.py:51 _with_request_format", 8
        ),
        FunctionTraceEvent(
            "proxy/common_request_processing.py:2174 ProxyBaseLLMRequestProcessing.base_process_llm_request",
            7,
        ),
        FunctionTraceEvent(
            "proxy/common_request_processing.py:2239 ProxyBaseLLMRequestProcessing._process_llm_request",
            9,
        ),
        FunctionTraceEvent("proxy/route_llm_request.py:423 route_request", 10),
        FunctionTraceEvent("ocr/main.py:331 aocr", 1),
        FunctionTraceEvent(
            "llms/custom_httpx/http_handler.py:654 AsyncHTTPHandler.post", 4
        ),
        FunctionTraceEvent(
            "proxy/utils.py:2757 ProxyLogging.post_call_success_hook", 10
        ),
        FunctionTraceEvent("proxy/ocr_endpoints/endpoints.py:71 _native_response", 7),
    )

    steps = python_ocr_proxy_steps(events)

    assert steps[0] == FunctionTraceEvent("POST /ocr", 0)
    assert FunctionTraceEvent("user_api_key_auth", 1) in steps
    assert FunctionTraceEvent("ocr_fastapi_endpoint", 1) in steps
    assert FunctionTraceEvent("route_request", 4) in steps
    assert FunctionTraceEvent("aocr", 6) in steps
    assert FunctionTraceEvent("http_request", 9) in steps
    assert FunctionTraceEvent("build_proxy_ocr_response", 2) in steps
    assert "missing parse_ocr_request" not in python_ocr_proxy_issues(steps)
