from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

from tests.sdk_function_trace.profiler import FunctionTraceEvent


@dataclass(frozen=True, slots=True)
class ProxyStep:
    name: str
    python: re.Pattern[str]
    depth: int


def _step(name: str, python: str, depth: int) -> ProxyStep:
    return ProxyStep(name=name, python=re.compile(python), depth=depth)


PYTHON_OCR_PROXY_STEPS: Final = (
    _step("user_api_key_auth", r"user_api_key_auth\.py:\d+ user_api_key_auth$", 1),
    _step(
        "read_request_body_for_auth", r"_read_request_body_deferring_parse_failure$", 2
    ),
    _step("build_api_key_auth", r"_user_api_key_auth_builder$", 2),
    _step("authorize_request", r"_authorize_authenticated_request$", 2),
    _step("run_common_auth_checks", r"_run_centralized_common_checks$", 3),
    _step("ocr_fastapi_endpoint", r"proxy/ocr_endpoints/endpoints\.py:\d+ ocr$", 1),
    _step("parse_ocr_request", r"_parse_ocr_request$", 2),
    _step("parse_ocr_request_body", r"_parse_ocr_request_body$", 3),
    _step("resolve_request_format", r"_with_request_format$", 3),
    _step(
        "base_process_llm_request",
        r"ProxyBaseLLMRequestProcessing\.base_process_llm_request$",
        2,
    ),
    _step(
        "process_llm_request",
        r"ProxyBaseLLMRequestProcessing\._process_llm_request$",
        3,
    ),
    _step(
        "proxy_pre_call", r"ProxyBaseLLMRequestProcessing\._pre_call_with_fallbacks$", 4
    ),
    _step(
        "common_proxy_pre_call",
        r"ProxyBaseLLMRequestProcessing\.common_processing_pre_call_logic$",
        5,
    ),
    _step("add_proxy_request_metadata", r"add_litellm_data_to_request$", 6),
    _step("route_request", r"proxy/route_llm_request\.py:\d+ route_request$", 4),
    _step("route_request_single_attempt", r"_route_request_single_attempt$", 5),
    _step("aocr", r"ocr/main\.py:\d+ aocr$", 6),
    _step("prepare_ocr_call", r"ocr/main\.py:\d+ _prepare_ocr_request$", 7),
    _step(
        "get_provider_ocr_config", r"ProviderConfigManager\.get_provider_ocr_config$", 8
    ),
    _step("supported_ocr_params", r"get_supported_ocr_params$", 9),
    _step("map_ocr_params", r"(?<!async_)map_ocr_params$", 9),
    _step("execute_ocr_provider_call", r"BaseLLMHTTPHandler\.async_ocr$", 7),
    _step("validate_environment", r"(?<!_)validate_environment$", 8),
    _step("complete_url", r"get_complete_url$", 8),
    _step("transform_ocr_request", r"(?<!async_)transform_ocr_request$", 9),
    _step("http_request", r"AsyncHTTPHandler\.post$", 9),
    _step("transform_ocr_response", r"(?<!async_)transform_ocr_response$", 8),
    _step("proxy_post_call_success", r"ProxyLogging\.post_call_success_hook$", 4),
    _step("build_proxy_ocr_response", r"_native_response$", 2),
)


def python_ocr_proxy_steps(
    events: tuple[FunctionTraceEvent, ...]
) -> tuple[FunctionTraceEvent, ...]:
    projected = (FunctionTraceEvent("POST /ocr", 0),)
    for step in PYTHON_OCR_PROXY_STEPS:
        if any(step.python.search(event.function) for event in events):
            projected += (FunctionTraceEvent(step.name, step.depth),)
    return projected


def python_ocr_proxy_issues(steps: tuple[FunctionTraceEvent, ...]) -> tuple[str, ...]:
    names: Final = {event.function for event in steps}
    return tuple(
        f"missing {step.name}"
        for step in PYTHON_OCR_PROXY_STEPS
        if step.name not in names
    )
