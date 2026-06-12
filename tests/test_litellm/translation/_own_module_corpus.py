"""Shared v1 invokers for the wave-2b own-module differential gates.

Each wave-2b-alpha provider (deepseek, openrouter, hosted_vllm,
fireworks_ai, snowflake, huggingface) rides a dedicated httpx
``completion()`` elif into ``base_llm_http_handler``, so the v1 side is the
xai/compat_httpx invoker shape, parameterized by provider:

- requests: ``get_optional_params(custom_llm_provider=p)`` with
  completion()'s ``stream=None`` default, then the handler's ``extra_body``
  pop (hh:399 pops it BEFORE transform and merges its CONTENTS top-level
  AFTER, hh:448 — callers that need the wire-level merge pin it with the
  mock-transport helper below), then the LIVE ``transform_request`` of the
  config production threads (``litellm_params`` passes through for the
  providers whose transform reads it — huggingface's api_base arm).
- responses: the config's ``transform_response`` over an ``httpx.Response``
  with a FRESH ``ModelResponse`` (no model preset on this path — xai R4).
- streams: SSE ``data:`` lines through the config's
  ``get_model_response_iterator`` + ``CustomStreamWrapper``.

May RAISE UnsupportedParamsError — that IS the pinned v1 behavior for the
supported-list gate rows.
"""

import copy
import json
import time
from typing import Any, Dict, List, Optional

import httpx

import litellm
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
from litellm.llms.custom_httpx.http_handler import HTTPHandler
from litellm.types.utils import LlmProviders, ModelResponse
from litellm.utils import ProviderConfigManager, get_optional_params

_Case = dict[str, object]


def provider_config(provider: str, model: str):
    return ProviderConfigManager.get_provider_chat_config(
        model=model, provider=LlmProviders(provider)
    )


def run_v1_request_transform(
    provider: str, case: _Case, litellm_params: Optional[dict] = None
) -> dict:
    request = copy.deepcopy(case)
    model = request.pop("model")
    messages = request.pop("messages")
    optional_params = get_optional_params(
        model=model,
        custom_llm_provider=provider,
        messages=copy.deepcopy(messages),
        stream=request.pop("stream", None),
        **request,
    )
    optional_params.pop("extra_body", None)
    config = provider_config(provider, model)
    return config.transform_request(
        model=model,
        messages=messages,
        optional_params=optional_params,
        litellm_params=litellm_params or {},
        headers={},
    )


def make_logging(model: str, messages: List[dict], stream: bool = False) -> Logging:
    logging_obj = Logging(
        model=model,
        messages=messages,
        stream=stream,
        call_type="completion",
        start_time=time.time(),
        litellm_call_id="diff-own-module-call-id",
        function_id="diff-own-module-function-id",
    )
    logging_obj.update_environment_variables(
        model=model, user=None, optional_params={}, litellm_params={}
    )
    return logging_obj


def run_v1_response_transform(
    provider: str,
    provider_response: Dict[str, Any],
    model: str,
    optional_params: Optional[dict] = None,
) -> ModelResponse:
    messages = [{"role": "user", "content": "hi"}]
    raw_response = httpx.Response(
        status_code=200,
        json=copy.deepcopy(provider_response),
        request=httpx.Request("POST", "https://example.test/v1/chat/completions"),
    )
    return provider_config(provider, model).transform_response(
        model=model,
        raw_response=raw_response,
        model_response=ModelResponse(),
        logging_obj=make_logging(model, messages),
        request_data={},
        messages=messages,
        optional_params=optional_params or {},
        litellm_params={},
        encoding=litellm.encoding,
        api_key=None,
        json_mode=None,
    )


def replay_v1_sse_lines(
    provider: str,
    events: List[dict],
    stream_model: str,
    stream_options: Optional[dict] = None,
) -> List[dict]:
    lines = [f"data: {json.dumps(event)}" for event in copy.deepcopy(events)]
    lines.append("data: [DONE]")
    handler = provider_config(provider, stream_model).get_model_response_iterator(
        streaming_response=iter(lines), sync_stream=True
    )
    wrapper = CustomStreamWrapper(
        completion_stream=handler,
        model=stream_model,
        custom_llm_provider=provider,
        logging_obj=make_logging(
            stream_model, [{"role": "user", "content": "stream"}], stream=True
        ),
        stream_options=stream_options,
    )
    return [chunk.model_dump() for chunk in wrapper]


def capture_v1_wire_body(
    model: str, api_base: Optional[str] = None, **kwargs: object
) -> dict:
    """The full v1 stack (completion() -> hh) against a mock transport,
    returning the EXACT JSON body POSTed to the wire. This is the
    wire-level truth the transform-seam invoker cannot see (hh merges
    ``extra_body`` contents top-level AFTER transform_request, hh:448) —
    the top_k extra_body rows pin against it."""
    captured: dict = {}

    def transport_handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode())
        return httpx.Response(
            200,
            json={
                "id": "wire-1",
                "object": "chat.completion",
                "created": 1718000000,
                "model": "m",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            },
        )

    client = HTTPHandler(
        client=httpx.Client(transport=httpx.MockTransport(transport_handler))
    )
    litellm.completion(
        model=model,
        messages=[{"role": "user", "content": "hi"}],
        api_key="wire-test-key",
        api_base=api_base,
        client=client,
        **kwargs,
    )
    return captured["body"]
