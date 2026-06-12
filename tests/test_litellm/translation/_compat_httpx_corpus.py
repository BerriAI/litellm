"""Shared plumbing for the compat_httpx (wave-1b) differential gates.

The v1 invokers mirror each provider's dedicated ``completion()`` elif into
``base_llm_http_handler`` (the xai corpus shape, NOT the SDK shape):

- requests: ``get_optional_params(custom_llm_provider=p)`` with
  completion()'s ``stream=None`` default, then the handler's ``extra_body``
  pop (hh pops the injected ``{}`` and merges its CONTENTS top-level), then
  the LIVE ``transform_request`` of the config production threads
  (``ProviderConfigManager`` for eight; ``LemonadeChatConfig()`` explicitly
  for lemonade, mirroring main.py's ``provider_config=lemonade_transformation``
  — its chat config is unregistered at HEAD, see the lemonade facts canary).
- responses: the config's ``transform_response`` over an ``httpx.Response``
  with a FRESH ``ModelResponse()`` (no model preset on this path — xai R4).
- streams: SSE ``data:`` lines through the config's
  ``get_model_response_iterator`` (the base
  ``OpenAIChatCompletionStreamingHandler`` for every member except
  cometapi's own strict handler — see BASE_HANDLER_PROVIDERS) +
  ``CustomStreamWrapper(custom_llm_provider=p)``.

May RAISE UnsupportedParamsError — that IS the pinned v1 behavior for the
supported-list gate rows.
"""

import copy
import json
import time
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, Dict, List, Literal, NamedTuple, Optional

import httpx

import litellm
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
from litellm.llms.lemonade.chat.transformation import LemonadeChatConfig
from litellm.types.utils import LlmProviders, ModelResponse
from litellm.utils import ProviderConfigManager, get_optional_params

_Case = dict[str, object]

STREAM_MODEL = "stream-model"


class HttpxSpec(NamedTuple):
    """One row per compat_httpx provider: the corpus model, the optional
    surfaces it serves, the mct behavior, the reasoning_effort surface
    ("none" raises, "capability" is model-map gated, "unconditional" is in
    the static list), and the v1 response-model prefix (None = bare wire
    model)."""

    model: str
    tools: bool
    response_format: bool
    parallel_tool_calls: bool
    mct: Literal["rename", "verbatim"]
    reasoning: Literal["none", "capability", "unconditional"] = "none"
    prefix: Optional[str] = None


SPECS: Mapping[str, HttpxSpec] = MappingProxyType(
    {
        "heroku": HttpxSpec(
            model="claude-3-7-sonnet",
            tools=True,
            response_format=True,
            parallel_tool_calls=True,
            mct="verbatim",
        ),
        "bedrock_mantle": HttpxSpec(
            model="openai.gpt-oss-120b",  # map: supports_reasoning
            tools=True,
            response_format=True,
            parallel_tool_calls=True,
            mct="rename",
            reasoning="capability",
        ),
        "minimax": HttpxSpec(
            model="MiniMax-M2",
            tools=True,
            response_format=True,
            parallel_tool_calls=True,
            mct="verbatim",
        ),
        "compactifai": HttpxSpec(
            model="cai-llama-3-1-8b-slim",
            tools=True,
            response_format=True,
            parallel_tool_calls=True,
            mct="verbatim",
            prefix="compactifai",
        ),
        "amazon_nova": HttpxSpec(
            model="nova-pro",
            tools=True,
            response_format=False,
            parallel_tool_calls=False,
            mct="rename",
            reasoning="unconditional",
            prefix="amazon-nova",  # the literal hyphenated string in v1
        ),
        "datarobot": HttpxSpec(
            model="datarobot-deployed-llm",
            tools=True,
            response_format=True,
            parallel_tool_calls=True,
            mct="rename",
        ),
        "gradient_ai": HttpxSpec(
            model="llama3.3-70b-instruct",
            tools=False,
            response_format=False,
            parallel_tool_calls=False,
            mct="verbatim",  # its own map never renames (no super call)
        ),
        "ovhcloud": HttpxSpec(
            model="Meta-Llama-3_1-70B-Instruct",
            tools=True,
            response_format=True,
            parallel_tool_calls=True,
            mct="verbatim",
        ),
        "lemonade": HttpxSpec(
            model="gpt-oss-20b-mxfp4-GGUF",
            tools=True,
            response_format=True,
            parallel_tool_calls=True,
            mct="rename",  # via the get_optional_params OpenAILike else-arm
            prefix="lemonade",
        ),
        # wave-2a's httpx member, a family row since the sibling merge
        # (critic-wave1b reconciliation; main.py:2547 elif).
        "cometapi": HttpxSpec(
            model="gpt-4o-mini",
            tools=True,
            response_format=True,
            parallel_tool_calls=True,
            mct="verbatim",  # map is super() over the base list
        ),
    }
)

PROVIDERS = tuple(sorted(SPECS))
# cometapi streams through its OWN CometAPIChatCompletionStreamingHandler
# (strict envelope, copy-both reasoning) — NOT the base handler the family
# stream gates replay; its line-seam gates are
# test_differential_cometapi_stream.py and its v2 parser is the family's
# LINE_PARSERS["cometapi"] policy row.
BASE_HANDLER_PROVIDERS = tuple(p for p in PROVIDERS if p != "cometapi")

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get weather",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}


def provider_config(provider: str, model: str):
    if provider == "lemonade":
        return LemonadeChatConfig()
    return ProviderConfigManager.get_provider_chat_config(
        model=model, provider=LlmProviders(provider)
    )


def run_v1_request_transform(provider: str, case: _Case) -> dict:
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
        litellm_params={},
        headers={},
    )


def make_logging(model: str, messages: List[dict], stream: bool = False) -> Logging:
    logging_obj = Logging(
        model=model,
        messages=messages,
        stream=stream,
        call_type="completion",
        start_time=time.time(),
        litellm_call_id="diff-compat-httpx-call-id",
        function_id="diff-compat-httpx-function-id",
    )
    logging_obj.update_environment_variables(
        model=model, user=None, optional_params={}, litellm_params={}
    )
    return logging_obj


def run_v1_response_transform(
    provider: str, provider_response: Dict[str, Any], model: str
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
        optional_params={},
        litellm_params={},
        encoding=litellm.encoding,
        api_key=None,
        json_mode=None,
    )


def replay_v1_sse_lines(
    provider: str, events: List[dict], stream_options: Optional[dict] = None
) -> List[dict]:
    lines = [f"data: {json.dumps(event)}" for event in copy.deepcopy(events)]
    lines.append("data: [DONE]")
    handler = provider_config(provider, STREAM_MODEL).get_model_response_iterator(
        streaming_response=iter(lines), sync_stream=True
    )
    wrapper = CustomStreamWrapper(
        completion_stream=handler,
        model=STREAM_MODEL,
        custom_llm_provider=provider,
        logging_obj=make_logging(
            STREAM_MODEL, [{"role": "user", "content": "stream"}], stream=True
        ),
        stream_options=stream_options,
    )
    return [chunk.model_dump() for chunk in wrapper]


def corpus_for(provider: str) -> dict[str, _Case]:
    """The generated served corpus: every row must be byte-identical
    (normalized JSON) between v1-in-process and v2."""
    spec = SPECS[provider]
    model = spec.model
    user_msg = [{"role": "user", "content": "Hello, world"}]
    cases: dict[str, _Case] = {
        "text": {"model": model, "messages": user_msg},
        "system_and_sampling": {
            "model": model,
            "max_tokens": 64,
            "temperature": 0.5,
            "top_p": 0.9,
            "messages": [
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "Hi"},
            ],
        },
        "stream_true": {"model": model, "stream": True, "messages": user_msg},
        "stop_list": {"model": model, "stop": ["END", "STOP"], "messages": user_msg},
        "max_completion_tokens": {
            "model": model,
            "max_completion_tokens": 128,
            "messages": user_msg,
        },
        # int temperature stays int through both sides (the json-dump pin;
        # added at the sibling merge so cometapi keeps its wave-2a coverage
        # and the family gains the row)
        "temperature_int_stays_int": {
            "model": model,
            "temperature": 1,
            "messages": user_msg,
        },
    }
    if spec.tools:
        cases["tools_auto"] = {
            "model": model,
            "tools": [WEATHER_TOOL],
            "tool_choice": "auto",
            "messages": [{"role": "user", "content": "Weather in Paris?"}],
        }
        cases["tool_choice_specific"] = {
            "model": model,
            "tools": [WEATHER_TOOL],
            "tool_choice": {
                "type": "function",
                "function": {"name": "get_weather"},
            },
            "messages": [{"role": "user", "content": "Weather in Paris?"}],
        }
        cases["tool_call_compact_roundtrip"] = {
            "model": model,
            "tools": [WEATHER_TOOL],
            "messages": [
                {"role": "user", "content": "w?"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city":"Paris"}',
                            },
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
            ],
        }
    if spec.response_format:
        cases["response_format_json_object"] = {
            "model": model,
            "response_format": {"type": "json_object"},
            "messages": user_msg,
        }
        cases["response_format_json_schema_strict"] = {
            "model": model,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "a",
                    "schema": {"type": "object"},
                    "strict": True,
                },
            },
            "messages": user_msg,
        }
    if spec.parallel_tool_calls:
        cases["parallel_tool_calls_false"] = {
            "model": model,
            "tools": [WEATHER_TOOL],
            "parallel_tool_calls": False,
            "messages": [{"role": "user", "content": "Weather in Paris and Rome?"}],
        }
    if spec.reasoning != "none":
        cases["reasoning_effort_served"] = {
            "model": model,
            "reasoning_effort": "high",
            "messages": user_msg,
        }
    return cases
