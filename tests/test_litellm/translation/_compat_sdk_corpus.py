"""Shared plumbing for the compat_sdk (wave-1a) differential gates.

The v1 request invoker mirrors the SDK path exactly as production runs it
for these providers (the big openai elif, main.py:2646-2667):
``get_optional_params(custom_llm_provider=p)`` — the RAISE-unless-drop_params
``_check_valid_arg`` gate over the provider config's supported list plus the
config's ``map_openai_params`` — then the resolved provider config's
``transform_request`` (openai.py:727; the inherited base five-touch
assembly). ``extra_body`` is popped after get_optional_params: the injected
``{}`` merges nothing onto the wire (the SDK spreads its CONTENTS top-level,
same as the xai corpus invoker documents for hh:398-399).

May RAISE UnsupportedParamsError — that IS the pinned v1 behavior for the
supported-list gate rows (the xai R2 pattern: assert the raise, never a
remap).
"""

import copy
from collections.abc import Mapping
from types import MappingProxyType
from typing import Literal, NamedTuple

from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager, get_optional_params

_Case = dict[str, object]


class CompatSpec(NamedTuple):
    """One row per surviving wave-1a provider: the corpus model, which
    optional surfaces the provider serves (drives generated
    corpus/raise/fallback rows), and how max_completion_tokens behaves in
    v1 ("rename" -> max_tokens, "verbatim" -> passes through, "raise" ->
    outside the supported list)."""

    model: str
    tools: bool
    response_format: bool
    parallel_tool_calls: bool
    user: bool
    mct: Literal["rename", "verbatim", "raise"]


SPECS: Mapping[str, CompatSpec] = MappingProxyType(
    {
        "together_ai": CompatSpec(
            model="Qwen/Qwen2.5-72B-Instruct-Turbo",  # map: supports_function_calling
            tools=True,
            response_format=True,
            parallel_tool_calls=True,
            user=False,
            mct="verbatim",
        ),
        "cerebras": CompatSpec(
            model="llama3.1-8b",
            tools=True,
            response_format=True,
            parallel_tool_calls=False,
            user=True,
            mct="rename",
        ),
        "nvidia_nim": CompatSpec(
            model="meta/llama3-70b-instruct",  # the default-list arm
            tools=True,
            response_format=True,
            parallel_tool_calls=True,
            user=False,
            mct="rename",
        ),
        "lm_studio": CompatSpec(
            model="qwen2.5-7b-instruct-1m",
            tools=True,
            response_format=True,
            parallel_tool_calls=True,
            user=False,
            mct="verbatim",
        ),
        "llamafile": CompatSpec(
            model="LLaMA_CPP",
            tools=True,
            response_format=True,
            parallel_tool_calls=True,
            user=False,
            mct="verbatim",
        ),
        "lambda_ai": CompatSpec(
            model="llama3.1-70b-instruct-fp8",
            tools=True,
            response_format=True,
            parallel_tool_calls=True,
            user=False,
            mct="rename",
        ),
        "nebius": CompatSpec(
            model="meta-llama/Meta-Llama-3.1-70B-Instruct",
            tools=True,
            response_format=True,
            parallel_tool_calls=True,
            user=False,
            mct="rename",
        ),
        "novita": CompatSpec(
            model="meta-llama/llama-3.1-8b-instruct",
            tools=True,
            response_format=True,
            parallel_tool_calls=True,
            user=False,
            mct="verbatim",
        ),
        "wandb": CompatSpec(
            model="meta-llama/Llama-3.1-8B-Instruct",
            tools=True,
            response_format=True,
            parallel_tool_calls=True,
            user=False,
            mct="rename",
        ),
        "featherless_ai": CompatSpec(
            model="featherless-ai/Qwerky-72B",
            tools=False,
            response_format=False,
            parallel_tool_calls=False,
            user=False,
            mct="rename",
        ),
        "nscale": CompatSpec(
            model="meta-llama/Llama-4-Scout-17B-16E-Instruct",
            tools=False,
            response_format=True,
            parallel_tool_calls=False,
            user=False,
            mct="raise",
        ),
        "hyperbolic": CompatSpec(
            model="meta-llama/Meta-Llama-3-70B-Instruct",
            tools=True,
            response_format=True,
            parallel_tool_calls=False,
            user=True,
            mct="raise",
        ),
        "volcengine": CompatSpec(
            model="doubao-pro-32k-241215",
            tools=True,
            response_format=False,
            parallel_tool_calls=False,
            user=False,
            mct="rename",
        ),
    }
)

PROVIDERS = tuple(sorted(SPECS))

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


def corpus_for(provider: str) -> dict[str, _Case]:
    """The generated served corpus: every row here must be byte-identical
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
        "temperature_int_stays_int": {
            "model": model,
            "temperature": 1,
            "messages": user_msg,
        },
    }
    if spec.mct in ("rename", "verbatim"):
        cases["max_completion_tokens"] = {
            "model": model,
            "max_completion_tokens": 128,
            "messages": user_msg,
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
            "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
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
                    "name": "answer",
                    "schema": {
                        "type": "object",
                        "properties": {"capital": {"type": "string"}},
                        "required": ["capital"],
                        "additionalProperties": False,
                    },
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
    if spec.user:
        cases["user_param"] = {"model": model, "user": "u-1", "messages": user_msg}
    return cases
