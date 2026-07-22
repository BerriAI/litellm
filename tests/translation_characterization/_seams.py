"""Invokers for the REAL v1 transform seams, called the way completion() calls them.

Request path mirrors ``main.py`` + ``base_llm_http_handler``:
``get_llm_provider`` -> ``get_optional_params`` (provider ``map_openai_params``)
-> ``get_litellm_params`` -> ``ProviderConfigManager.get_provider_chat_config``
-> ``validate_environment`` -> ``transform_request``.

Nothing in the transforms is mocked; the only stubbed ambient context is the
LiteLLM ``Logging`` object (the real class, fixed call ids) and an
``httpx.Response`` carrying the recorded provider payload.
"""

import copy
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

import httpx

import litellm
from litellm.litellm_core_utils.get_litellm_params import get_litellm_params
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager, get_optional_params

PROVIDERS: Dict[str, str] = {
    "anthropic": "anthropic/claude-sonnet-4-20250514",
    "bedrock_converse": "bedrock/anthropic.claude-sonnet-4-20250514-v1:0",
    "bedrock_invoke": "bedrock/invoke/anthropic.claude-3-7-sonnet-20250219-v1:0",
}


def resolve(provider_key: str) -> Tuple[str, str, Any]:
    """(model, custom_llm_provider, provider chat config) — v1's own resolution."""
    model, custom_llm_provider, _, _ = litellm.get_llm_provider(
        model=PROVIDERS[provider_key]
    )
    config = ProviderConfigManager.get_provider_chat_config(
        model=model, provider=LlmProviders(custom_llm_provider)
    )
    assert config is not None
    return model, custom_llm_provider, config


def make_logging(model: str, messages: List[dict], stream: bool = False) -> Logging:
    return Logging(
        model=model,
        messages=messages,
        stream=stream,
        call_type="completion",
        start_time=time.time(),
        litellm_call_id="char-litellm-call-id",
        function_id="char-function-id",
    )


def run_request_transform(provider_key: str, case: Dict[str, Any]) -> Dict[str, Any]:
    model, custom_llm_provider, config = resolve(provider_key)
    messages = copy.deepcopy(case["messages"])
    optional_params = get_optional_params(
        model=model,
        custom_llm_provider=custom_llm_provider,
        messages=messages,
        **copy.deepcopy(case["params"]),
    )
    litellm_params = get_litellm_params(custom_llm_provider=custom_llm_provider)
    headers = config.validate_environment(
        headers={},
        model=model,
        messages=messages,
        optional_params=optional_params,
        litellm_params=litellm_params,
        api_key=None if provider_key.startswith("bedrock") else "sk-ant-char-test-key",
        api_base=None,
    )
    return config.transform_request(
        model=model,
        messages=messages,
        optional_params=optional_params,
        litellm_params=litellm_params,
        headers=headers,
    )


def run_response_transform(
    provider_key: str,
    provider_response: Dict[str, Any],
    messages: List[dict],
    optional_params: Optional[Dict[str, Any]] = None,
) -> litellm.ModelResponse:
    model, _, config = resolve(provider_key)
    raw_response = httpx.Response(
        status_code=200,
        json=provider_response,
        request=httpx.Request("POST", "https://characterization.invalid/v1/messages"),
    )
    return config.transform_response(
        model=model,
        raw_response=raw_response,
        model_response=litellm.ModelResponse(),
        logging_obj=make_logging(model, messages),
        request_data={},
        messages=messages,
        optional_params=optional_params or {},
        litellm_params={},
        encoding=litellm.encoding,
        api_key=None,
        json_mode=None,
    )


def _wrap_stream(
    provider_key: str, model: str, completion_stream: Iterable[Any]
) -> List[dict]:
    custom_llm_provider = (
        "anthropic" if provider_key == "anthropic" else "bedrock"
    )
    wrapper = CustomStreamWrapper(
        completion_stream=iter(completion_stream),
        model=model,
        custom_llm_provider=custom_llm_provider,
        logging_obj=make_logging(
            model, [{"role": "user", "content": "stream"}], stream=True
        ),
    )
    return [chunk.model_dump() for chunk in wrapper]


def replay_anthropic_sse(sse_lines: List[str]) -> List[dict]:
    """Recorded anthropic SSE lines -> provider iterator -> CustomStreamWrapper."""
    from litellm.llms.anthropic.chat.handler import ModelResponseIterator

    model, _, _ = resolve("anthropic")
    iterator = ModelResponseIterator(
        streaming_response=iter(sse_lines), sync_stream=True
    )
    return _wrap_stream("anthropic", model, iterator)


def replay_bedrock_converse_events(events: List[dict]) -> List[dict]:
    """Parsed converse stream event payloads -> AWSEventStreamDecoder parser.

    Pinned at the parsed-event seam: the binary AWS event-stream framing that
    precedes ``converse_chunk_parser`` in production is botocore plumbing, not
    translation logic (documented in README.md).
    """
    from litellm.llms.bedrock.chat.invoke_handler import AWSEventStreamDecoder

    model, _, _ = resolve("bedrock_converse")
    decoder = AWSEventStreamDecoder(model=model)
    chunks = (decoder.converse_chunk_parser(event) for event in events)
    return _wrap_stream("bedrock_converse", model, chunks)


def replay_bedrock_invoke_events(events: List[dict]) -> List[dict]:
    """Parsed invoke (anthropic messages) event payloads -> claude stream decoder."""
    from litellm.llms.bedrock.chat.invoke_handler import (
        AmazonAnthropicClaudeStreamDecoder,
    )

    model, _, _ = resolve("bedrock_invoke")
    decoder = AmazonAnthropicClaudeStreamDecoder(model=model, sync_stream=True)
    chunks = (decoder._chunk_parser(event) for event in events)
    return _wrap_stream("bedrock_invoke", model, chunks)
