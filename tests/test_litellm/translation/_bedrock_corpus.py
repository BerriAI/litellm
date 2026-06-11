"""Shared plumbing for the bedrock differential gates.

The reference corpus under ``characterization_bedrock/`` is a verbatim copy of
the translation characterization corpus (mateo/translation-characterization
branch, tests/translation_characterization) for the two bedrock chat routes:
``cases/`` are OpenAI-format requests, ``snapshots/`` pin the REAL v1
transform output (canonical JSON: pretty-printed, sorted keys, trailing
newline), and ``fixtures/`` carry recorded provider payloads. The v1 invokers
here reproduce the corpus's ``_seams.py`` invocation exactly
(``get_llm_provider`` -> ``get_optional_params`` -> ``transform_request`` /
``transform_response`` / ``CustomStreamWrapper`` over the real decoders), so
each differential row proves snapshot == v1-at-HEAD == v2.
"""

import copy
import json
import pathlib
import time
from typing import Any, Dict, List

import httpx

import litellm
from litellm.litellm_core_utils.get_litellm_params import get_litellm_params
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager, get_optional_params

CORPUS_DIR = pathlib.Path(__file__).parent / "characterization_bedrock"
CASES_DIR = CORPUS_DIR / "cases"
FIXTURES_DIR = CORPUS_DIR / "fixtures"
SNAPSHOTS_DIR = CORPUS_DIR / "snapshots"

PROVIDERS: Dict[str, str] = {
    "bedrock_converse": "bedrock/anthropic.claude-sonnet-4-20250514-v1:0",
    "bedrock_invoke": "bedrock/invoke/anthropic.claude-3-7-sonnet-20250219-v1:0",
}

FROZEN_TIME = 1718064000.0  # matches the characterization corpus conftest


def load_json(path: pathlib.Path) -> Any:
    with open(path) as f:
        return json.load(f)


def cases() -> Dict[str, Dict[str, Any]]:
    return {path.stem: load_json(path) for path in sorted(CASES_DIR.glob("*.json"))}


def jsonable(obj: Any) -> Any:
    if hasattr(obj, "model_dump"):
        return jsonable(obj.model_dump())
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return repr(obj)


def canonical_json(obj: Any) -> str:
    return json.dumps(jsonable(obj), indent=2, sort_keys=True) + "\n"


def resolve_model(model_alias: str):
    model, custom_llm_provider, _, _ = litellm.get_llm_provider(model=model_alias)
    config = ProviderConfigManager.get_provider_chat_config(
        model=model, provider=LlmProviders(custom_llm_provider)
    )
    assert config is not None
    return model, custom_llm_provider, config


def resolve(provider_key: str):
    return resolve_model(PROVIDERS[provider_key])


def make_logging(model: str, messages: List[dict], stream: bool = False) -> Logging:
    return Logging(
        model=model,
        messages=messages,
        stream=stream,
        call_type="completion",
        start_time=time.time(),
        litellm_call_id="diff-litellm-call-id",
        function_id="diff-function-id",
    )


def run_v1_request_transform(provider_key: str, case: Dict[str, Any]) -> Dict[str, Any]:
    return run_v1_request_transform_for_model(PROVIDERS[provider_key], case)


def run_v1_request_transform_for_model(
    model_alias: str, case: Dict[str, Any], drop_params: bool = False
) -> Dict[str, Any]:
    model, custom_llm_provider, config = resolve_model(model_alias)
    messages = copy.deepcopy(case["messages"])
    optional_params = get_optional_params(
        model=model,
        custom_llm_provider=custom_llm_provider,
        messages=messages,
        drop_params=drop_params or None,
        **copy.deepcopy(case["params"]),
    )
    litellm_params = get_litellm_params(custom_llm_provider=custom_llm_provider)
    headers = config.validate_environment(
        headers={},
        model=model,
        messages=messages,
        optional_params=optional_params,
        litellm_params=litellm_params,
        api_key=None,
        api_base=None,
    )
    return config.transform_request(
        model=model,
        messages=messages,
        optional_params=optional_params,
        litellm_params=litellm_params,
        headers=headers,
    )


def run_v1_response_transform(
    provider_key: str,
    provider_response: Dict[str, Any],
    messages: List[dict],
) -> litellm.ModelResponse:
    model, _, config = resolve(provider_key)
    raw_response = httpx.Response(
        status_code=200,
        json=provider_response,
        request=httpx.Request("POST", "https://differential.invalid/model/invoke"),
    )
    return config.transform_response(
        model=model,
        raw_response=raw_response,
        model_response=litellm.ModelResponse(),
        logging_obj=make_logging(model, messages),
        request_data={},
        messages=messages,
        optional_params={},
        litellm_params={},
        encoding=litellm.encoding,
        api_key=None,
        json_mode=None,
    )


def _wrap_stream(model: str, completion_stream) -> List[dict]:
    wrapper = CustomStreamWrapper(
        completion_stream=iter(completion_stream),
        model=model,
        custom_llm_provider="bedrock",
        logging_obj=make_logging(
            model, [{"role": "user", "content": "stream"}], stream=True
        ),
    )
    return [chunk.model_dump() for chunk in wrapper]


def replay_v1_converse_events(events: List[dict]) -> List[dict]:
    """Parsed converse stream events -> the real AWSEventStreamDecoder parser.

    Pinned at the parsed-event seam: the binary AWS event-stream framing in
    front of ``converse_chunk_parser`` is botocore plumbing, not translation.
    """
    from litellm.llms.bedrock.chat.invoke_handler import AWSEventStreamDecoder

    model, _, _ = resolve("bedrock_converse")
    decoder = AWSEventStreamDecoder(model=model)
    chunks = (decoder.converse_chunk_parser(event) for event in events)
    return _wrap_stream(model, chunks)


def replay_v1_invoke_events(events: List[dict]) -> List[dict]:
    from litellm.llms.bedrock.chat.invoke_handler import (
        AmazonAnthropicClaudeStreamDecoder,
    )

    model, _, _ = resolve("bedrock_invoke")
    decoder = AmazonAnthropicClaudeStreamDecoder(model=model, sync_stream=True)
    chunks = (decoder._chunk_parser(event) for event in events)
    return _wrap_stream(model, chunks)
