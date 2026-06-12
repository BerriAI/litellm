"""Shared plumbing for the azure differential gates.

The reference corpus under ``characterization_azure/`` is a verbatim copy of
the translation characterization corpus (branch
mateo/translation-characterization-providers @ d8375ccb38,
tests/translation_characterization) for the azure chat route: ``cases/`` are
OpenAI-format requests, ``snapshots/`` pin the REAL v1 transform output
(canonical JSON: pretty-printed, sorted keys, trailing newline), ``fixtures/``
carry recorded provider payloads (incl. the azure-only empty-choices
``prompt_filter_results`` stream chunk). The v1 invokers reproduce the
corpus's ``_seams.py`` invocation exactly: ``get_optional_params`` (azure
branch: api-version-aware ``AzureOpenAIConfig.map_openai_params``) ->
``transform_request`` for requests, ``convert_to_model_response_object``
(azure.py's exact args; ``transform_response`` raises NotImplementedError)
for responses, and SDK ``ChatCompletionChunk`` replay through
``CustomStreamWrapper(custom_llm_provider="azure")`` for streams.

The corpus request seam calls ``get_optional_params`` with its ``stream``
parameter DEFAULT (False), so every request snapshot carries ``stream: false``
plus the always-injected ``extra_body: {}``; at runtime ``completion()``
passes ``stream=None`` (no key) and the SDK consumes ``extra_body`` as a
kwarg (an empty one merges nothing onto the wire). ``v2_comparable`` strips
exactly those two corpus-seam artifacts before the byte comparison; the v1
drift guard compares the UNSTRIPPED output, keeping the corpus honest.
"""

import copy
import json
import pathlib
import time
from typing import Any, Dict, List

from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
from litellm.llms.azure.chat.gpt_transformation import AzureOpenAIConfig
from litellm.types.utils import ModelResponse
from litellm.utils import convert_to_model_response_object, get_optional_params

CORPUS_DIR = pathlib.Path(__file__).parent / "characterization_azure"
CASES_DIR = CORPUS_DIR / "cases"
FIXTURES_DIR = CORPUS_DIR / "fixtures"
SNAPSHOTS_DIR = CORPUS_DIR / "snapshots"

MODEL = "gpt-4.1"  # the corpus deployment ("azure/gpt-4.1" pre-strip)

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


def stream_snapshot_chunks(path: pathlib.Path) -> List[Dict[str, Any]]:
    """Load a vendored stream snapshot, dropping the one SDK-version artifact:
    the corpus box's openai SDK dumped a ``moderation: null`` top-level key on
    every ChatCompletionChunk that the locked 2.33.0 SDK does not have, and
    ``preserve_upstream_non_openai_attributes`` copies whatever the installed
    SDK dumps. Everything else compares byte-for-byte."""
    chunks = load_json(path)
    for chunk in chunks:
        assert chunk.get("moderation") is None, (
            "vendored stream snapshot carries a NON-NULL moderation payload; "
            "the SDK-artifact strip only covers the null key"
        )
    return [
        {key: value for key, value in chunk.items() if key != "moderation"}
        for chunk in chunks
    ]


def v2_comparable(snapshot_body: Dict[str, Any]) -> Dict[str, Any]:
    """Strip the two corpus-seam artifacts that never reach the wire (module
    docstring); everything else must match v2 byte-for-byte."""
    assert snapshot_body.get("extra_body") in (None, {}), (
        "snapshot carries a NON-EMPTY extra_body; the strip only covers the "
        "always-empty corpus artifact"
    )
    out = {k: v for k, v in snapshot_body.items() if k != "extra_body"}
    if out.get("stream") is False:
        out = {k: v for k, v in out.items() if k != "stream"}
    return out


def run_v1_request_transform(case: Dict[str, Any]) -> Dict[str, Any]:
    messages = copy.deepcopy(case["messages"])
    optional_params = get_optional_params(
        model=MODEL,
        custom_llm_provider="azure",
        messages=messages,
        **copy.deepcopy(case["params"]),
    )
    return AzureOpenAIConfig().transform_request(
        model=MODEL,
        messages=messages,
        optional_params=optional_params,
        litellm_params={},
        headers={},
    )


def run_v1_response_transform(provider_response: Dict[str, Any]) -> ModelResponse:
    result = convert_to_model_response_object(
        response_object=copy.deepcopy(provider_response),
        model_response_object=ModelResponse(),
        convert_tool_call_to_json_mode=None,
        _response_headers={},
    )
    assert isinstance(result, ModelResponse)
    return result


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


def replay_azure_sdk_chunks(
    events: List[dict], custom_llm_provider: str = "azure"
) -> List[dict]:
    from openai.types.chat.chat_completion_chunk import ChatCompletionChunk

    sdk_chunks = (
        ChatCompletionChunk.model_validate(event) for event in copy.deepcopy(events)
    )
    wrapper = CustomStreamWrapper(
        completion_stream=sdk_chunks,
        model=MODEL,
        custom_llm_provider=custom_llm_provider,
        logging_obj=make_logging(
            MODEL, [{"role": "user", "content": "stream"}], stream=True
        ),
    )
    return [chunk.model_dump() for chunk in wrapper]


__all__ = (
    "CASES_DIR",
    "CORPUS_DIR",
    "FIXTURES_DIR",
    "FROZEN_TIME",
    "MODEL",
    "SNAPSHOTS_DIR",
    "canonical_json",
    "cases",
    "jsonable",
    "load_json",
    "make_logging",
    "replay_azure_sdk_chunks",
    "stream_snapshot_chunks",
    "run_v1_request_transform",
    "run_v1_response_transform",
    "v2_comparable",
)
