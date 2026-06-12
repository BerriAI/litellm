"""Shared plumbing for the xai (Grok) differential gates.

The reference corpus under ``characterization_xai/`` is GENERATED, not
vendored: the characterization branch
(mateo/translation-characterization-providers) carries zero xai fixtures, so
every snapshot here pins v1 IN-PROCESS AT HEAD (the primary reference per
the differential rule) invoked exactly the way the xai httpx handler runs
(provenance documented in characterization_xai/README.md; regenerate with
``python -m tests.test_litellm.translation.generate_xai_snapshots``).

The v1 invokers mirror main.py's dedicated xai elif (main.py:2289 ->
``base_llm_http_handler.completion``):

- requests: ``get_optional_params(custom_llm_provider="xai")`` (the
  RAISE-unless-drop_params gate over XAIChatConfig's supported list) with
  completion()'s ``stream=None`` default, then the handler's ``extra_body``
  pop (hh:398-399; the injected ``{}`` merges nothing onto the wire), then
  ``XAIChatConfig.transform_request``.
- responses: ``XAIChatConfig.transform_response`` over an ``httpx.Response``
  — LIVE on the httpx path (the inverse of the openai SDK route), including
  the websearch/fold/normalize usage post-steps.
- streams: SSE ``data:`` lines through ``XAIChatCompletionStreamingHandler``
  + ``CustomStreamWrapper(custom_llm_provider="xai")`` (the line seam the
  dossier prescribes: the chunk_parser rewrites are xai BEHAVIOR and sit
  below the parsed-chunk seam).
"""

import copy
import json
import pathlib
import time
from typing import Any, Dict, List, Optional

import httpx

import litellm
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
from litellm.llms.xai.chat.transformation import (
    XAIChatCompletionStreamingHandler,
    XAIChatConfig,
)
from litellm.types.utils import ModelResponse
from litellm.utils import get_optional_params

CORPUS_DIR = pathlib.Path(__file__).parent / "characterization_xai"
CASES_DIR = CORPUS_DIR / "cases"
FIXTURES_DIR = CORPUS_DIR / "fixtures"
SNAPSHOTS_DIR = CORPUS_DIR / "snapshots"

STREAM_MODEL = "grok-3-mini"

FROZEN_TIME = 1718064000.0  # matches the translation conftest frozen_ambient


def load_json(path: pathlib.Path) -> Any:
    with open(path) as f:
        return json.load(f)


def corpus(kind: str) -> Dict[str, Any]:
    directory = CASES_DIR if kind == "cases" else FIXTURES_DIR / kind
    return {path.stem: load_json(path) for path in sorted(directory.glob("*.json"))}


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


def run_v1_request_transform(case: Dict[str, Any]) -> Dict[str, Any]:
    """May RAISE UnsupportedParamsError: that IS the pinned v1 behavior for
    the R2 gate rows (the differential asserts the raise, never a remap)."""
    request = copy.deepcopy(case)
    model = request.pop("model")
    messages = request.pop("messages")
    optional_params = get_optional_params(
        model=model,
        custom_llm_provider="xai",
        messages=copy.deepcopy(messages),
        stream=request.pop("stream", None),
        **request,
    )
    optional_params.pop("extra_body", None)
    return XAIChatConfig().transform_request(
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
        litellm_call_id="diff-xai-call-id",
        function_id="diff-xai-function-id",
    )
    logging_obj.update_environment_variables(
        model=model, user=None, optional_params={}, litellm_params={}
    )
    return logging_obj


def run_v1_response_transform(
    provider_response: Dict[str, Any], model: str
) -> ModelResponse:
    messages = [{"role": "user", "content": "hi"}]
    raw_response = httpx.Response(
        status_code=200,
        json=copy.deepcopy(provider_response),
        request=httpx.Request("POST", "https://api.x.ai/v1/chat/completions"),
    )
    return XAIChatConfig().transform_response(
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


def replay_xai_sse_lines(
    events: List[dict], stream_options: Optional[dict] = None
) -> List[dict]:
    lines = [f"data: {json.dumps(event)}" for event in copy.deepcopy(events)]
    lines.append("data: [DONE]")
    handler = XAIChatCompletionStreamingHandler(
        streaming_response=iter(lines), sync_stream=True
    )
    wrapper = CustomStreamWrapper(
        completion_stream=handler,
        model=STREAM_MODEL,
        custom_llm_provider="xai",
        logging_obj=make_logging(
            STREAM_MODEL, [{"role": "user", "content": "stream"}], stream=True
        ),
        stream_options=stream_options,
    )
    return [chunk.model_dump() for chunk in wrapper]


__all__ = (
    "CASES_DIR",
    "CORPUS_DIR",
    "FIXTURES_DIR",
    "FROZEN_TIME",
    "SNAPSHOTS_DIR",
    "STREAM_MODEL",
    "canonical_json",
    "corpus",
    "jsonable",
    "load_json",
    "make_logging",
    "replay_xai_sse_lines",
    "run_v1_request_transform",
    "run_v1_response_transform",
)
