"""Shared plumbing for the google differential gates (vertex gemini, AI
Studio gemini, vertex claude).

The reference corpus under ``characterization_google/`` is a verbatim copy of
the translation characterization corpus (mateo/translation-characterization-
providers branch). The v1 invokers reproduce that corpus's ``_seams.py``
exactly: the gemini body builder is ``sync_transform_request_body`` (v1's
``transform_request`` raises NotImplementedError; the wrapper is hermetic
below the 1024-token cache minimum with the vertex token fetch stubbed), and
the vertex claude body goes through ``VertexAIAnthropicConfig`` with the
partner route's ``anthropic_version``/``is_vertex_request`` injection plus
the beta-filtering step. Each differential row proves
snapshot == v1-at-HEAD == v2.
"""

import copy
import json
import pathlib
import time
from typing import Any, Dict, List, Tuple

import httpx

import litellm
from litellm.litellm_core_utils.get_litellm_params import get_litellm_params
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager, get_optional_params

CORPUS_DIR = pathlib.Path(__file__).parent / "characterization_google"
CASES_DIR = CORPUS_DIR / "cases"
FIXTURES_DIR = CORPUS_DIR / "fixtures"
SNAPSHOTS_DIR = CORPUS_DIR / "snapshots"

PROVIDERS: Dict[str, str] = {
    "vertex_gemini": "vertex_ai/gemini-2.5-pro",
    "gemini": "gemini/gemini-2.5-flash",
    "vertex_anthropic": "vertex_ai/claude-sonnet-4@20250514",
}

# differential provider key -> translation v2 provider key
V2_PROVIDERS: Dict[str, str] = {
    "vertex_gemini": "vertex_ai",
    "gemini": "gemini",
    "vertex_anthropic": "vertex_anthropic",
}

GEMINI_API_KEY = "char-gemini-test-key"
VERTEX_TOKEN = "char-vertex-token"
VERTEX_PROJECT = "char-test-project"
VERTEX_LOCATION_GEMINI = "us-central1"

FROZEN_TIME = 1718064000.0


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


def resolve_model(model_alias: str) -> Tuple[str, str, Any]:
    model, custom_llm_provider, _, _ = litellm.get_llm_provider(model=model_alias)
    config = ProviderConfigManager.get_provider_chat_config(
        model=model, provider=LlmProviders(custom_llm_provider)
    )
    assert config is not None
    return model, custom_llm_provider, config


def resolve(provider_key: str) -> Tuple[str, str, Any]:
    return resolve_model(PROVIDERS[provider_key])


def make_logging(model: str, messages: List[dict], stream: bool = False) -> Logging:
    logging_obj = Logging(
        model=model,
        messages=messages,
        stream=stream,
        call_type="completion",
        start_time=time.time(),
        litellm_call_id="diff-litellm-call-id",
        function_id="diff-function-id",
    )
    logging_obj.update_environment_variables(
        model=model, user=None, optional_params={}, litellm_params={}
    )
    return logging_obj


def _gemini_request_body(
    model: str,
    custom_llm_provider: str,
    messages: List[dict],
    optional_params: Dict[str, Any],
    litellm_params: Dict[str, Any],
) -> Dict[str, Any]:
    from litellm.llms.vertex_ai.gemini.transformation import (
        sync_transform_request_body,
    )
    from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (
        VertexGeminiConfig,
        VertexLLM,
    )

    is_vertex = custom_llm_provider != "gemini"
    gemini_api_key = None if is_vertex else GEMINI_API_KEY
    vertex_project = VERTEX_PROJECT if is_vertex else None
    vertex_location = VERTEX_LOCATION_GEMINI if is_vertex else None

    vertex_llm = VertexLLM()
    _auth_header, project = vertex_llm._ensure_access_token(
        credentials=None,
        project_id=vertex_project,
        custom_llm_provider=custom_llm_provider,  # type: ignore[arg-type]
    )
    auth_header, _url = vertex_llm._get_token_and_url(
        model=model,
        gemini_api_key=gemini_api_key,
        auth_header=_auth_header,
        vertex_project=project or None,
        vertex_location=vertex_location,
        vertex_credentials=None,
        stream=None,
        custom_llm_provider=custom_llm_provider,  # type: ignore[arg-type]
        api_base=None,
        should_use_v1beta1_features=False,
    )
    VertexGeminiConfig().validate_environment(
        api_key=auth_header,
        headers=None,
        model=model,
        messages=messages,
        optional_params=optional_params,
        litellm_params=litellm_params,
    )
    return dict(
        sync_transform_request_body(
            gemini_api_key=gemini_api_key,
            messages=messages,
            api_base=None,
            model=model,
            client=None,
            timeout=None,
            extra_headers=None,
            optional_params=optional_params,
            logging_obj=make_logging(model, messages),
            custom_llm_provider=custom_llm_provider,  # type: ignore[arg-type]
            litellm_params=litellm_params,
            vertex_project=project or None,
            vertex_location=vertex_location,
            vertex_auth_header=auth_header,
        )
    )


def _vertex_anthropic_request_body(
    model: str,
    config: Any,
    messages: List[dict],
    optional_params: Dict[str, Any],
    litellm_params: Dict[str, Any],
) -> Dict[str, Any]:
    from litellm.anthropic_beta_headers_manager import (
        update_request_with_filtered_beta,
    )
    from litellm.llms.anthropic.chat.transformation import AnthropicConfig

    optional_params.update(
        {"anthropic_version": "vertex-2023-10-16", "is_vertex_request": True}
    )
    optional_params.pop("stream", None)
    optional_params.pop("json_mode", None)
    is_vertex_request = optional_params.pop("is_vertex_request", False)
    merged_params = {**optional_params, "is_vertex_request": is_vertex_request}
    headers = AnthropicConfig().validate_environment(
        api_key=VERTEX_TOKEN,
        headers={"Authorization": f"Bearer {VERTEX_TOKEN}"},
        model=model,
        messages=messages,
        optional_params=merged_params,
        litellm_params=litellm_params,
    )
    data = config.transform_request(
        model=model,
        messages=messages,
        optional_params=merged_params,
        litellm_params=litellm_params,
        headers=headers,
    )
    _headers, data = update_request_with_filtered_beta(
        headers=headers, request_data=data, provider="vertex_ai"
    )
    return data


def run_v1_request_transform(
    provider_key: str, case: Dict[str, Any], drop_params: bool = False
) -> Dict[str, Any]:
    return run_v1_request_transform_for_model(
        PROVIDERS[provider_key], case, drop_params=drop_params
    )


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
    if custom_llm_provider in ("vertex_ai", "gemini") and "claude" not in model:
        return _gemini_request_body(
            model, custom_llm_provider, messages, optional_params, litellm_params
        )
    return _vertex_anthropic_request_body(
        model, config, messages, optional_params, litellm_params
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
        request=httpx.Request("POST", "https://differential.invalid/generateContent"),
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


def _wrap_stream(
    model: str, custom_llm_provider: str, completion_stream: Any
) -> List[dict]:
    wrapper = CustomStreamWrapper(
        completion_stream=iter(completion_stream),
        model=model,
        custom_llm_provider=custom_llm_provider,
        logging_obj=make_logging(
            model, [{"role": "user", "content": "stream"}], stream=True
        ),
    )
    return [chunk.model_dump() for chunk in wrapper]


def replay_v1_gemini_sse(provider_key: str, sse_lines: List[str]) -> List[dict]:
    """Raw ``alt=sse`` lines through the REAL vertex ``ModelResponseIterator``
    inside ``CustomStreamWrapper`` (which v1 tags ``vertex_ai_beta`` for both
    google routes)."""
    from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (
        ModelResponseIterator,
    )

    model, _, _ = resolve(provider_key)
    iterator = ModelResponseIterator(
        streaming_response=iter(sse_lines),
        sync_stream=True,
        logging_obj=make_logging(
            model, [{"role": "user", "content": "stream"}], stream=True
        ),
    )
    return _wrap_stream(model, "vertex_ai_beta", iterator)


def replay_v1_vertex_anthropic_sse(sse_lines: List[str]) -> List[dict]:
    from litellm.llms.anthropic.chat.handler import ModelResponseIterator

    model, _, _ = resolve("vertex_anthropic")
    iterator = ModelResponseIterator(
        streaming_response=iter(sse_lines), sync_stream=True
    )
    return _wrap_stream(model, "anthropic", iterator)


def sse_events(sse_lines: List[str]) -> List[dict]:
    """The parsed-event seam for gemini streams: strip the SSE framing that
    is transport plumbing in front of ``chunk_parser``."""
    return [
        json.loads(line[len("data: ") :])
        for line in sse_lines
        if line.startswith("data: ")
    ]
