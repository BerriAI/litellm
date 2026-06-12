"""Differential parity: v2 azure translation vs the v1 AzureOpenAIConfig chain.

Two gates:

1. In-process corpus: the v1 body is produced exactly as ``completion()``
   would for provider "azure" (``AzureOpenAIConfig().map_openai_params`` with
   the resolved api_version on the detection model, then
   ``transform_request`` on the deployment) and compared, as normalized JSON,
   to ``translate_chat_request(..., "azure")``.
2. Characterization corpus (``characterization_azure/``, vendored verbatim
   from mateo/translation-characterization-providers): v1-at-HEAD must still
   equal the committed snapshot (drift guard) AND v2 must equal the
   snapshot's wire-comparable form byte-for-byte (the snapshot-only
   ``extra_body: {}`` / default ``stream: false`` corpus-seam artifacts are
   stripped; see ``_azure_corpus``).

Azure-only typed fallbacks pinned below: cache_control anywhere (azure does
NOT strip it), explicit ``stream: false``, the api-version gates on
tool_choice/response_format, gpt-3.5/gpt-35 response_format (json-tool
strategy), and o-series/gpt-5 detection on ``base_model or model``.
"""

import copy
import json

import pytest

from litellm.llms.azure.chat.gpt_transformation import AzureOpenAIConfig
from litellm.translation import translate_chat_request

from . import _azure_corpus as corpus
from .conftest import build_real_deps

MODEL = "gpt-4.1"
DEFAULT_API_VERSION = "2025-02-01-preview"  # litellm.AZURE_DEFAULT_API_VERSION

_WEATHER_TOOL = {
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

_STRICT_TOOL = {
    "type": "function",
    "function": {
        "name": "report",
        "parameters": {
            "type": "object",
            "properties": {"body": {"type": "string"}},
            "required": ["body"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}


def _assistant_tool_call(call_id, city, name="get_weather"):
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps({"city": city})},
            }
        ],
    }


# name -> (request, api_version, base_model)
CORPUS = {
    "text": ({"model": MODEL, "messages": [{"role": "user", "content": "Hello"}]},),
    "system_and_sampling": (
        {
            "model": MODEL,
            "max_tokens": 50,
            "temperature": 0.5,
            "top_p": 0.9,
            "messages": [
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "Hi"},
            ],
        },
    ),
    "multiturn_stop_list_stream": (
        {
            "model": MODEL,
            "max_tokens": 64,
            "stop": ["END", "STOP"],
            "stream": True,
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"},
                {"role": "user", "content": "How are you?"},
            ],
        },
    ),
    "max_completion_tokens": (
        {
            "model": MODEL,
            "max_completion_tokens": 128,
            "messages": [{"role": "user", "content": "hi"}],
        },
    ),
    "temperature_int_stays_int": (
        {
            "model": MODEL,
            "temperature": 1,
            "messages": [{"role": "user", "content": "hi"}],
        },
    ),
    "tools_auto": (
        {
            "model": MODEL,
            "tools": [_WEATHER_TOOL],
            "tool_choice": "auto",
            "messages": [{"role": "user", "content": "Weather in Paris?"}],
        },
    ),
    "tools_strict": (
        {
            "model": MODEL,
            "tools": [_STRICT_TOOL],
            "messages": [{"role": "user", "content": "report this"}],
        },
    ),
    "tool_choice_required_current_api": (
        {
            "model": MODEL,
            "tools": [_WEATHER_TOOL],
            "tool_choice": "required",
            "messages": [{"role": "user", "content": "Weather in Paris?"}],
        },
    ),
    "tool_choice_specific": (
        {
            "model": MODEL,
            "tools": [_WEATHER_TOOL],
            "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
            "messages": [{"role": "user", "content": "Weather in Paris?"}],
        },
    ),
    "tool_choice_unparseable_api_version_passthrough": (
        # v1 splits the api_version on "-" and passes tool_choice through when
        # there are fewer than three segments; deps.api_version=None mirrors it
        {
            "model": MODEL,
            "tools": [_WEATHER_TOOL],
            "tool_choice": "required",
            "messages": [{"role": "user", "content": "Weather in Paris?"}],
        },
        "",
        None,
    ),
    "parallel_tool_calls_false": (
        {
            "model": MODEL,
            "tools": [_WEATHER_TOOL],
            "parallel_tool_calls": False,
            "messages": [{"role": "user", "content": "Weather in Paris and Rome?"}],
        },
    ),
    "tool_call_roundtrip": (
        {
            "model": MODEL,
            "tools": [_WEATHER_TOOL],
            "messages": [
                {"role": "user", "content": "Weather in Paris?"},
                _assistant_tool_call("call_1", "Paris"),
                {"role": "tool", "tool_call_id": "call_1", "content": "Sunny, 20C"},
            ],
        },
    ),
    "image_url_string_to_object": (
        # convert_to_azure_openai_messages' _azure_image_url_helper and the
        # openai transform produce the same object form
        {
            "model": MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "what is this"},
                        {"type": "image_url", "image_url": "https://e.test/a.png"},
                    ],
                }
            ],
        },
    ),
    "image_base64": (
        {
            "model": MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "and this"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="},
                        },
                    ],
                }
            ],
        },
    ),
    "response_format_json_object": (
        {
            "model": MODEL,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "user", "content": "json please"}],
        },
    ),
    "response_format_json_schema_strict": (
        {
            "model": MODEL,
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
            "messages": [{"role": "user", "content": "capital of France?"}],
        },
    ),
    "response_format_on_gpt4_deployment": (
        # azure's response_format model gate excludes only gpt-3.5/gpt-35
        # (unlike openai's exact-name gpt-4 exclusion); a deployment named
        # gpt-4.1 passes through both v1 and v2
        {
            "model": "gpt-4-1-suffix",
            "response_format": {"type": "json_object"},
            "messages": [{"role": "user", "content": "json"}],
        },
    ),
    "gpt5_chat_is_plain_azure": (
        {
            "model": "gpt-5-chat",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "hi"}],
        },
    ),
    "deployment_with_base_model": (
        # detection (param mapping) runs on base_model, the body keeps the
        # deployment name, exactly like utils.py:4831-4881 + azure.py:245-247
        {
            "model": "prod-chat-deployment",
            "max_tokens": 32,
            "messages": [{"role": "user", "content": "hi"}],
        },
        DEFAULT_API_VERSION,
        "gpt-4.1",
    ),
}

# Typed fallbacks: each row must return a TranslationError whose summary
# carries the reason fragment, so the seam serves the request through v1.
# name -> (request, api_version, base_model, reason fragment)
EXPECTED_FALLBACKS = {
    "o_series_substring_deployment": (
        {"model": "prod-o3-mini", "messages": [{"role": "user", "content": "hi"}]},
        DEFAULT_API_VERSION,
        None,
        "AzureOpenAIO1Config",
    ),
    "o_series_via_base_model": (
        {"model": "my-deployment", "messages": [{"role": "user", "content": "hi"}]},
        DEFAULT_API_VERSION,
        "o1-mini",
        "AzureOpenAIO1Config",
    ),
    "gpt5_model": (
        {"model": "gpt-5", "messages": [{"role": "user", "content": "hi"}]},
        DEFAULT_API_VERSION,
        None,
        "AzureOpenAIGPT5Config",
    ),
    "gpt5_series_prefix": (
        {
            "model": "gpt5_series/my-deploy",
            "messages": [{"role": "user", "content": "hi"}],
        },
        DEFAULT_API_VERSION,
        None,
        "AzureOpenAIGPT5Config",
    ),
    "cache_control_in_messages": (
        {
            "model": MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "cached",
                            "cache_control": {"type": "ephemeral"},
                        },
                        {"type": "text", "text": "question"},
                    ],
                }
            ],
        },
        DEFAULT_API_VERSION,
        None,
        "cache_control inside messages",
    ),
    "cache_control_in_tools": (
        {
            "model": MODEL,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "parameters": {"type": "object", "properties": {}},
                        "cache_control": {"type": "ephemeral"},
                    },
                }
            ],
            "messages": [{"role": "user", "content": "x"}],
        },
        DEFAULT_API_VERSION,
        None,
        "cache_control inside tools",
    ),
    "explicit_stream_false": (
        {
            "model": MODEL,
            "stream": False,
            "messages": [{"role": "user", "content": "x"}],
        },
        DEFAULT_API_VERSION,
        None,
        "explicit stream: false",
    ),
    "tool_choice_pre_2023_12_api": (
        {
            "model": MODEL,
            "tools": [_WEATHER_TOOL],
            "tool_choice": "auto",
            "messages": [{"role": "user", "content": "x"}],
        },
        "2023-07-01-preview",
        None,
        "tool_choice needs api_version",
    ),
    "tool_choice_required_2024_05_api": (
        {
            "model": MODEL,
            "tools": [_WEATHER_TOOL],
            "tool_choice": "required",
            "messages": [{"role": "user", "content": "x"}],
        },
        "2024-05-01-preview",
        None,
        "tool_choice='required' is unsupported",
    ),
    "response_format_gpt35": (
        {
            "model": "gpt-35-turbo",
            "response_format": {"type": "json_object"},
            "messages": [{"role": "user", "content": "x"}],
        },
        DEFAULT_API_VERSION,
        None,
        "json-tool strategy",
    ),
    "response_format_gpt_3_5_normalized": (
        {
            "model": "gpt-3-5-turbo",
            "response_format": {"type": "json_object"},
            "messages": [{"role": "user", "content": "x"}],
        },
        DEFAULT_API_VERSION,
        None,
        "json-tool strategy",
    ),
    "response_format_pre_2024_08_api": (
        {
            "model": MODEL,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "user", "content": "x"}],
        },
        "2024-05-01-preview",
        None,
        "response_format needs api_version",
    ),
    "user_param": (
        {"model": MODEL, "user": "u-1", "messages": [{"role": "user", "content": "x"}]},
        DEFAULT_API_VERSION,
        None,
        "user param",
    ),
    "reasoning_effort_plain_azure": (
        {
            "model": MODEL,
            "reasoning_effort": "high",
            "messages": [{"role": "user", "content": "x"}],
        },
        DEFAULT_API_VERSION,
        None,
        "reasoning_effort",
    ),
    "shared_guard_string_stop": (
        {"model": MODEL, "stop": "END", "messages": [{"role": "user", "content": "x"}]},
        DEFAULT_API_VERSION,
        None,
        "string-form stop",
    ),
}


def _row(name: str):
    entry = CORPUS[name]
    request = entry[0]
    api_version = entry[1] if len(entry) > 1 else DEFAULT_API_VERSION
    base_model = entry[2] if len(entry) > 2 else None
    return request, api_version, base_model


def _v1_body(case: dict, api_version: str, base_model) -> dict:
    request = copy.deepcopy(case)
    config = AzureOpenAIConfig()
    model = request["model"]
    detection_model = base_model or model
    params = {
        key: value for key, value in request.items() if key not in ("model", "messages")
    }
    optional = config.map_openai_params(
        copy.deepcopy(params),
        {},
        detection_model,
        drop_params=False,
        api_version=api_version,
    )
    return config.transform_request(
        model, copy.deepcopy(request["messages"]), optional, {}, {}
    )


def _v2_body(case: dict, api_version, base_model):
    return translate_chat_request(
        copy.deepcopy(case),
        "azure",
        build_real_deps(api_version=api_version or None, base_model=base_model),
    )


def _norm(body: dict) -> str:
    return json.dumps(body, sort_keys=True, default=str)


@pytest.mark.parametrize("name", sorted(CORPUS))
def test_v2_request_matches_v1(name: str) -> None:
    request, api_version, base_model = _row(name)
    result = _v2_body(request, api_version, base_model)
    assert result.is_ok(), result.error.summary
    assert _norm(result.ok) == _norm(_v1_body(request, api_version, base_model))


@pytest.mark.parametrize("name", sorted(EXPECTED_FALLBACKS))
def test_unsupported_shape_is_a_typed_fallback(name: str) -> None:
    case, api_version, base_model, reason_fragment = EXPECTED_FALLBACKS[name]
    result = _v2_body(case, api_version, base_model)
    assert result.is_error(), f"{name} unexpectedly translated: {result.ok!r}"
    assert reason_fragment in result.error.summary, result.error.summary


# ---------------------------------------------------------------------------
# Second gate: the vendored characterization corpus.
# ---------------------------------------------------------------------------

CHAR_CASES = corpus.cases()

# Shapes v2 deliberately routes back to v1, with the asserted reason.
CHAR_EXPECTED_FALLBACKS = {
    # the shared guard's list-form system check fires before the azure
    # cache_control check; either way the request stays on v1 untouched
    "cache_control_messages": "list-form system content",
    "cache_control_tools": "cache_control inside tools",
    "params_sampling": "user param",
    "pdf_base64": "messages",  # the inbound schema has no file part at all
}


@pytest.fixture(autouse=True)
def _no_ambient_azure_api_version(monkeypatch):
    """The corpus seam resolved api_version through the env-default chain
    with AZURE_API_VERSION unset (-> AZURE_DEFAULT_API_VERSION); keep the
    drift guard hermetic on boxes that set it."""
    monkeypatch.delenv("AZURE_API_VERSION", raising=False)
    monkeypatch.setattr("litellm.api_version", None, raising=False)


@pytest.mark.parametrize("case_id", sorted(CHAR_CASES))
def test_v1_still_matches_snapshot(case_id: str) -> None:
    case = CHAR_CASES[case_id]
    if "azure" in case["skip"]:
        pytest.skip(case["skip"]["azure"])
    snapshot = corpus.SNAPSHOTS_DIR / "requests" / "azure" / f"{case_id}.json"
    body = corpus.run_v1_request_transform(case)
    assert corpus.canonical_json(body) == snapshot.read_text(), (
        f"v1 drifted from the characterization snapshot for {case_id}; "
        "regenerate the corpus and ship the diff as its own PR"
    )


@pytest.mark.parametrize("case_id", sorted(CHAR_CASES))
def test_v2_matches_snapshot_or_falls_back(case_id: str) -> None:
    case = CHAR_CASES[case_id]
    if "azure" in case["skip"]:
        pytest.skip(case["skip"]["azure"])
    raw = {
        "model": corpus.MODEL,
        "messages": copy.deepcopy(case["messages"]),
        **copy.deepcopy(case["params"]),
    }
    result = translate_chat_request(
        raw, "azure", build_real_deps(api_version=DEFAULT_API_VERSION)
    )
    if case_id in CHAR_EXPECTED_FALLBACKS:
        assert result.is_error(), f"{case_id} unexpectedly translated"
        assert CHAR_EXPECTED_FALLBACKS[case_id] in result.error.summary
        return
    assert result.is_ok(), result.error.summary
    snapshot = corpus.SNAPSHOTS_DIR / "requests" / "azure" / f"{case_id}.json"
    expected = corpus.v2_comparable(corpus.load_json(snapshot))
    assert corpus.canonical_json(result.ok) == corpus.canonical_json(expected)
