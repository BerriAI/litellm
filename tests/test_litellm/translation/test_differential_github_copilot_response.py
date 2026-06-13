"""Differential parity for the github_copilot response path (wave 3).

github_copilot is SDK-path: the LIVE v1 normalizer is
``convert_to_model_response_object`` (the config's ``transform_response`` and
its Anthropic-native synthesis are DEAD on this path). The per-provider delta
is the envelope PRESET — main.py sets ``model_response.model =
f"github_copilot/{model}"`` before conversion and cdr re-prefixes onto the
wire model -> ``github_copilot/{wire_model}`` (the compat_sdk seam arm,
construction arm "openai"). These rows pin that through the seam's
``_to_model_response_openai`` exactly like the compat_sdk family, plus the
wrong-construction-arm divergence the OWN_MODULE_RESPONSE_STYLES table must
pin (the cdr float-``created`` template).

OAuth safety: parsing is pure (no token files, no network); these rows touch
no provider resolution at all, so they cannot reach the device flow.
"""

import copy
import json

import pydantic
import pytest

from litellm.types.utils import ModelResponse
from litellm.utils import convert_to_model_response_object

from litellm.translation.engine.pipeline import OWN_MODULE_RESPONSE_STYLES
from litellm.translation.inbound.openai_chat import parse_request
from litellm.translation.inbound.openai_chat.response import serialize_response
from litellm.translation.providers.github_copilot import parse_response
from litellm.translation_seam import (
    UsageStyle,
    build_translation_deps,
    to_model_response,
)

from .test_differential_openai_response import _RESPONSES

PROVIDER = "github_copilot"
MODEL = "gpt-4o"
_REQUEST = {"model": MODEL, "messages": [{"role": "user", "content": "hi"}]}
_PRESET = f"{PROVIDER}/{MODEL}"

_RESPONSE_ROWS = (
    "text",
    "tool_calls_rewrites_stop",
    "cached_and_reasoning_usage_details",
)


def _v1_model_response(raw: dict, preset_model: str) -> dict:
    result = convert_to_model_response_object(
        response_object=copy.deepcopy(raw),
        model_response_object=ModelResponse(model=preset_model),
    )
    return result.model_dump()


def _v2_with_style(raw: dict, preset_model: str, style: UsageStyle) -> dict:
    parsed = parse_request(copy.deepcopy(_REQUEST))
    assert parsed.is_ok(), parsed.error.summary
    response = parse_response(copy.deepcopy(raw), parsed.ok)
    assert response.is_ok(), response.error.summary
    body = serialize_response(response.ok, build_translation_deps(), "openai")
    return to_model_response(
        body, ModelResponse(model=preset_model), usage_style=style
    ).model_dump()


def _v2_model_response(raw: dict, preset_model: str) -> dict:
    return _v2_with_style(raw, preset_model, "openai")


def _norm(payload: dict) -> str:
    return json.dumps(payload, sort_keys=True, default=str)


@pytest.mark.parametrize("name", sorted(_RESPONSE_ROWS))
def test_preset_reprefix_matches_v1(name: str, frozen_ambient) -> None:
    """SDK-path preset: github_copilot/{request model} in,
    github_copilot/{wire model} out, byte-identical dumps both sides."""
    raw = _RESPONSES[name]
    v1 = _v1_model_response(raw, _PRESET)
    v2 = _v2_model_response(raw, _PRESET)
    assert _norm(v2) == _norm(v1)
    assert v2["model"] == f"{PROVIDER}/{raw['model']}"


def test_preset_survives_when_wire_model_missing(frozen_ambient) -> None:
    """cdr's elif arm needs a non-None wire model; without one the preset
    github_copilot/{request model} survives verbatim on both sides."""
    raw = {k: v for k, v in _RESPONSES["text"].items() if k != "model"}
    v1 = _v1_model_response(raw, _PRESET)
    v2 = _v2_model_response(raw, _PRESET)
    assert _norm(v2) == _norm(v1)
    assert v2["model"] == _PRESET


def test_wrong_construction_arm_diverges_and_the_table_pins_it(
    frozen_ambient,
) -> None:
    """The OWN_MODULE_RESPONSE_STYLES value is enforced by this divergence
    (verifier-wave2b-final F1): a FLOAT wire ``created`` rides the normalized
    body; the correct cdr ("openai") arm coerces it and serves exactly like
    v1, while the wrong "openai_like" arm (ModelResponse(**json)) raises
    ValidationError on traffic v1 serves. If the arms stop diverging here the
    pin is dead — re-decide before relying on it."""
    assert OWN_MODULE_RESPONSE_STYLES[PROVIDER] == "openai"
    raw = copy.deepcopy(_RESPONSES["text"])
    raw["created"] = raw["created"] + 0.5
    v1 = _v1_model_response(raw, _PRESET)
    correct = _v2_with_style(raw, _PRESET, OWN_MODULE_RESPONSE_STYLES[PROVIDER])
    assert _norm(correct) == _norm(v1)
    assert correct["created"] == _RESPONSES["text"]["created"]
    with pytest.raises(pydantic.ValidationError):
        _v2_with_style(raw, _PRESET, "openai_like")
