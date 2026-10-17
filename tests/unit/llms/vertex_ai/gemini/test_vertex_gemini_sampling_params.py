"""Test vertex_ai / gemini supports_sampling_params flag behavior (Issue #47015)."""

import copy
from typing import Generator

import pytest
import litellm
from litellm import Router
from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import VertexGeminiConfig
from litellm.llms.vertex_ai.gemini.transformation import _transform_request_body

MODEL = "gemini-3.8-flash"


@pytest.fixture(autouse=True)
def clean_model_cost() -> Generator[None, None, None]:
    """Isolate litellm.model_cost mutations so they don't leak to subsequent tests."""
    key = f"vertex_ai/{MODEL}"
    orig_entry = copy.deepcopy(litellm.model_cost.get(key))
    had_key = key in litellm.model_cost
    yield
    if had_key and orig_entry is not None:
        litellm.model_cost[key] = orig_entry
    else:
        litellm.model_cost.pop(key, None)


def _router() -> None:
    Router(
        model_list=[
            {
                "model_name": MODEL,
                "litellm_params": {
                    "model": f"vertex_ai/{MODEL}",
                    "vertex_project": "p",
                    "vertex_location": "global",
                },
                "model_info": {"supports_sampling_params": False},
            }
        ]
    )


def test_router_keeps_supports_sampling_params_on_shared_key() -> None:
    _router()
    assert litellm.model_cost[f"vertex_ai/{MODEL}"].get("supports_sampling_params") is False


def test_gemini_drops_sampling_params_when_flag_is_false() -> None:
    litellm.model_cost[f"vertex_ai/{MODEL}"] = {
        **litellm.model_cost.get(f"vertex_ai/{MODEL}", {}),
        "supports_sampling_params": False,
    }
    out = VertexGeminiConfig().map_openai_params(
        non_default_params={"temperature": 0.3, "top_p": 0.9, "top_k": 20},
        optional_params={},
        model=MODEL,
        drop_params=True,
    )
    assert "temperature" not in out
    assert "top_p" not in out
    assert "top_k" not in out


def test_gemini_raises_unsupported_params_when_drop_params_is_false() -> None:
    litellm.model_cost[f"vertex_ai/{MODEL}"] = {
        **litellm.model_cost.get(f"vertex_ai/{MODEL}", {}),
        "supports_sampling_params": False,
    }
    with pytest.raises(litellm.utils.UnsupportedParamsError):
        VertexGeminiConfig().map_openai_params(
            non_default_params={"temperature": 0.3},
            optional_params={},
            model=MODEL,
            drop_params=False,
        )


def test_gemini_skips_deprecation_warning_when_unsupported() -> None:
    litellm.model_cost[f"vertex_ai/{MODEL}"] = {
        **litellm.model_cost.get(f"vertex_ai/{MODEL}", {}),
        "supports_sampling_params": False,
    }
    out = VertexGeminiConfig().map_openai_params(
        non_default_params={"temperature": 0.5},
        optional_params={},
        model=MODEL,
        drop_params=True,
    )
    assert "temperature" not in out


def test_gemini_request_body_drops_top_k_when_drop_params_true() -> None:
    litellm.model_cost[f"vertex_ai/{MODEL}"] = {
        **litellm.model_cost.get(f"vertex_ai/{MODEL}", {}),
        "supports_sampling_params": False,
    }
    body = _transform_request_body(
        messages=[{"role": "user", "content": "hi"}],
        model=MODEL,
        optional_params={"top_k": 20},
        custom_llm_provider="vertex_ai",
        litellm_params={"drop_params": True},
        cached_content=None,
    )
    assert "top_k" not in (body.get("generationConfig") or {})


def test_gemini_request_body_raises_when_drop_params_false() -> None:
    litellm.model_cost[f"vertex_ai/{MODEL}"] = {
        **litellm.model_cost.get(f"vertex_ai/{MODEL}", {}),
        "supports_sampling_params": False,
    }
    with pytest.raises(litellm.utils.UnsupportedParamsError):
        _transform_request_body(
            messages=[{"role": "user", "content": "hi"}],
            model=MODEL,
            optional_params={"top_k": 20},
            custom_llm_provider="vertex_ai",
            litellm_params={"drop_params": False},
            cached_content=None,
        )
