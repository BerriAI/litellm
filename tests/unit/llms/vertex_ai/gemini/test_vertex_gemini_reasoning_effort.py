from typing import Final

import pytest

import litellm
from litellm.exceptions import UnsupportedParamsError
from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import VertexGeminiConfig


def test_vertex_gemini_unsupported_reasoning_effort_with_drop_params_true() -> None:
    """Verifies that unsupported reasoning_effort (e.g., 'xhigh') is safely dropped

    when drop_params=True is passed directly or inside optional_params.
    """
    config: Final = VertexGeminiConfig()

    # 1. Directly passing drop_params=True (string format)
    result_direct: Final = config.map_openai_params(
        non_default_params={"reasoning_effort": "xhigh"},
        optional_params={},
        model="gemini-2.5-flash",
        drop_params=True,
    )
    assert "thinkingConfig" not in result_direct
    assert "reasoning_effort" not in result_direct

    # 2. Directly passing drop_params=True (dict format from Agents SDK)
    result_dict: Final = config.map_openai_params(
        non_default_params={"reasoning_effort": {"effort": "xhigh", "summary": "auto"}},
        optional_params={},
        model="gemini-2.5-flash",
        drop_params=True,
    )
    assert "thinkingConfig" not in result_dict

    # 3. Passing drop_params=True in optional_params
    result_optional: Final = config.map_openai_params(
        non_default_params={"reasoning_effort": "xhigh"},
        optional_params={"drop_params": True},
        model="gemini-3-pro-preview",
        drop_params=False,
    )
    assert "thinkingConfig" not in result_optional


def test_vertex_gemini_unsupported_reasoning_effort_with_global_drop_params(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifies that unsupported reasoning_effort (e.g., 'max') is safely dropped

    when litellm.drop_params = True is globally active.
    """
    monkeypatch.setattr(litellm, "drop_params", True)
    config: Final = VertexGeminiConfig()

    # Test with Gemini 2.5 (budget-based thinking)
    result_gemini_25: Final = config.map_openai_params(
        non_default_params={"reasoning_effort": "max"},
        optional_params={},
        model="gemini-2.5-flash",
        drop_params=False,
    )
    assert "thinkingConfig" not in result_gemini_25

    # Test with Gemini 3+ (level-based thinking)
    result_gemini_3: Final = config.map_openai_params(
        non_default_params={"reasoning_effort": "max"},
        optional_params={},
        model="gemini-3-pro-preview",
        drop_params=False,
    )
    assert "thinkingConfig" not in result_gemini_3


def test_vertex_gemini_unsupported_reasoning_effort_raises_without_drop_params() -> None:
    """Verifies that unsupported reasoning_effort values raise UnsupportedParamsError

    when drop_params is False.
    """
    config: Final = VertexGeminiConfig()

    for effort in ("xhigh", "max", "unsupported_level"):
        with pytest.raises(UnsupportedParamsError):
            config.map_openai_params(
                non_default_params={"reasoning_effort": effort},
                optional_params={},
                model="gemini-2.5-flash",
                drop_params=False,
            )

        with pytest.raises(UnsupportedParamsError):
            config.map_openai_params(
                non_default_params={"reasoning_effort": effort},
                optional_params={},
                model="gemini-3-pro-preview",
                drop_params=False,
            )


def test_vertex_gemini_direct_reasoning_effort_mappers_drop_params() -> None:
    """Verifies direct mapper methods honor drop_params=True and return None."""
    # Test _map_reasoning_effort_to_thinking_budget directly
    budget_result: Final = VertexGeminiConfig._map_reasoning_effort_to_thinking_budget(
        reasoning_effort="xhigh",
        model="gemini-2.5-flash",
        drop_params=True,
    )
    assert budget_result is None

    # Test _map_reasoning_effort_to_thinking_level directly
    level_result: Final = VertexGeminiConfig._map_reasoning_effort_to_thinking_level(
        reasoning_effort="xhigh",
        model="gemini-3-pro-preview",
        drop_params=True,
    )
    assert level_result is None

    # Test _map_reasoning_effort directly
    effort_result: Final = VertexGeminiConfig._map_reasoning_effort(
        effort="max",
        model="gemini-2.5-flash",
        reasoning_effort="max",
        drop_params=True,
    )
    assert effort_result is None

    effort_level_result: Final = VertexGeminiConfig._map_reasoning_effort(
        effort="max",
        model="gemini-3-pro-preview",
        reasoning_effort="max",
        drop_params=True,
    )
    assert effort_level_result is None


@pytest.mark.parametrize("effort", ["low", "medium", "high"])
@pytest.mark.parametrize("model", ["gemini-2.5-flash", "gemini-3-pro-preview"])
def test_vertex_gemini_valid_reasoning_effort_mapping(effort: str, model: str) -> None:
    """Verifies that supported reasoning_effort values continue to map as expected."""
    config: Final = VertexGeminiConfig()
    result: Final = config.map_openai_params(
        non_default_params={"reasoning_effort": effort},
        optional_params={},
        model=model,
        drop_params=False,
    )
    assert "thinkingConfig" in result
    thinking_config: Final = result["thinkingConfig"]
    assert thinking_config is not None
    if "gemini-3" in model:
        assert thinking_config.get("thinkingLevel") == effort
    else:
        assert "thinkingBudget" in thinking_config
