import json
import os
from typing import Any, Dict, List, Optional, Tuple

import pytest

import litellm
from litellm.exceptions import BadRequestError

from .grid_spec import (
    OMIT,
    ROUTES,
    CellExpectation,
    ModelEntry,
    all_cells,
)


_PROMPT_MESSAGES: List[Dict[str, str]] = [
    {"role": "user", "content": "Step by step, calculate 47 * 53. Show your work."}
]


def _required_env_missing(model: ModelEntry) -> Optional[str]:
    missing = [key for key in model.required_env if not os.environ.get(key)]
    if missing:
        return "missing env: " + ", ".join(sorted(missing))
    return None


def _max_tokens_for(model: ModelEntry) -> int:
    return 200 if model.mode == "adaptive" else 8192


def _build_completion_kwargs(model: ModelEntry, effort: str) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "model": model.model,
        "messages": _PROMPT_MESSAGES,
        "max_tokens": _max_tokens_for(model),
    }
    kwargs.update(model.params())
    if effort != "__omit__":
        kwargs["reasoning_effort"] = effort
    if model.model.startswith("vertex_ai/"):
        kwargs["vertex_project"] = os.environ["VERTEX_PROJECT"]
    if model.model.startswith("azure_ai/"):
        kwargs["api_base"] = os.environ["AZURE_FOUNDRY_API_BASE"]
        kwargs["api_key"] = os.environ["AZURE_FOUNDRY_API_KEY"]
    return kwargs


def _converse_subbody(body: Dict[str, Any]) -> Dict[str, Any]:
    return body.get("additionalModelRequestFields", body)


def _max_tokens_from_body(body: Dict[str, Any], route_name: str) -> Optional[int]:
    if route_name == "bedrock_converse":
        return body.get("inferenceConfig", {}).get("maxTokens")
    return body.get("max_tokens")


def _assert_cell(
    route_name: str,
    body: Optional[Dict[str, Any]],
    status: int,
    cell: CellExpectation,
) -> None:
    assert status == cell.status, f"expected status={cell.status}, got status={status}"

    if cell.status != 200:
        return

    assert body is not None, "wire body was not captured for a 200-status cell"
    subbody = _converse_subbody(body) if route_name == "bedrock_converse" else body
    thinking = subbody.get("thinking")
    output_config = subbody.get("output_config")

    if cell.thinking_type is OMIT:
        assert thinking is None, f"expected thinking omitted, got {thinking!r}"
    else:
        assert thinking is not None, "expected thinking present, got omit"
        assert thinking.get("type") == cell.thinking_type, (
            f"expected thinking.type={cell.thinking_type!r}, "
            f"got {thinking.get('type')!r}"
        )

    if cell.output_config_effort is OMIT:
        assert (
            output_config is None or "effort" not in output_config
        ), f"expected output_config.effort omitted, got {output_config!r}"
    else:
        assert output_config is not None, (
            f"expected output_config.effort={cell.output_config_effort!r}, "
            "got output_config omitted"
        )
        assert output_config.get("effort") == cell.output_config_effort, (
            f"expected output_config.effort={cell.output_config_effort!r}, "
            f"got {output_config.get('effort')!r}"
        )

    if cell.thinking_budget_tokens is not OMIT:
        assert thinking is not None
        assert thinking.get("budget_tokens") == cell.thinking_budget_tokens, (
            f"expected thinking.budget_tokens={cell.thinking_budget_tokens!r}, "
            f"got {thinking.get('budget_tokens')!r}"
        )

    if cell.max_tokens is not OMIT:
        wire_max = _max_tokens_from_body(body, route_name)
        assert (
            wire_max == cell.max_tokens
        ), f"expected max_tokens={cell.max_tokens!r}, got {wire_max!r}"


_PARAMS: List[Tuple[str, ModelEntry, str, CellExpectation]] = all_cells()


def _cell_id(case: Tuple[str, ModelEntry, str, CellExpectation]) -> str:
    route_name, model, effort, _ = case
    effort_label = "__empty__" if effort == "" else effort
    return f"{route_name}-{model.alias}-{effort_label}"


_PARAM_IDS: List[str] = [_cell_id(case) for case in _PARAMS]


def _classify_status(exc: Exception) -> int:
    if isinstance(exc, BadRequestError):
        return 400
    code = getattr(exc, "status_code", None)
    if isinstance(code, int):
        return code
    return 500


def _model_unavailable(model: ModelEntry, exc: Optional[Exception]) -> bool:
    if not model.unavailable_error or exc is None:
        return False
    return model.unavailable_error in str(exc)


async def _call_chat(model: ModelEntry, effort: str) -> Tuple[int, Optional[Exception]]:
    kwargs = _build_completion_kwargs(model, effort)
    try:
        await litellm.acompletion(**kwargs)
        return 200, None
    except Exception as exc:
        return _classify_status(exc), exc


async def _call_messages(
    model: ModelEntry, effort: str
) -> Tuple[int, Optional[Exception]]:
    kwargs = _build_completion_kwargs(model, effort)
    try:
        await litellm.anthropic_messages(**kwargs)
        return 200, None
    except Exception as exc:
        return _classify_status(exc), exc


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("route_name", "model", "effort", "cell"), _PARAMS, ids=_PARAM_IDS
)
async def test_reasoning_effort_grid(
    route_name: str,
    model: ModelEntry,
    effort: str,
    cell: CellExpectation,
    wire_capture,
) -> None:
    skip_reason = _required_env_missing(model)
    if skip_reason:
        pytest.skip(skip_reason)

    if model.fail_reason:
        pytest.xfail(model.fail_reason)

    if route_name == "bedrock_invoke_messages":
        status, exc = await _call_messages(model, effort)
    else:
        status, exc = await _call_chat(model, effort)

    if _model_unavailable(model, exc):
        pytest.skip(f"{model.alias}: {model.unavailable_error}")

    record = wire_capture.latest()
    body = record["body"] if record else None
    if route_name == "bedrock_converse" and isinstance(body, str):
        body = json.loads(body)

    try:
        _assert_cell(route_name, body, status, cell)
    except AssertionError:
        if exc is not None:
            raise AssertionError(
                f"underlying exception ({type(exc).__name__}): {exc}"
            ) from None
        raise


def test_grid_cell_count() -> None:
    assert len(_PARAMS) == 25 * 11, (
        f"expected 275 cells (25 provider x model combos x 11 efforts), "
        f"got {len(_PARAMS)}"
    )


def test_grid_route_coverage() -> None:
    route_names = {route.name for route in ROUTES}
    assert route_names == {
        "anthropic_direct",
        "azure_ai",
        "vertex_ai",
        "bedrock_converse",
        "bedrock_invoke_chat",
        "bedrock_invoke_messages",
    }


def test_model_unavailable_tolerates_only_the_declared_error() -> None:
    gated = ModelEntry(
        alias="bedrock-claude-opus-4-7",
        model="bedrock/converse/us.anthropic.claude-opus-4-7",
        mode="adaptive",
        unavailable_error="is not available for this account",
    )
    entitlement_error = Exception(
        "litellm.APIConnectionError: BedrockException - "
        '{"message":"anthropic.claude-opus-4-7 is not available for this account."}'
    )

    assert _model_unavailable(gated, entitlement_error) is True
    assert (
        _model_unavailable(gated, Exception("ThrottlingException: rate exceeded"))
        is False
    )
    assert _model_unavailable(gated, None) is False

    ungated = ModelEntry(
        alias="bedrock-claude-opus-4-6",
        model="bedrock/converse/us.anthropic.claude-opus-4-6-v1",
        mode="adaptive",
    )
    assert _model_unavailable(ungated, entitlement_error) is False
