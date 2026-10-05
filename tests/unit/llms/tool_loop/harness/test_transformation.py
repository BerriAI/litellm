"""Tests for Tool Loop schemas and completion routing."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Final, Literal

import pytest
from pydantic import BaseModel

from litellm import sandbox
from litellm.harness.context import GatewayTarget, SessionContext
from litellm.harness.options import ToolLoopOptions
from litellm.harness.types import Harness
from litellm.llms.tool_loop.harness.transformation import (
    ToolLoopHarnessConfig,
    completion_kwargs,
    function_tool,
)


def make_context(
    tmp_path: Path,
    *,
    model: str | None = "anthropic/claude",
    gateway: GatewayTarget | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
    options: ToolLoopOptions | None = None,
    output: type[BaseModel] | None = None,
) -> SessionContext:
    return SessionContext(
        harness=Harness.TOOL_LOOP,
        sandbox=sandbox.local(tmp_path),
        session_id="tool-loop-transform",
        model=model,
        gateway=gateway,
        api_key=api_key,
        api_base=api_base,
        options=options,
        output=output,
    )


def search(
    query: str,
    limit: int = 5,
    state: Literal["open", "closed"] = "open",
) -> str:
    """Search records."""
    return query


def test_function_tool_schema_has_required_defaulted_and_literal_fields() -> None:
    specification: Final = function_tool(search).spec
    schema: Final = specification["function"]["parameters"]

    assert schema["required"] == ["query"]
    assert schema["properties"]["query"] == {"title": "Query", "type": "string"}
    assert schema["properties"]["limit"] == {"default": 5, "title": "Limit", "type": "integer"}
    assert schema["properties"]["state"] == {
        "default": "open",
        "enum": ["open", "closed"],
        "title": "State",
        "type": "string",
    }
    assert specification["function"]["description"] == "Search records."


def test_function_schema_rejects_unknown_arguments() -> None:
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        function_tool(search).args_model.model_validate({"query": "owner", "unknown": "value"})


def variadic_positional(*args: int) -> int:
    return len(args)


def variadic_keyword(**kwargs: int) -> int:
    return len(kwargs)


@pytest.mark.parametrize("fn", [variadic_positional, variadic_keyword])
def test_variadic_tools_are_rejected(fn: Callable[..., object]) -> None:
    with pytest.raises(ValueError, match="variadic parameters"):
        function_tool(fn)


def test_sdk_routing_overrides_completion_kwargs(tmp_path: Path) -> None:
    ctx: Final = make_context(
        tmp_path,
        api_key="provided-key",
        api_base="https://provider",
        options=ToolLoopOptions(
            completion_kwargs={
                "model": "wrong-model",
                "api_key": "wrong-key",
                "api_base": "https://wrong",
                "temperature": 0.2,
            }
        ),
    )

    kwargs: Final = completion_kwargs(ctx)

    assert kwargs == {
        "model": "anthropic/claude",
        "api_key": "provided-key",
        "api_base": "https://provider",
        "temperature": 0.2,
    }


def test_gateway_routing_and_response_format_override_options(tmp_path: Path) -> None:
    class OutputModel(BaseModel):
        pass

    gateway: Final = GatewayTarget(api_base="https://gateway", api_key="virtual-key")
    ctx: Final = make_context(
        tmp_path,
        gateway=gateway,
        options=ToolLoopOptions(
            completion_kwargs={
                "model": "wrong-model",
                "api_key": "wrong-key",
                "api_base": "https://wrong",
                "response_format": "wrong-format",
            }
        ),
        output=OutputModel,
    )

    kwargs: Final = completion_kwargs(ctx)

    assert kwargs == {
        "model": "litellm_proxy/anthropic/claude",
        "api_base": "https://gateway",
        "api_key": "virtual-key",
        "extra_headers": {"x-litellm-tags": "harness,tool_loop"},
        "response_format": OutputModel,
    }


def test_configuration_requires_model_and_declares_capabilities(tmp_path: Path) -> None:
    config: Final = ToolLoopHarnessConfig()
    assert config.uses_model_endpoint is False
    assert config.capabilities.structured_output
    assert config.capabilities.tool_approval
    assert config.capabilities.history
    assert config.capabilities.custom_tools
    assert config.capabilities.permission_modes == frozenset({"ask", "full"})
    with pytest.raises(ValueError, match=r"Harness\.TOOL_LOOP needs model="):
        config.validate_environment(make_context(tmp_path, model=None))
