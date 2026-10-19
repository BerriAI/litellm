"""Configuration and tool-schema helpers for the in-process Tool Loop harness."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

from pydantic import BaseModel, ConfigDict, TypeAdapter, create_model

from litellm.harness.context import SessionContext
from litellm.harness.options import ToolLoopOptions
from litellm.harness.types import Capabilities, Harness
from litellm.llms.base_llm.harness.transformation import BaseHarnessConfig
from litellm.llms.base_llm.harness.utils import gateway_headers
from litellm.types.utils import ChatCompletionToolParam

TOOL_LOOP_MAX_MODEL_CALLS: Final = 100
_ANNOTATIONS_ADAPTER: Final = TypeAdapter(Mapping[str, object])
_OBJECT_ADAPTER: Final = TypeAdapter(object)
_MODEL_FACTORY: Final[Callable[..., type[BaseModel]]] = create_model


@dataclass(frozen=True, slots=True)
class FunctionTool:
    name: str
    fn: Callable[..., object]
    args_model: type[BaseModel]
    spec: ChatCompletionToolParam


def _field_definition(
    parameter: inspect.Parameter,
    annotations: Mapping[str, object],
) -> tuple[object, object]:
    annotation: Final = annotations.get(parameter.name, object)
    raw_default: Final[object] = parameter.default  # pyright: ignore[reportAny]  # inspect exposes defaults as Any
    if raw_default is inspect.Parameter.empty:
        return annotation, ...
    default: Final = _OBJECT_ADAPTER.validate_python(raw_default)
    return annotation, default


def function_tool(fn: Callable[..., object]) -> FunctionTool:
    signature: Final = inspect.signature(fn)
    parameters: Final = tuple(signature.parameters.values())
    raw_annotations: Final[object] = inspect.get_annotations(fn, eval_str=True)
    annotations: Final = _ANNOTATIONS_ADAPTER.validate_python(raw_annotations)
    if any(
        parameter.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD) for parameter in parameters
    ):
        raise ValueError(f"Tool {fn.__name__} cannot use variadic parameters")

    fields: Final = MappingProxyType(
        {parameter.name: _field_definition(parameter, annotations) for parameter in parameters}
    )
    args_model: Final[type[BaseModel]] = _MODEL_FACTORY(
        f"{fn.__name__}_args",
        __config__=ConfigDict(extra="forbid"),
        **fields,  # pyright: ignore[reportCallIssue, reportArgumentType]  # Pydantic creates fields dynamically
    )
    spec: Final[ChatCompletionToolParam] = {
        "type": "function",
        "function": {
            "name": fn.__name__,
            "description": inspect.getdoc(fn) or "",
            "parameters": args_model.model_json_schema(),
        },
    }
    return FunctionTool(name=fn.__name__, fn=fn, args_model=args_model, spec=spec)


def _routing_kwargs(ctx: SessionContext) -> Mapping[str, object]:
    if not ctx.model:
        raise ValueError("Harness.TOOL_LOOP needs model=")
    if ctx.gateway is not None:
        return {
            "model": f"litellm_proxy/{ctx.model}",
            "api_base": ctx.gateway.api_base,
            "api_key": ctx.gateway.api_key,
            "extra_headers": gateway_headers(ctx),
        }
    return {
        "model": ctx.model,
        **({"api_key": ctx.api_key} if ctx.api_key is not None else {}),
        **({"api_base": ctx.api_base} if ctx.api_base is not None else {}),
    }


def completion_kwargs(ctx: SessionContext) -> Mapping[str, object]:
    options: Final = ToolLoopHarnessConfig().get_options(ctx)
    routing: Final = _routing_kwargs(ctx)
    kwargs: Final[Mapping[str, object]] = MappingProxyType({**options.completion_kwargs, **routing})
    if ctx.output is None:
        return kwargs
    return {**kwargs, "response_format": ctx.output}


class ToolLoopHarnessConfig(BaseHarnessConfig[ToolLoopOptions]):
    harness = Harness.TOOL_LOOP
    options_type = ToolLoopOptions
    uses_model_endpoint = False
    capabilities = Capabilities(
        structured_output=True,
        tool_approval=True,
        tool_filtering=False,
        history=True,
        custom_tools=True,
        skills=False,
        resume=False,
        permission_modes=frozenset({"ask", "full"}),
    )

    def validate_environment(self, ctx: SessionContext) -> None:
        self.get_options(ctx)
        if not ctx.model:
            raise ValueError("Harness.TOOL_LOOP needs model=")
