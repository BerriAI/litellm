from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from pydantic import BaseModel, ConfigDict, TypeAdapter

from litellm.types.proxy.model_metadata import GatewayModelMetadata

_METADATA_ADAPTER: Final = TypeAdapter(GatewayModelMetadata)
_EMPTY_DEFAULTS: Final[Mapping[str, int]] = MappingProxyType({})
_OUTPUT_BUDGET_PARAMS: Final = frozenset(("max_tokens", "max_completion_tokens", "max_output_tokens"))


class _Reasoning(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)
    effort: str | int | None = None


class _RequestEffort(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)
    reasoning_effort: str | int | None = None
    reasoning: _Reasoning | None = None

    def effort(self) -> str | int | None:
        return (
            self.reasoning_effort
            if self.reasoning_effort is not None
            else self.reasoning.effort
            if self.reasoning is not None
            else None
        )


def model_request_defaults(
    model_info: object,
    deployment_params: Mapping[str, object],
    request_params: Mapping[str, object],
    call_type: str,
) -> Mapping[str, int]:
    if call_type not in ("completion", "acompletion", "responses", "aresponses", "anthropic_messages"):
        return _EMPTY_DEFAULTS
    if any(request_params.get(key) is not None for key in _OUTPUT_BUDGET_PARAMS):
        return _EMPTY_DEFAULTS
    metadata: Final = _METADATA_ADAPTER.validate_python(model_info)
    defaults: Final = metadata.request_defaults
    if defaults is None:
        return _EMPTY_DEFAULTS
    requested_effort: Final = _RequestEffort.model_validate(request_params).effort()
    deployment_effort: Final = _RequestEffort.model_validate(deployment_params).effort()
    effort: Final = (
        requested_effort if requested_effort is not None else deployment_effort or metadata.default_reasoning_effort
    )
    budget: Final = defaults.output_budget(str(effort) if effort is not None else None)
    if budget is None:
        return _EMPTY_DEFAULTS
    key: Final = "max_output_tokens" if call_type in ("responses", "aresponses") else "max_tokens"
    return MappingProxyType({key: budget})


def deployment_params_with_request_defaults(
    deployment_params: Mapping[str, object],
    defaults: Mapping[str, int],
    model_info: object = None,
    request_params: Mapping[str, object] | None = None,
) -> Mapping[str, object]:
    caller_budget: Final = request_params is not None and any(
        request_params.get(key) is not None for key in _OUTPUT_BUDGET_PARAMS
    )
    configured_policy: Final = (
        model_info is not None and _METADATA_ADAPTER.validate_python(model_info).request_defaults is not None
    )
    replace_budget: Final = bool(defaults) or caller_budget and configured_policy
    return MappingProxyType(
        {
            key: value
            for key, value in deployment_params.items()
            if not replace_budget or key not in _OUTPUT_BUDGET_PARAMS
        }
    )
