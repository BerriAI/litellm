"""Keeps a mid-stream continuation on deployments whose model supports assistant prefill."""

from collections.abc import Mapping, Sequence
from typing import Final

from pydantic import TypeAdapter, ValidationError

from litellm.integrations.custom_logger import CustomLogger, Span
from litellm.types.llms.openai import AllMessageValues
from litellm.utils import supports_assistant_prefill

MID_STREAM_CONTINUATION_KWARG: Final = "_mid_stream_continuation"


class _ContinuationMarker:
    """Only the router constructs one, so a JSON request body cannot forge the continuation flag."""


MID_STREAM_CONTINUATION_MARKER: Final = _ContinuationMarker()

_STR_KEYED_DICT_ADAPTER: Final = TypeAdapter(dict[str, object])


def _deployment_supports_prefill(deployment: object) -> bool:
    try:
        deployment_map: Final = _STR_KEYED_DICT_ADAPTER.validate_python(deployment)
    except ValidationError:
        return False
    try:
        model_info: Final = _STR_KEYED_DICT_ADAPTER.validate_python(deployment_map.get("model_info"))
        declared: Final = model_info.get("supports_assistant_prefill")
        if isinstance(declared, bool):
            return declared
    except ValidationError:
        pass
    try:
        litellm_params: Final = _STR_KEYED_DICT_ADAPTER.validate_python(deployment_map.get("litellm_params"))
    except ValidationError:
        return False
    model: Final = litellm_params.get("model")
    return isinstance(model, str) and bool(model) and supports_assistant_prefill(model=model)


class ContinuationPrefillDeploymentCheck(CustomLogger):
    async def async_filter_deployments(
        self,
        model: str,
        healthy_deployments: list[dict[str, object]],  # mutable-ok: CustomLogger deployment-list contract
        messages: Sequence[AllMessageValues] | None,
        request_kwargs: Mapping[str, object] | None = None,
        parent_otel_span: Span | None = None,
    ) -> list[dict[str, object]]:  # mutable-ok: returns a mutable deployment list
        marker: Final = (request_kwargs or {}).get(MID_STREAM_CONTINUATION_KWARG)
        if not isinstance(marker, _ContinuationMarker):
            return healthy_deployments
        eligible: Final = (deployment for deployment in healthy_deployments if _deployment_supports_prefill(deployment))
        return list(eligible)
