"""
Mid-stream fallback continuation: keep the fallback on a deployment that can
continue a prefilled assistant message. When the router marks a fallback
re-entry as a continuation, deployments whose model does not support assistant
prefill are dropped, so the partial text is continued rather than regenerated or
rejected. Requests without the marker pass through untouched.
"""

from collections.abc import Mapping, Sequence
from typing import Final

from pydantic import TypeAdapter, ValidationError

from litellm.integrations.custom_logger import CustomLogger, Span
from litellm.types.llms.openai import AllMessageValues
from litellm.utils import supports_assistant_prefill

# The router marks a continuation re-entry by placing MID_STREAM_CONTINUATION_MARKER
# under this key. Since the proxy can forward arbitrary request-body fields into the
# router, the marker is a private object checked by type rather than a truthy value:
# a JSON request body cannot construct one, so a client cannot forge the flag to steer
# deployment selection toward prefill-capable deployments.
MID_STREAM_CONTINUATION_KWARG: Final = "_mid_stream_continuation"


class _ContinuationMarker:
    """Unforgeable sentinel; only the router can produce an instance."""


MID_STREAM_CONTINUATION_MARKER: Final = _ContinuationMarker()

_STR_KEYED_DICT_ADAPTER: Final = TypeAdapter(dict[str, object])


def _deployment_supports_prefill(deployment: object) -> bool:
    try:
        deployment_map: Final = _STR_KEYED_DICT_ADAPTER.validate_python(deployment)
    except ValidationError:
        return False
    # A per-deployment model_info override wins, so a model that is not in the cost
    # map (or is registered generically) can still opt in or out explicitly with
    # `model_info: {"supports_assistant_prefill": true|false}`.
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
        return list(eligible)  # mutable-ok: downstream deployment selection consumes a mutable list
