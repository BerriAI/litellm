"""
Mid-stream fallback continuation: keep the fallback on a deployment that can
continue a prefilled assistant message. When a request carries
``MID_STREAM_CONTINUATION_KWARG``, deployments whose model does not support
assistant prefill are dropped, so the partial text is continued rather than
regenerated or rejected. Requests without the marker pass through untouched.
"""

from collections.abc import Mapping, Sequence
from typing import Final

from pydantic import TypeAdapter, ValidationError

from litellm.constants import MID_STREAM_CONTINUATION_KWARG
from litellm.integrations.custom_logger import CustomLogger, Span
from litellm.types.llms.openai import AllMessageValues
from litellm.utils import supports_assistant_prefill

_STR_KEYED_DICT_ADAPTER: Final = TypeAdapter(dict[str, object])


def _deployment_supports_prefill(deployment: object) -> bool:
    try:
        deployment_map: Final = _STR_KEYED_DICT_ADAPTER.validate_python(deployment)
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
        if not (request_kwargs or {}).get(MID_STREAM_CONTINUATION_KWARG):
            return healthy_deployments
        eligible: Final = (deployment for deployment in healthy_deployments if _deployment_supports_prefill(deployment))
        return list(eligible)  # mutable-ok: downstream deployment selection consumes a mutable list
