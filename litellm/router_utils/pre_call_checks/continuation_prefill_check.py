"""
Mid-stream fallback continuation: keep the fallback on a deployment that can
actually continue a prefilled assistant message.

When a chat-completions stream breaks after content and the Router re-enters the
fallback chain to continue it (the request is marked with
``MID_STREAM_CONTINUATION_KWARG``), only a deployment whose model supports
assistant prefill can pick up the partial text without regenerating it.
Deployments that cannot are dropped, so selection lands on a
continuation-capable one. If a group has none it empties and the fallback chain
moves on, surfacing the original error rather than sending a request the target
would reject or duplicate. A request without the marker is passed through
untouched.
"""

from typing import Final

from pydantic import TypeAdapter, ValidationError

from litellm.integrations.custom_logger import CustomLogger, Span
from litellm.types.llms.openai import AllMessageValues
from litellm.utils import supports_assistant_prefill

MID_STREAM_CONTINUATION_KWARG: Final = "_mid_stream_continuation"

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
        healthy_deployments: list[dict[str, object]],
        messages: list[AllMessageValues] | None,
        request_kwargs: dict[str, object] | None = None,
        parent_otel_span: Span | None = None,
    ) -> list[dict[str, object]]:
        if not (request_kwargs or {}).get(MID_STREAM_CONTINUATION_KWARG):
            return healthy_deployments
        return [deployment for deployment in healthy_deployments if _deployment_supports_prefill(deployment)]
