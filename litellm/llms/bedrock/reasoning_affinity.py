from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Final, cast

from litellm.llms.bedrock.common_utils import get_bedrock_openai_model
from litellm.responses.utils import ResponsesAPIRequestUtils
from litellm.types.router import Deployment


def strip_incompatible_bedrock_openai_reasoning(
    request_input: object,
    healthy_deployments: Sequence[Mapping[str, object]],
    get_routed_model_ids: Callable[[], frozenset[str]],
    get_deployment: Callable[[str], Deployment | None],
) -> None:
    if not isinstance(request_input, list):
        return
    target_models: Final = frozenset(
        get_bedrock_openai_model(cast(Mapping[str, object], params).get("model"))
        if isinstance(params, Mapping)
        else None
        for deployment in healthy_deployments
        for params in (deployment.get("litellm_params"),)
    )
    if len(target_models) != 1 or None in target_models:
        return
    target_model: Final = next(iter(target_models))
    items: Final = cast(list[object], request_input)  # cast-ok: narrowed by isinstance
    item_model_ids: Final = frozenset(
        model_id
        for item in items
        if (model_id := ResponsesAPIRequestUtils.get_encrypted_content_model_id(item)) is not None
    )
    routed_model_ids: Final = get_routed_model_ids()
    foreign_model_ids: Final = frozenset(
        model_id
        for model_id in item_model_ids
        if model_id not in routed_model_ids
        if (origin := get_deployment(model_id)) is None
        or get_bedrock_openai_model(origin.litellm_params.model) != target_model
    )
    if foreign_model_ids:
        ResponsesAPIRequestUtils.strip_encrypted_reasoning_from_input(items, originating_model_ids=foreign_model_ids)
