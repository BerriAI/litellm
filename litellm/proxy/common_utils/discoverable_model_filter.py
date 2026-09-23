"""Operator-declared discoverability shared by the model listing endpoints.

A `model_list` entry marked `model_info: {discoverable: false}` is left out of
`/v1/models`, `/v1/model/info` and `/model_group/info` for every caller without
the admin view, while a request that names the model directly still routes to
it. Filtering is presentation-only and fails open: a name the router cannot
resolve, a deployment without the flag, or a missing router hides nothing.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Final

from litellm.proxy._types import UserAPIKeyAuth, user_api_key_has_admin_view

if TYPE_CHECKING:
    from litellm.router import Router


def is_undiscoverable_deployment(deployment: Mapping[str, object]) -> bool:
    model_info: Final = deployment.get("model_info")
    if not isinstance(model_info, Mapping):
        return False
    return "discoverable" in model_info and model_info["discoverable"] is False


def is_undiscoverable_model_name(model_name: str, llm_router: Router | None) -> bool:
    if llm_router is None:
        return False
    deployments: Final = llm_router.get_model_list(model_name=model_name)
    if not deployments:
        return False
    return all(is_undiscoverable_deployment(deployment) for deployment in deployments)


def undiscoverable_model_names(
    model_names: Iterable[str],
    llm_router: Router | None,
    user_api_key_dict: UserAPIKeyAuth,
) -> frozenset[str]:
    if llm_router is None or user_api_key_has_admin_view(user_api_key_dict):
        return frozenset()
    if not any(is_undiscoverable_deployment(deployment) for deployment in llm_router.get_model_list() or ()):
        return frozenset()
    return frozenset(name for name in model_names if is_undiscoverable_model_name(name, llm_router))


def discoverable_rows(
    rows: Iterable[Mapping[str, object]],
    user_api_key_dict: UserAPIKeyAuth,
) -> tuple[Mapping[str, object], ...]:
    if user_api_key_has_admin_view(user_api_key_dict):
        return tuple(rows)
    return tuple(row for row in rows if not is_undiscoverable_deployment(row))
