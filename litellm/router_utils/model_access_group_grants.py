"""
Under general_settings.model_access_group_deny_by_default, auth stamps the caller's explicit model grants on its
auth object, and the router drops every candidate deployment in a model access group those grants do not name.
Filtering the deployments the router is about to pick, rather than predicting them at auth time, covers aliases,
routing groups, deployment ids, default fallbacks and models loaded after auth alike
"""

from collections.abc import Iterable, Mapping, Sequence
from typing import Final, TypeVar

import httpx
from pydantic import ConfigDict

import litellm
from litellm.router_utils.common_utils import request_user_api_key_auth
from litellm.types.llms.base_model import LiteLLMBaseModel
from litellm.types.proxy.auth.user_api_key_auth import UserAPIKeyAuth

_BLANKET_GRANTS: Final = frozenset({"*", "all-proxy-models", "all-team-models"})

_Deployment: Final = TypeVar("_Deployment", bound=Mapping[str, object])


class _DeploymentModelInfo(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    access_groups: tuple[str, ...] | None = None


class _DeploymentRow(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    model_name: str | None = None
    model_info: _DeploymentModelInfo | None = None


def explicit_model_grants(models: Iterable[str]) -> frozenset[str]:
    """The names in `models` that grant one thing by name: blanket entries and wildcard patterns name nothing"""
    return frozenset(model for model in models if model not in _BLANKET_GRANTS and "*" not in model)


def request_model_access_group_grants(request_kwargs: Mapping[str, object] | None) -> frozenset[str] | None:
    if request_kwargs is None:
        return None
    user_api_key_auth: Final = request_user_api_key_auth(request_kwargs)
    return user_api_key_auth.model_access_group_grants if isinstance(user_api_key_auth, UserAPIKeyAuth) else None


def _is_granted(deployment: Mapping[str, object], requested_model: str, grants: frozenset[str]) -> bool:
    row: Final = _DeploymentRow.model_validate(deployment)
    groups: Final = (row.model_info.access_groups if row.model_info is not None else None) or ()
    return not groups or requested_model in grants or row.model_name in grants or not grants.isdisjoint(groups)


def _denied(model: str) -> litellm.PermissionDeniedError:
    from litellm.proxy.auth.model_access_denied import model_access_denied_client_message

    return litellm.PermissionDeniedError(
        message=model_access_denied_client_message(model=model),
        llm_provider="",
        model=model,
        response=httpx.Response(status_code=403, request=httpx.Request(method="POST", url="https://litellm.ai")),
    )


def require_granted_deployment(
    deployment: _Deployment, requested_model: str, request_kwargs: Mapping[str, object] | None
) -> _Deployment:
    grants: Final = request_model_access_group_grants(request_kwargs)
    if grants is None or _is_granted(deployment, requested_model, grants):
        return deployment
    raise _denied(requested_model)


def drop_ungranted_deployments(
    deployments: Sequence[_Deployment], requested_model: str, request_kwargs: Mapping[str, object] | None
) -> tuple[_Deployment, ...]:
    grants: Final = request_model_access_group_grants(request_kwargs)
    if grants is None:
        return tuple(deployments)
    granted: Final = tuple(deployment for deployment in deployments if _is_granted(deployment, requested_model, grants))
    if deployments and not granted:
        raise _denied(requested_model)
    return granted
