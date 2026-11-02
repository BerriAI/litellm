"""The caller facts the Rust core routes on, read from the request's auth metadata: whether the
caller is a proxy admin, the models its key and team allow (which narrow an access-group-only
caller to its groups' deployments) and, for a proxy admin, the model the request names."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

from pydantic import TypeAdapter

from litellm.litellm_core_utils.core_helpers import as_str_mapping
from litellm.litellm_core_utils.routing_common_utils import is_proxy_admin_request, request_user_api_key_auth

_NAMES: Final = TypeAdapter(tuple[object, ...])


def _names(value: object) -> tuple[str, ...]:
    names: Final = _NAMES.validate_python(value) if isinstance(value, list | tuple | set | frozenset) else ()
    return tuple(str(name) for name in names)


def _admin_model(request: Mapping[str, object]) -> object:
    """The model ``filter_team_based_models`` compares team deployments against."""
    metadata: Final = as_str_mapping(request.get("metadata")) or {}
    litellm_metadata: Final = as_str_mapping(request.get("litellm_metadata")) or {}
    return request.get("model") or metadata.get("model_group") or litellm_metadata.get("model_group")


def caller_fields(request: Mapping[str, object]) -> Mapping[str, object]:
    """``proxy_admin``, ``allowed_models`` and ``admin_model`` for the Rust request document."""
    auth: Final[object] = request_user_api_key_auth(request)
    if auth is None:
        return {}
    allowed: Final = sorted({*_names(getattr(auth, "models", None)), *_names(getattr(auth, "team_models", None))})
    admin: Final = is_proxy_admin_request(request)
    model: Final = _admin_model(request) if admin else None
    return {
        **({"allowed_models": allowed} if allowed else {}),
        **({"proxy_admin": True} if admin else {}),
        **({"admin_model": model} if isinstance(model, str) else {}),
    }


def _router_weights_error(raw: object) -> BaseException:
    """The error ``validate_router_weights`` raises for weights the Rust core refused."""
    from pydantic import ValidationError

    from litellm.types.router_weights import validate_router_weights

    try:
        _ = validate_router_weights(raw)
    except ValidationError as error:
        return error
    return RuntimeError(f"router-core refused router weights the validator accepts: {raw!r}")


def selection_refusal(step: Mapping[str, object], model: str, request: Mapping[str, object]) -> BaseException | None:
    """The error for a pick the Rust core refused over who the caller is or what it asked for:
    a public name several teams share, deployments reserved for other teams, invalid request
    router weights or a non-numeric deployment weight. None for any other failure."""
    import litellm.exceptions as litellm_exceptions

    kind: Final = step.get("error_kind")
    named: Final = str(step.get("error_model") or model)
    if kind == "multiple_teams":
        return litellm_exceptions.BadRequestError(
            message=(
                f"Model name '{named}' matches deployments from multiple teams. "
                "Specify the deployment ID directly to disambiguate."
            ),
            model=named,
            llm_provider="",
        )
    if kind == "reserved":
        return litellm_exceptions.BadRequestError(
            message=f"Deployment {named} is reserved for another team until {step.get('error_detail')}",
            model=named,
            llm_provider="",
        )
    if kind == "invalid_router_weights":
        return _router_weights_error(request.get("_router_weights"))
    if kind == "non_numeric_weight":
        return TypeError(f"Deployment {step.get('error_detail')} must be numeric")
    return None
