from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

from litellm.rust_bridge.host.routes.tokenize import body_params


@dataclass(frozen=True, slots=True)
class TokenizeAdmissionContext:
    authorized_fields: frozenset[str] = frozenset()


@dataclass(frozen=True, slots=True)
class TokenizeRoutingContext:
    trusted_fields: frozenset[str]
    deployment_params: Mapping[str, object]


def validate_internal_fields(request: Mapping[str, object]) -> None:
    if any(name.startswith("_tokenize_") and name != "_tokenize_context" for name in request):
        raise ValueError("tokenization internal fields are reserved")
    context: Final = request.get("_tokenize_context")
    if context is not None and not isinstance(context, (TokenizeAdmissionContext, TokenizeRoutingContext)):
        raise ValueError("invalid tokenization context")


def source_for(request: Mapping[str, object], name: str) -> str:
    validate_internal_fields(request)
    context: Final = request.get("_tokenize_context")
    if context is None:
        return "deployment"
    names: Final = (
        tuple(field for field in ("api_base", "base_url") if request.get(field) is not None)
        if name == "api_base"
        else (name,)
    )
    if isinstance(context, TokenizeAdmissionContext):
        return "deployment" if all(field in context.authorized_fields for field in names) else "request"
    if isinstance(context, TokenizeRoutingContext):
        return "deployment" if all(field in context.trusted_fields for field in names) else "request"
    raise ValueError("invalid tokenization context")


def deployment_params(request: Mapping[str, object]) -> Mapping[str, object]:
    validate_internal_fields(request)
    context: Final = request.get("_tokenize_context")
    return context.deployment_params if isinstance(context, TokenizeRoutingContext) else MappingProxyType({})


def deployment_kwargs(
    deployment: Mapping[str, object],
    request: Mapping[str, object],
    *,
    caller_fields: frozenset[str],
) -> Mapping[str, object]:
    validate_internal_fields(request)
    admission: Final = request.get("_tokenize_context")
    authorized: Final = (
        admission.authorized_fields if isinstance(admission, TokenizeAdmissionContext) else frozenset[str]()
    )
    provider_params: Final = body_params(deployment)
    return MappingProxyType(
        {
            **{name: value for name, value in deployment.items() if name not in provider_params},
            **{name: value for name, value in request.items() if name not in provider_params or name in caller_fields},
            "_tokenize_context": TokenizeRoutingContext(
                trusted_fields=(frozenset(deployment) | frozenset(request)).difference(caller_fields) | authorized,
                deployment_params=MappingProxyType(dict(provider_params)),
            ),
        }
    )
