"""The Python proxy's answers to the request body check.

The check and every rule in it are Rust (``gateway-auth-core/src/body``). It asks the process it runs in through two
traits in ``body/host.rs``, ``Host`` and ``Deployments``, and each function here answers one of their methods. Nothing
here decides whether a body is accepted. A function either reads state this process holds (the router, the stored
credentials, a setting) or runs the parser the proxy later reads the body with, so the check judges what the consumer
will see.

Only ``python-bridge/src/auth/body.rs`` calls these. A host without Python, such as the gateway, implements the same
traits natively and never loads this module."""

from collections.abc import Mapping
from typing import Final, TypeAlias

from pydantic import TypeAdapter, ValidationError

import litellm
from litellm import Router
from litellm.litellm_core_utils.credential_accessor import CredentialAccessor
from litellm.litellm_core_utils.safe_json_loads import safe_json_loads
from litellm.litellm_core_utils.url_utils import is_url_destination_allowed_by_host, provider_url_destination_candidates
from litellm.proxy.auth.auth_utils import (
    EVERY_FALLBACK_TYPE,
    check_regex_or_str_match,
    deployment_clientside_auth_params,
    deployment_provider,
    model_group_deployments,
    router_fallback_model_names,
)
from litellm.proxy.common_utils.http_parsing_utils import extract_nested_form_metadata
from litellm.types.router import LiteLLMParamsTypedDict

_MAPPING: Final = TypeAdapter(Mapping[str, object])
_ALLOWED_HOSTS: Final = TypeAdapter(tuple[str, ...])

OptIn: TypeAlias = tuple[tuple[str, ...], tuple[str, ...]]
GroupDeployment: TypeAlias = tuple[str, str | None, OptIn | None]


def decode_mapping(text: str) -> Mapping[str, object] | None:
    """``Host::decode``. A JSON string a caller sent where a mapping belongs, such as ``extra_body`` or ``metadata``,
    parsed with the proxy's own ``safe_json_loads``. ``None`` when it does not hold a mapping."""
    try:
        return _MAPPING.validate_python(safe_json_loads(text))
    except ValidationError:
        return None


def form_metadata(body: Mapping[str, object], prefix: str) -> Mapping[str, object]:
    """``Host::form_metadata``. The mapping a form's bracket-spelled fields (``metadata[tags]``) build, read with
    the same ``extract_nested_form_metadata`` the form handlers use. ``prefix`` is ``metadata[`` or
    ``litellm_metadata[``."""
    return extract_nested_form_metadata(form_data=body, prefix=prefix)


def credential_values(name: str) -> Mapping[str, object]:
    """``Host::credential``. The values stored under a ``litellm_credential_name``, empty when no credential has
    that name."""
    return _MAPPING.validate_python(CredentialAccessor.get_credential_values(name))


def url_destination_allowed(value: str) -> bool:
    """``Host::url_destination_allowed``. Whether every HTTP destination a URL-valued fallback target could send to
    is a host ``litellm.provider_url_destination_allowed_hosts`` allows."""
    allowed_hosts: Final = _ALLOWED_HOSTS.validate_python(
        getattr(litellm, "provider_url_destination_allowed_hosts", None) or ()
    )
    return all(
        is_url_destination_allowed_by_host(candidate, list(allowed_hosts))
        for candidate in provider_url_destination_candidates(value)
        if candidate.lower().startswith(("http://", "https://"))
    )


def _opt_in(litellm_params: LiteLLMParamsTypedDict) -> OptIn | None:
    """A deployment's ``configurable_clientside_auth_params`` as parameter names and ``api_base`` patterns, ``None``
    when it sets none."""
    items: Final = deployment_clientside_auth_params(litellm_params)
    if items is None:
        return None
    return (
        tuple(item for item in items if isinstance(item, str)),
        tuple(item["api_base"] for item in items if not isinstance(item, str)),
    )


def group_deployments(router: Router, model: str, team_id: str | None) -> tuple[GroupDeployment, ...]:
    """``Deployments::group``. Every deployment that can serve ``model`` for this team, each as its model group
    name, its provider and what it lets a caller override."""
    return tuple(
        (
            deployment["model_name"],
            deployment_provider(deployment["litellm_params"]),
            _opt_in(deployment["litellm_params"]),
        )
        for deployment in model_group_deployments(model, router, team_id)
    )


def alias(router: Router, name: str) -> str | None:
    """``Deployments::alias``. The model group ``name`` is a router alias of, ``None`` when it is not an alias."""
    return router.registry().resolve(name, kind="alias") or None


def fallbacks(router: Router, name: str) -> tuple[str, ...]:
    """``Deployments::fallbacks``. Every model group the router can fall back to from ``name``, through fallbacks
    of every kind, sorted so the check visits them in a stable order."""
    return tuple(sorted(router_fallback_model_names(name, router, EVERY_FALLBACK_TYPE)))


def api_base_matches(pattern: str, value: str) -> bool:
    """``Deployments::api_base_matches``. Whether a caller's ``api_base`` equals the pattern a deployment opted in
    with, or matches it as a Python regex."""
    return check_regex_or_str_match(request_body_value=value, regex_str=pattern)
