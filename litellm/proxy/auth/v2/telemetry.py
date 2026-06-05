from typing import Any, Dict

from .context import RequestAuthContext


def identity_span_attributes(context: RequestAuthContext) -> Dict[str, Any]:
    """OTel attributes describing the request's identity, for the route span to set.

    Telemetry lives on the route (OTel v2) and reads the auth context here instead
    of auth seeding the span itself. Only present fields are emitted so spans stay
    free of empty attributes.
    """
    identity = context.identity
    attributes: Dict[str, Any] = {
        "litellm.auth.method": context.auth_method.value,
        "litellm.auth.subject": context.principal.subject,
    }
    for attr_name, span_key in (
        ("user_id", "litellm.user_id"),
        ("team_id", "litellm.team_id"),
        ("key_alias", "litellm.key_alias"),
        ("org_id", "litellm.org_id"),
    ):
        value = getattr(identity, attr_name, None)
        if value:
            attributes[span_key] = value
    if context.end_user_id:
        attributes["litellm.end_user_id"] = context.end_user_id
    return attributes
