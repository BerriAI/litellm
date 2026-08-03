from __future__ import annotations

from litellm.types.utils import StandardLoggingPayload

# Tiers a caller may name in a request, across the providers that accept the
# parameter: OpenAI ("auto", "default", "flex", "priority", "scale"), Bedrock and
# Groq (subsets of those), Anthropic ("auto", "standard_only") and Vertex, which
# maps "default" to "standard". Used to bound the caller-controlled fallback in
# ``get_service_tier_from_standard_logging_payload``.
KNOWN_REQUEST_SERVICE_TIERS = frozenset(
    {"auto", "batch", "default", "flex", "priority", "scale", "standard", "standard_only"}
)


def get_served_service_tier(standard_logging_payload: StandardLoggingPayload) -> str | None:
    """
    The tier the provider reports it actually served the request on.

    Providers report it either at the top level of the response (OpenAI, Bedrock,
    Groq) or on the usage object (Anthropic). Streaming responses carry no served
    tier.
    """
    response = standard_logging_payload.get("response")
    usage_object = standard_logging_payload.get("metadata", {}).get("usage_object")

    served_candidates: tuple[object, ...] = (
        response.get("service_tier") if isinstance(response, dict) else None,
        usage_object.get("service_tier") if isinstance(usage_object, dict) else None,
    )
    return next((tier for tier in served_candidates if isinstance(tier, str) and tier), None)


def get_requested_service_tier(standard_logging_payload: StandardLoggingPayload) -> str | None:
    """The tier the caller asked for, as sent to the provider."""
    model_parameters = standard_logging_payload.get("model_parameters")
    requested_tier = model_parameters.get("service_tier") if isinstance(model_parameters, dict) else None
    return requested_tier if isinstance(requested_tier, str) and requested_tier else None


def get_service_tier_from_standard_logging_payload(
    standard_logging_payload: StandardLoggingPayload,
) -> str | None:
    """
    Resolve the service tier a request ran on, for the Prometheus ``service_tier`` label.

    The tier the provider actually served wins over the tier the caller asked for,
    so latency and spend stay segmentable when the request said ``auto`` and the
    provider picked the concrete tier.

    Streaming responses carry no served tier, so the requested tier is the
    fallback. That value is caller-controlled and survives param mapping even
    where the provider then ignores it (Bedrock and Groq accept the request and
    drop an unrecognized tier), so it is only labelled when it names a known
    tier; otherwise one caller could mint a Prometheus series per string. Values
    the provider itself reports are not caller-controlled and stay unrestricted,
    so a tier a provider adds later is still labelled correctly.
    """
    served_tier = get_served_service_tier(standard_logging_payload)
    if served_tier is not None:
        return served_tier

    requested_tier = get_requested_service_tier(standard_logging_payload)
    if requested_tier is not None and requested_tier in KNOWN_REQUEST_SERVICE_TIERS:
        return requested_tier
    return None
