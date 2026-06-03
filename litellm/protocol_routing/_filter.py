from typing import List, Optional

from litellm.protocol_routing._types import (
    ProtocolMismatchError,
    ProtocolRoutingMode,
    SupportedProtocol,
    get_protocol_routing_mode,
    ROUTE_TYPE_TO_PROTOCOL,
)
from litellm.protocol_routing._mapping import infer_protocols


def _extract_provider(deployment: dict) -> Optional[str]:
    """Extract provider from deployment, falling back to model-prefix parsing.

    Most deployments leave litellm_params.custom_llm_provider unset and rely
    on the "provider/model" prefix in litellm_params.model. Strict-mode
    filtering needs the same resolution rule the rest of the router uses.
    """
    litellm_params = deployment.get("litellm_params")
    if litellm_params is None:
        return None
    provider = litellm_params.get("custom_llm_provider")
    if provider:
        return provider
    model = litellm_params.get("model")
    if not model:
        return None
    try:
        import litellm as _litellm
        _, provider, _, _ = _litellm.get_llm_provider(model=model)
        return provider
    except Exception:
        return None


def _extract_model_info(deployment: dict) -> Optional[dict]:
    """Extract model_info from deployment."""
    return deployment.get("model_info")


def filter_deployments_by_protocol(
    healthy_deployments: List[dict],
    route_type: str,
    model: str,
    mode: Optional[ProtocolRoutingMode] = None,
) -> List[dict]:
    """Filter deployments by protocol compatibility in strict mode.

    In bridged mode (default), this is a no-op that returns all deployments.
    In strict mode, only returns deployments whose provider supports the
    requested protocol.

    Args:
        healthy_deployments: List of deployment dicts from router
        route_type: The route type (e.g., "anthropic_messages", "acompletion")
        model: The model name requested by the user
        mode: Optional override for protocol routing mode

    Returns:
        Filtered list of deployments (or all deployments in bridged mode)

    Raises:
        ProtocolMismatchError: In strict mode when no deployments support the protocol
    """
    if not healthy_deployments:
        return healthy_deployments

    if route_type is None:
        return healthy_deployments

    # Determine requested protocol from route type
    requested_protocol = ROUTE_TYPE_TO_PROTOCOL.get(route_type)
    if requested_protocol is None:
        # Unknown route type - don't filter (bridged behavior)
        return healthy_deployments

    # Pass-through routes bypass protocol filtering
    if requested_protocol == SupportedProtocol.LLM_PASSTHROUGH:
        return healthy_deployments

    # Get effective mode (explicit override > global setting)
    effective_mode = mode if mode is not None else get_protocol_routing_mode()

    # Bridged mode: allow all deployments (protocol conversion enabled)
    if effective_mode == "bridged":
        return healthy_deployments

    # Strict mode: filter to protocol-compatible deployments only
    compatible_deployments = []
    available_protocols = set()

    for deployment in healthy_deployments:
        provider = _extract_provider(deployment)
        model_info = _extract_model_info(deployment)
        deployment_protocols = infer_protocols(provider, model_info)

        # Track all available protocols for error message
        available_protocols.update(deployment_protocols)

        # Check if this deployment supports the requested protocol
        if requested_protocol in deployment_protocols:
            compatible_deployments.append(deployment)

    # If no compatible deployments found, raise clear error
    if not compatible_deployments:
        available_protocol_names = sorted(
            {p.value for p in available_protocols}
        )
        raise ProtocolMismatchError(
            model=model,
            requested_protocol=requested_protocol.value,
            available_protocols=available_protocol_names,
        )

    return compatible_deployments


def check_strict_protocol_for_provider(
    provider: Optional[str],
    requested_protocol: SupportedProtocol,
    model: str,
    mode: Optional[ProtocolRoutingMode] = None,
) -> None:
    """Check if a provider supports the requested protocol in strict mode.

    This is called at the protocol conversion decision point (e.g., when
    deciding whether to use the Anthropic adapter for a non-Anthropic provider).

    Args:
        provider: The LLM provider name
        requested_protocol: The protocol being requested
        model: The model name (for error messages)
        mode: Optional override for protocol routing mode

    Raises:
        ProtocolMismatchError: In strict mode when provider doesn't support the protocol
    """
    # Get effective mode
    effective_mode = mode if mode is not None else get_protocol_routing_mode()

    # Bridged mode: allow conversion
    if effective_mode == "bridged":
        return

    # Strict mode: check provider support
    provider_protocols = infer_protocols(provider)
    if requested_protocol not in provider_protocols:
        available_protocol_names = sorted(
            {p.value for p in provider_protocols}
        )
        raise ProtocolMismatchError(
            model=model,
            requested_protocol=requested_protocol.value,
            available_protocols=available_protocol_names,
        )
