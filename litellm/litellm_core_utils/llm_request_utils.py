from typing import Any, Dict, Optional, Tuple

import litellm


def safe_merge_extra_body(
    data: Dict[str, Any], extra_body: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """Merge ``extra_body`` into an already-transformed request WITHOUT letting it
    overwrite any field the provider transform already produced.

    The proxy authorizes a request against the top-level ``model`` and validates
    ``messages``/``input``/``safetySettings`` etc., then the provider transform
    builds the outbound payload from those validated values. A blind
    ``{**data, **extra_body}`` / ``data.update(extra_body)`` afterwards lets a
    client-supplied ``extra_body`` clobber those post-auth fields (model/cost
    bypass, guardrail/safety-policy override). This merges only keys that the
    transform did not already set, so genuine provider passthrough still works
    but validated fields are immutable. For nested dicts present on both sides,
    the validated (already-present) keys win on collision while new sub-keys are
    still added.
    """
    if not extra_body:
        return data
    for key, value in extra_body.items():
        if key not in data:
            data[key] = value
        elif isinstance(data[key], dict) and isinstance(value, dict):
            data[key] = {**value, **data[key]}
    return data


def strip_validated_keys_from_extra_body(
    body: Dict[str, Any], keys: Tuple[str, ...] = ("model", "messages")
) -> Dict[str, Any]:
    """Return a copy of ``body`` with ``keys`` removed from a nested ``extra_body``.

    The OpenAI / Azure SDKs deep-merge ``extra_body`` *over* the request body, so a
    client-supplied ``extra_body`` could otherwise override the validated top-level
    ``model``/``messages`` (post-auth model/cost bypass) on the native-SDK providers
    where ``extra_body`` is forwarded to the SDK as a kwarg rather than merged into
    the body by litellm. Genuine provider passthrough keys inside ``extra_body`` are
    preserved.
    """
    extra_body = body.get("extra_body")
    if not isinstance(extra_body, dict) or not any(k in extra_body for k in keys):
        return body
    return {
        **body,
        "extra_body": {k: v for k, v in extra_body.items() if k not in keys},
    }


def _ensure_extra_body_is_safe(extra_body: Optional[Dict]) -> Optional[Dict]:
    """
    Ensure that the extra_body sent in the request is safe,  otherwise users will see this error

    "Object of type TextPromptClient is not JSON serializable


    Relevant Issue: https://github.com/BerriAI/litellm/issues/4140
    """
    if extra_body is None:
        return None

    if not isinstance(extra_body, dict):
        return extra_body

    if "metadata" in extra_body and isinstance(extra_body["metadata"], dict):
        if "prompt" in extra_body["metadata"]:
            _prompt = extra_body["metadata"].get("prompt")

            # users can send Langfuse TextPromptClient objects, so we need to convert them to dicts
            # Langfuse TextPromptClients have .__dict__ attribute
            if _prompt is not None and hasattr(_prompt, "__dict__"):
                extra_body["metadata"]["prompt"] = _prompt.__dict__

    return extra_body


def pick_cheapest_chat_models_from_llm_provider(custom_llm_provider: str, n=1):
    """
    Pick the n cheapest chat models from the LLM provider.

    Args:
        custom_llm_provider (str): The name of the LLM provider.
        n (int): The number of cheapest models to return.

    Returns:
        list[str]: A list of the n cheapest chat models.
    """
    if custom_llm_provider not in litellm.models_by_provider:
        return []

    known_models = litellm.models_by_provider.get(custom_llm_provider, [])
    model_costs = []

    for model in known_models:
        try:
            model_info = litellm.get_model_info(
                model=model, custom_llm_provider=custom_llm_provider
            )
        except Exception:
            continue
        if model_info.get("mode") != "chat":
            continue
        _cost = (model_info.get("input_cost_per_token") or 0.0) + (
            model_info.get("output_cost_per_token") or 0.0
        )
        model_costs.append((model, _cost))

    # Sort by cost (ascending)
    model_costs.sort(key=lambda x: x[1])

    # Return the top n cheapest models
    return [model for model, _ in model_costs[:n]]


def get_proxy_server_request_headers(litellm_params: Optional[dict]) -> dict:
    """
    Get the `proxy_server_request` headers from the litellm_params.\

    Use this if you want to access the request headers made to LiteLLM proxy server.
    """
    if litellm_params is None:
        return {}

    proxy_request_headers = (litellm_params.get("proxy_server_request") or {}).get(
        "headers"
    ) or {}

    return proxy_request_headers
