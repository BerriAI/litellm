from typing import Dict, Optional

import litellm
from litellm._logging import verbose_logger

# Params that callers (e.g. LangChain) sometimes pass at the top level but
# that are NOT valid top-level OpenAI API fields.  Forwarding them in
# extra_body causes OpenAI to return "400 Unrecognized request argument".
_OPENAI_INVALID_TOP_LEVEL_PARAMS = frozenset({"strict"})


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
    
    # Drop params that are invalid at the OpenAI top-level request body.
    # Some callers (e.g. LangChain) pass these at the completion() top level;
    # they end up in extra_body and cause "400 Unrecognized request argument".
    dropped = [k for k in _OPENAI_INVALID_TOP_LEVEL_PARAMS if k in extra_body]
    for k in dropped:
        extra_body.pop(k)
    if dropped:
        verbose_logger.debug(
            "LiteLLM: dropped invalid top-level OpenAI params from extra_body: %s",
            dropped,
        )

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

    proxy_request_headers = (
        litellm_params.get("proxy_server_request", {}).get("headers", {}) or {}
    )

    return proxy_request_headers
