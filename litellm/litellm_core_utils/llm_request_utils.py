from typing import Dict, Optional

import litellm


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


def pick_cheapest_chat_model_from_llm_provider(custom_llm_provider: str):
    """
    Pick the cheapest chat model from the LLM provider.
    """
    if custom_llm_provider not in litellm.models_by_provider:
        raise ValueError(f"Unknown LLM provider: {custom_llm_provider}")

    known_models = litellm.models_by_provider.get(custom_llm_provider, [])
    min_cost = float("inf")
    cheapest_model = None
    for model in known_models:
        try:
            model_info = litellm.get_model_info(
                model=model, custom_llm_provider=custom_llm_provider
            )
        except:
            continue
        if model_info.get("mode") != "chat":
            continue
        _cost = model_info.get("input_cost_per_token", 0) + model_info.get(
            "output_cost_per_token", 0
        )
        if _cost < min_cost:
            min_cost = _cost
            cheapest_model = model
    return cheapest_model
