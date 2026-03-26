from typing import Any, Dict, Optional

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


def contains_surrogate_code_point(value: str) -> bool:
    return any(0xD800 <= ord(char) <= 0xDFFF for char in value)


def sanitize_surrogate_code_points(value: str) -> str:
    if not contains_surrogate_code_point(value):
        return value
    return value.encode("utf-16", "surrogatepass").decode("utf-16", "replace")


def _format_path_key(key: str) -> str:
    return key.encode("unicode_escape").decode("ascii")


def sanitize_request_payload(value: Any) -> Any:
    if isinstance(value, str):
        return sanitize_surrogate_code_points(value)

    if isinstance(value, dict):
        return {
            (sanitize_surrogate_code_points(key) if isinstance(key, str) else key): sanitize_request_payload(
                nested_value
            )
            for key, nested_value in value.items()
        }

    if isinstance(value, list):
        return [sanitize_request_payload(nested_value) for nested_value in value]

    return value


def find_surrogate_code_point_path(
    value: Any,
    path: str = "payload",
) -> Optional[str]:
    """
    Recursively searches nested structures for strings or dict keys containing Unicode surrogate code points.

    Args:
        value: The value to inspect (str, dict, list, or other).
        path: Base path to use when reporting the surrogate location.

    Returns:
        Optional[str] path to the first occurrence or None.
        Examples: "payload.key", "payload[key][0]"
    """
    if isinstance(value, str):
        if contains_surrogate_code_point(value):
            return path
        return None

    if isinstance(value, dict):
        for key, nested_value in value.items():
            if isinstance(key, str) and contains_surrogate_code_point(key):
                return f"{path}.{_format_path_key(key)}"
            nested_path = find_surrogate_code_point_path(
                nested_value,
                path=f"{path}.{_format_path_key(key) if isinstance(key, str) else key}",
            )
            if nested_path is not None:
                return nested_path
        return None

    if isinstance(value, list):
        for index, nested_value in enumerate(value):
            nested_path = find_surrogate_code_point_path(
                nested_value,
                path=f"{path}[{index}]",
            )
            if nested_path is not None:
                return nested_path
        return None

    return None


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
            model_info = litellm.get_model_info(model=model, custom_llm_provider=custom_llm_provider)
        except Exception:
            continue
        if model_info.get("mode") != "chat":
            continue
        _cost = model_info.get("input_cost_per_token", 0) + model_info.get("output_cost_per_token", 0)
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

    proxy_request_headers = litellm_params.get("proxy_server_request", {}).get("headers", {}) or {}

    return proxy_request_headers
