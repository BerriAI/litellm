from collections.abc import Mapping
from typing import Final

import litellm


def _form_field_value(value: object) -> str:
    if value is True:
        return "true"
    if value is False:
        return "false"
    return str(value)


def _flatten_form_field(key: str, value: object) -> tuple[tuple[str, str], ...]:
    if isinstance(value, Mapping):
        return tuple(
            item for subkey, subvalue in value.items() for item in _flatten_form_field(f"{key}[{subkey}]", subvalue)
        )
    if isinstance(value, (list, tuple)):
        return tuple(item for entry in value for item in _flatten_form_field(f"{key}[]", entry))
    if value is None:
        return ()
    serialized: Final = _form_field_value(value)
    if not serialized:
        return ()
    return ((key, serialized),)


def _is_form_scalar(value: object) -> bool:
    return value is not None and not isinstance(value, (Mapping, list, tuple))


def _flatten_form_data_field(key: str, value: object) -> tuple[tuple[str, str | tuple[str, ...]], ...]:
    if isinstance(value, Mapping):
        return tuple(
            item
            for subkey, subvalue in value.items()
            for item in _flatten_form_data_field(f"{key}[{subkey}]", subvalue)
        )
    if isinstance(value, (list, tuple)):
        if all(_is_form_scalar(entry) for entry in value):
            serialized_fields: Final = tuple(field for entry in value if (field := _form_field_value(entry)))
            return ((key, serialized_fields),) if serialized_fields else ()
        return tuple(item for entry in value for item in _flatten_form_data_field(f"{key}[]", entry))
    if value is None:
        return ()
    serialized: Final = _form_field_value(value)
    if not serialized:
        return ()
    return ((key, serialized),)


def flatten_form_field_values(*sources: Mapping[str, object] | None) -> tuple[tuple[str, str | tuple[str, ...]], ...]:
    """
    Flatten JSON-shaped bodies into ``(name, value)`` form fields for a ``dict``-backed
    multipart body, applying ``sources`` in order so a later source wins on a key collision
    under ``dict.update``. Nested objects become ``key[subkey]`` fields the way the OpenAI SDK
    serializes them, so provider params reach a multipart request without handing the httpx
    encoder a nested value it rejects with ``Invalid type for value``. A scalar list becomes a
    single field carrying a tuple value, which httpx emits as one repeated part per element, so
    every element survives instead of collapsing to the last under ``dict.update``.
    """
    return tuple(
        pair
        for source in sources
        if source is not None
        for top_key, top_value in source.items()
        for pair in _flatten_form_data_field(top_key, top_value)
    )


def serialize_multipart_form_fields(data: Mapping[str, object]) -> tuple[tuple[str, tuple[None, str]], ...]:
    """
    Encode a JSON-shaped body as OpenAI-SDK-style multipart file-tuples so a file-less
    request is still sent as multipart/form-data, working around httpx downgrading a
    file-less ``data=`` payload to application/x-www-form-urlencoded.
    """
    return tuple(
        (key, (None, serialized))
        for top_key, top_value in data.items()
        for key, serialized in _flatten_form_field(top_key, top_value)
    )


def _ensure_extra_body_is_safe(extra_body: dict | None) -> dict | None:
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
            _prompt: Final = extra_body["metadata"].get("prompt")

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

    known_models: Final = litellm.models_by_provider.get(custom_llm_provider, [])
    model_costs: Final = []

    for model in known_models:
        try:
            model_info = litellm.get_model_info(model=model, custom_llm_provider=custom_llm_provider)
        except Exception:
            continue
        if model_info.get("mode") != "chat":
            continue
        _cost = (model_info.get("input_cost_per_token") or 0.0) + (model_info.get("output_cost_per_token") or 0.0)
        model_costs.append((model, _cost))

    # Sort by cost (ascending)
    model_costs.sort(key=lambda x: x[1])

    # Return the top n cheapest models
    return [model for model, _ in model_costs[:n]]


def get_proxy_server_request_headers(litellm_params: dict | None) -> dict:
    """
    Get the `proxy_server_request` headers from the litellm_params.\

    Use this if you want to access the request headers made to LiteLLM proxy server.
    """
    if litellm_params is None:
        return {}

    proxy_request_headers: Final = (litellm_params.get("proxy_server_request") or {}).get("headers") or {}

    return proxy_request_headers
