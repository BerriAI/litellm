LITELLM_INTERNAL_PARAM_NAMES = frozenset(
    (
        "litellm_params",
        "proxy_server_request",
        "model_info",
        "preset_cache_key",
        "litellm_metadata",
        "acompletion",
    )
)


def strip_litellm_internal_params(
    data: dict[str, object],  # mutable-ok: interface expects mutable dict payload
) -> dict[str, object]:  # mutable-ok: returns mutable dict payload
    """
    Remove LiteLLM internal params (e.g. litellm_params, proxy_server_request, _litellm_ prefixed keys)
    from request data before passing to client libraries (e.g. OpenAI).

    This avoids throwing API validation/schema errors (e.g. 400 Bad Request) due to unknown parameters.
    """
    if not isinstance(data, dict):  # pyright: ignore[reportUnnecessaryIsInstance]  # runtime guard for unsanitized input
        return data  # pyright: ignore[reportUnreachable]  # runtime guard

    # Create a shallow copy so we don't modify the input dictionary in-place
    cleaned_data: dict[str, object] = {}  # mutable-ok: building cleaned payload dict
    for key, value in data.items():
        if key in LITELLM_INTERNAL_PARAM_NAMES or key.startswith("_litellm_"):
            continue
        if key == "extra_body" and isinstance(value, dict):
            cleaned_extra_body: dict[str, object] = {}  # mutable-ok: building cleaned extra_body dict
            extra_body_dict: dict[object, object] = value  # pyright: ignore[reportUnknownVariableType]  # mutable-ok: reading from extra_body
            for k, v in extra_body_dict.items():
                if isinstance(k, str) and (k in LITELLM_INTERNAL_PARAM_NAMES or k.startswith("_litellm_")):
                    continue
                cleaned_extra_body[str(k)] = v
            cleaned_data["extra_body"] = cleaned_extra_body
        else:
            cleaned_data[key] = value
    return cleaned_data
