from __future__ import annotations

from typing import Final

from pydantic import TypeAdapter

from litellm.rust_bridge.lifecycle import Complete

_OBJECT_MAPPING: Final = TypeAdapter(dict[str, object])


def _mapping(value: object) -> dict[str, object]:
    return _OBJECT_MAPPING.validate_python(value)


def _provider(model: object, explicit: object) -> tuple[str, str | None]:
    if not isinstance(model, str):
        raise TypeError("model must be a string")
    if isinstance(explicit, str) and explicit:
        return model.removeprefix(f"{explicit}/"), explicit
    prefix, separator, suffix = model.partition("/")
    if separator and prefix in ("anthropic", "bedrock"):
        return suffix, prefix
    return model, None


class MessagesLifecycleHost:
    def invoke(
        self,
        operation: str,
        payload: object,
        request: object,
        kwargs: dict[str, object],
        logger: object,
    ) -> Complete:
        if operation == "project":
            initial: Final = _mapping(request)
            merged: Final = {**initial, **kwargs}
            model, provider = _provider(merged.get("model"), merged.get("custom_llm_provider"))
            body: Final = _mapping(initial.get("body", {}))
            return Complete(
                {
                    "model": model,
                    "body": {**body, "model": model},
                    "api_key": merged.get("api_key"),
                    "api_base": merged.get("api_base"),
                    "custom_llm_provider": provider,
                    "extra_headers": merged.get("extra_headers"),
                    "timeout": merged.get("timeout"),
                    "litellm_call_id": merged.get("litellm_call_id"),
                }
            )
        if operation in ("response", "cached_response"):
            response: Final = _mapping(payload)
            return Complete({**response, "_hidden_params": {"additional_headers": {"x-litellm-rust": "true"}}})
        if operation == "cache_response":
            return Complete(payload)
        if operation in ("before_request", "after_response", "map_failure"):
            return Complete(payload)
        if operation == "post_process":
            return Complete(None)
        raise ValueError(f"unknown messages lifecycle operation: {operation}")


HOST: Final = MessagesLifecycleHost()
