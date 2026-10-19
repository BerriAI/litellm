"""Python proxy stand-ins for the Rust gateway's four request hooks, so a load test can run both
gateways with the same work per request. genconfig -features full lists them as
custom_callbacks.<name> and writes this file beside the config.yaml it generates.

Each mirrors the Rust hook configured in that config (litellm-rust/crates/gateway/src/request_hooks.rs):

    forward_headers    copies x-request-origin and x-trace-* to the provider request
    spend_metadata     adds x-litellm-spend-logs-metadata (JSON) and x-cost-center to the
                       request's spend_logs_metadata; the key's cost_center wins
    strip_headers      drops x-litellm-spend-logs-metadata and x-cost-center from the request
    provider_metadata  sets metadata.user_id in the provider body and x-caller-team in its headers
"""

import json
from typing import Any

from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy._types import UserAPIKeyAuth

FORWARDED = ("x-request-origin",)
FORWARDED_PREFIXES = ("x-trace-",)
STRIPPED = ("x-litellm-spend-logs-metadata", "x-cost-center")


def _client_headers(data: dict) -> dict:
    request = data.get("proxy_server_request") or {}
    headers = request.get("headers")
    return headers if isinstance(headers, dict) else {}


def _litellm_metadata(data: dict) -> dict:
    # /v1/messages and other newer routes keep proxy metadata under litellm_metadata.
    name = "litellm_metadata" if "litellm_metadata" in data else "metadata"
    metadata = data.get(name)
    if not isinstance(metadata, dict):
        metadata = {}
        data[name] = metadata
    return metadata


def _extra(data: dict, field: str) -> dict:
    value = data.get(field)
    if not isinstance(value, dict):
        value = {}
        data[field] = value
    return value


class ForwardHeaders(CustomLogger):
    async def async_pre_call_hook(
        self, user_api_key_dict: UserAPIKeyAuth, cache: Any, data: dict, call_type: str
    ) -> dict | None:
        headers = _client_headers(data)
        forwarded = {
            name: value
            for name, value in headers.items()
            if name.lower() in FORWARDED or name.lower().startswith(FORWARDED_PREFIXES)
        }
        if forwarded:
            _extra(data, "extra_headers").update(forwarded)
        return data


class SpendMetadata(CustomLogger):
    async def async_pre_call_hook(
        self, user_api_key_dict: UserAPIKeyAuth, cache: Any, data: dict, call_type: str
    ) -> dict | None:
        headers = {name.lower(): value for name, value in _client_headers(data).items()}
        added: dict = {}
        raw = headers.get("x-litellm-spend-logs-metadata")
        if raw:
            try:
                parsed = json.loads(raw)
            except ValueError:
                parsed = None
            if isinstance(parsed, dict):
                added.update(parsed)
        if headers.get("x-cost-center"):
            added["cost_center"] = headers["x-cost-center"]
        key_metadata = user_api_key_dict.metadata or {}
        if "cost_center" in key_metadata:
            added["cost_center"] = key_metadata["cost_center"]
        if added:
            metadata = _litellm_metadata(data)
            spend = metadata.get("spend_logs_metadata")
            spend = dict(spend) if isinstance(spend, dict) else {}
            spend.update(added)
            metadata["spend_logs_metadata"] = spend
        return data


class StripHeaders(CustomLogger):
    async def async_pre_call_hook(
        self, user_api_key_dict: UserAPIKeyAuth, cache: Any, data: dict, call_type: str
    ) -> dict | None:
        headers = _client_headers(data)
        for name in [name for name in headers if name.lower() in STRIPPED]:
            headers.pop(name)
        extra = data.get("extra_headers")
        if isinstance(extra, dict):
            for name in [name for name in extra if name.lower() in STRIPPED]:
                extra.pop(name)
        return data


class ProviderMetadata(CustomLogger):
    async def async_pre_call_hook(
        self, user_api_key_dict: UserAPIKeyAuth, cache: Any, data: dict, call_type: str
    ) -> dict | None:
        user_id = user_api_key_dict.end_user_id or user_api_key_dict.user_id
        if user_id:
            body_metadata = _extra(data, "extra_body").setdefault("metadata", {})
            if isinstance(body_metadata, dict):
                body_metadata["user_id"] = user_id
        if user_api_key_dict.team_alias:
            _extra(data, "extra_headers")["x-caller-team"] = user_api_key_dict.team_alias
        return data


forward_headers = ForwardHeaders()
spend_metadata = SpendMetadata()
strip_headers = StripHeaders()
provider_metadata = ProviderMetadata()
