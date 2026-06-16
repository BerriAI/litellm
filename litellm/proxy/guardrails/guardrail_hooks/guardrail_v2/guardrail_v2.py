import json
import os
from typing import TYPE_CHECKING, Literal, Optional

from litellm.proxy.guardrails import _rust

from litellm._version import version as litellm_version
from litellm.exceptions import GuardrailRaisedException
from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    log_guardrail_information,
)
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.utils import GenericGuardrailAPIInputs

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.types.guardrails import LitellmParams


def _extract_raw_headers(
    request_data: dict,
    logging_obj: Optional["LiteLLMLoggingObj"],
) -> Optional[dict]:
    for source in (
        (request_data.get("proxy_server_request") or {}).get("headers"),
        request_data.get("headers"),
        (request_data.get("metadata") or {}).get("headers"),
        (request_data.get("litellm_metadata") or {}).get("headers"),
    ):
        if source and isinstance(source, dict):
            return {str(k): str(v) for k, v in source.items() if k is not None}

    if logging_obj and getattr(logging_obj, "model_call_details", None):
        try:
            hdrs = (
                (logging_obj.model_call_details or {})
                .get("litellm_params", {})
                .get("metadata", {})
                .get("headers")
            )
            if hdrs and isinstance(hdrs, dict):
                return {str(k): str(v) for k, v in hdrs.items() if k is not None}
        except Exception:
            pass
    return None


def _extract_user_metadata(request_data: dict) -> dict:
    top = request_data.get("metadata") or {}
    litellm_meta = request_data.get("litellm_metadata") or {}
    merged = {**top, **litellm_meta}

    keys = (
        "user_api_key_hash",
        "user_api_key_alias",
        "user_api_key_user_id",
        "user_api_key_user_email",
        "user_api_key_team_id",
        "user_api_key_team_alias",
        "user_api_key_end_user_id",
        "user_api_key_org_id",
    )
    out = {k: merged[k] for k in keys if merged.get(k) is not None}
    if "user_api_key_token" in merged and "user_api_key_hash" not in out:
        out["user_api_key_hash"] = merged["user_api_key_token"]
    return out


def _resolve_secret(value: Optional[str]) -> Optional[str]:
    if value and value.startswith("os.environ/"):
        return os.environ.get(value.removeprefix("os.environ/"))
    return value


def build_v2_config(
    guardrail_type: str, litellm_params: "LitellmParams"
) -> Optional[dict]:
    """Map litellm_params to the engine ProviderConfig JSON for this guardrail type.

    Returns None when the config uses features the Rust engine does not support
    yet, in which case the caller must fall back to the Python implementation.
    """
    api_key = _resolve_secret(getattr(litellm_params, "api_key", None))
    api_base = _resolve_secret(getattr(litellm_params, "api_base", None))

    if guardrail_type == "generic_guardrail_api":
        if not api_base:
            return None
        return {
            "guardrail": guardrail_type,
            "api_base": api_base,
            "api_key": api_key,
            "headers": getattr(litellm_params, "headers", None),
            "additional_provider_specific_params": getattr(
                litellm_params, "additional_provider_specific_params", None
            )
            or {},
            "unreachable_fallback": getattr(
                litellm_params, "unreachable_fallback", None
            ),
        }

    if guardrail_type == "openai_moderation":
        return {
            "guardrail": guardrail_type,
            "api_key": api_key or os.environ.get("OPENAI_API_KEY"),
            "api_base": api_base,
            "model": getattr(litellm_params, "model", None),
        }

    if guardrail_type == "azure/prompt_shield":
        if not api_base:
            return None
        return {
            "guardrail": guardrail_type,
            "api_key": api_key,
            "api_base": api_base,
            "api_version": getattr(litellm_params, "api_version", None),
        }

    if guardrail_type == "azure/text_moderations":
        if not api_base:
            return None
        severity_threshold = getattr(litellm_params, "severity_threshold", None)
        by_category = (
            getattr(litellm_params, "severity_threshold_by_category", None) or {}
        )
        return {
            "guardrail": guardrail_type,
            "api_key": api_key,
            "api_base": api_base,
            "api_version": getattr(litellm_params, "api_version", None),
            "severity_threshold": (
                int(severity_threshold) if severity_threshold is not None else None
            ),
            "severity_threshold_by_category": {
                str(k): int(v) for k, v in by_category.items()
            },
            "categories": getattr(litellm_params, "categories", None),
            "blocklist_names": getattr(litellm_params, "blocklistNames", None) or [],
            "halt_on_blocklist_hit": bool(
                getattr(litellm_params, "haltOnBlocklistHit", None)
            ),
            "output_type": getattr(litellm_params, "outputType", None),
        }

    if guardrail_type == "presidio":
        if getattr(litellm_params, "output_parse_pii", None) or getattr(
            litellm_params, "mock_redacted_text", None
        ):
            return None
        analyzer_base = _resolve_secret(
            getattr(litellm_params, "presidio_analyzer_api_base", None)
        ) or os.environ.get("PRESIDIO_ANALYZER_API_BASE")
        anonymizer_base = _resolve_secret(
            getattr(litellm_params, "presidio_anonymizer_api_base", None)
        ) or os.environ.get("PRESIDIO_ANONYMIZER_API_BASE")
        if not analyzer_base:
            return None

        ad_hoc_recognizers = None
        recognizers_path = getattr(litellm_params, "presidio_ad_hoc_recognizers", None)
        if recognizers_path:
            try:
                with open(recognizers_path) as f:
                    ad_hoc_recognizers = json.load(f).get("recognizers")
            except (OSError, json.JSONDecodeError):
                return None

        return {
            "guardrail": guardrail_type,
            "presidio_analyzer_api_base": analyzer_base,
            "presidio_anonymizer_api_base": anonymizer_base,
            "pii_entities_config": getattr(litellm_params, "pii_entities_config", None)
            or {},
            "presidio_language": getattr(litellm_params, "presidio_language", None),
            "presidio_score_thresholds": getattr(
                litellm_params, "presidio_score_thresholds", None
            )
            or {},
            "presidio_entities_deny_list": getattr(
                litellm_params, "presidio_entities_deny_list", None
            )
            or [],
            "presidio_ad_hoc_recognizers": ad_hoc_recognizers,
        }

    if guardrail_type == "lakera_v2":
        return {
            "guardrail": guardrail_type,
            "api_key": api_key or os.environ.get("LAKERA_API_KEY"),
            "api_base": api_base,
            "project_id": getattr(litellm_params, "project_id", None),
            "payload": getattr(litellm_params, "payload", None),
            "breakdown": getattr(litellm_params, "breakdown", None),
            "metadata": getattr(litellm_params, "metadata", None),
            "dev_info": getattr(litellm_params, "dev_info", None),
            "on_flagged": getattr(litellm_params, "on_flagged", None),
        }

    if guardrail_type == "bedrock":
        unsupported = (
            getattr(litellm_params, "aws_role_name", None),
            getattr(litellm_params, "aws_profile_name", None),
            getattr(litellm_params, "aws_web_identity_token", None),
            getattr(litellm_params, "disable_exception_on_block", None),
        )
        if any(unsupported):
            return None
        identifier = getattr(litellm_params, "guardrailIdentifier", None)
        version = getattr(litellm_params, "guardrailVersion", None)
        if not identifier or not version:
            return None
        return {
            "guardrail": guardrail_type,
            "guardrailIdentifier": identifier,
            "guardrailVersion": version,
            "aws_region_name": getattr(litellm_params, "aws_region_name", None),
            "aws_access_key_id": _resolve_secret(
                getattr(litellm_params, "aws_access_key_id", None)
            ),
            "aws_secret_access_key": _resolve_secret(
                getattr(litellm_params, "aws_secret_access_key", None)
            ),
            "aws_session_token": _resolve_secret(
                getattr(litellm_params, "aws_session_token", None)
            ),
            "aws_bedrock_runtime_endpoint": getattr(
                litellm_params, "aws_bedrock_runtime_endpoint", None
            ),
        }

    return None


class GuardrailV2(CustomGuardrail):
    def __init__(
        self,
        engine_config: dict,
        extra_headers: Optional[list] = None,
        **kwargs: object,
    ):
        self.engine_config = engine_config
        self.extra_header_allowlist = [
            h for h in (extra_headers or []) if isinstance(h, str)
        ]

        if "supported_event_hooks" not in kwargs:
            kwargs["supported_event_hooks"] = [
                GuardrailEventHooks.pre_call,
                GuardrailEventHooks.post_call,
                GuardrailEventHooks.during_call,
            ]

        super().__init__(**kwargs)

    @log_guardrail_information
    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: Optional["LiteLLMLoggingObj"] = None,
    ) -> GenericGuardrailAPIInputs:
        request_body = (request_data or {}).get("body") or {}
        dynamic_params = self.get_guardrail_dynamic_request_body_params(request_body)

        payload = {
            "config": self.engine_config,
            "input": {
                "texts": inputs.get("texts", []),
                "images": inputs.get("images") or [],
                "structured_messages": inputs.get("structured_messages") or [],
                "tools": inputs.get("tools") or [],
                "tool_calls": inputs.get("tool_calls") or [],
                "model": inputs.get("model"),
            },
            "input_type": input_type,
            "context": {
                "litellm_call_id": (
                    getattr(logging_obj, "litellm_call_id", None)
                    if logging_obj
                    else None
                ),
                "litellm_trace_id": (
                    getattr(logging_obj, "litellm_trace_id", None)
                    if logging_obj
                    else None
                ),
                "user_api_key_metadata": _extract_user_metadata(request_data or {}),
                "request_headers": _extract_raw_headers(
                    request_data or {}, logging_obj
                ),
                "dynamic_params": dynamic_params or {},
                "litellm_version": litellm_version,
                "extra_header_allowlist": self.extra_header_allowlist,
            },
            "timeout_ms": 10000,
        }

        result_json = await _rust.apply_guardrail(json.dumps(payload, default=str))
        result = json.loads(result_json)
        verdict = result.get("verdict", {})
        action = verdict.get("action")

        if action == "block":
            raise GuardrailRaisedException(
                guardrail_name=self.guardrail_name
                or self.engine_config.get("guardrail", "guardrail_v2"),
                message=verdict.get("violation_message", "Content violates policy"),
                should_wrap_with_default_message=False,
            )

        if action == "mask":
            out: GenericGuardrailAPIInputs = {}
            out.update(inputs)
            masked = verdict.get("texts")
            if masked is not None:
                out["texts"] = masked
            return out

        out = {}
        out.update(inputs)
        return out
