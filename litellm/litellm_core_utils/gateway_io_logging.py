from __future__ import annotations

import base64
import json
import re
import sys
from collections.abc import Callable, Mapping
from typing import Any, Final
from urllib.parse import parse_qsl, quote, urlencode, urlsplit, urlunsplit

MASKED_VALUE: Final[str] = "***"
MAX_HEADER_COUNT: Final[int] = 100
MAX_HEADER_VALUE_LEN: Final[int] = 512
MAX_BODY_SCALAR_LEN: Final[int] = 256
MAX_ERROR_MSG_LEN: Final[int] = 1000
MAX_REGEX_SCAN_LEN: Final[int] = 4096
MAX_ESTIMATE_DEPTH: Final[int] = 6
MAX_ESTIMATE_NODES: Final[int] = 500
MAX_NESTED_SANITIZE_DEPTH: Final[int] = 3

AUTH_HEADER_NAMES: Final[frozenset[str]] = frozenset(
    {
        "authorization",
        "x-api-key",
        "api-key",
        "x-goog-api-key",
        "x-litellm-api-key",
        "x-auth-token",
        "ocp-apim-subscription-key",
        "anthropic-api-key",
        "openai-api-key",
    }
)

SAFE_HEADER_NAMES: Final[frozenset[str]] = frozenset(
    {
        "accept",
        "accept-encoding",
        "accept-language",
        "alt-svc",
        "anthropic-beta",
        "anthropic-organization-id",
        "anthropic-version",
        "cache-control",
        "cf-cache-status",
        "cf-ray",
        "connection",
        "content-encoding",
        "content-length",
        "content-type",
        "date",
        "etag",
        "host",
        "openai-organization",
        "openai-processing-ms",
        "openai-project",
        "openai-version",
        "pragma",
        "request-id",
        "retry-after",
        "server",
        "strict-transport-security",
        "traceparent",
        "tracestate",
        "transfer-encoding",
        "user-agent",
        "vary",
        "x-amzn-requestid",
        "x-b3-sampled",
        "x-b3-spanid",
        "x-b3-traceid",
        "x-cloud-trace-context",
        "x-content-type-options",
        "x-correlation-id",
        "x-envoy-upstream-service-time",
        "x-goog-api-client",
        "x-ms-region",
        "x-ms-request-id",
        "x-request-id",
    }
)

SAFE_HEADER_PREFIXES: Final[tuple[str, ...]] = (
    "x-ratelimit-",
    "anthropic-ratelimit-",
    "x-stainless-",
    "openai-",
    "x-litellm-response-",
    "x-litellm-call-",
    "x-litellm-model-",
    "x-litellm-overhead-",
    "x-litellm-attempted-",
)

SAFE_QUERY_PARAMS: Final[frozenset[str]] = frozenset(
    {
        "api-version",
        "alt",
        "beta",
    }
)

SENSITIVE_QUERY_PARAMS: Final[frozenset[str]] = frozenset(
    {
        "key",
        "api_key",
        "api-key",
        "apikey",
        "token",
        "access_token",
        "sig",
        "signature",
        "auth",
    }
)

SAFE_BODY_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "access_programs",
        "background",
        "caching",
        "drop_params",
        "frequency_penalty",
        "include",
        "logprobs",
        "max_completion_tokens",
        "max_output_tokens",
        "max_tokens",
        "max_tool_calls",
        "modalities",
        "model",
        "n",
        "num_retries",
        "parallel_tool_calls",
        "presence_penalty",
        "reasoning",
        "reasoning_effort",
        "seed",
        "service_tier",
        "store",
        "stream",
        "stream_options",
        "temperature",
        "thinking",
        "timeout",
        "tool_choice",
        "top_k",
        "top_logprobs",
        "top_p",
        "truncation",
        "verbosity",
    }
)

SECRET_BODY_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "api_key",
        "vertex_credentials",
        "aws_secret_access_key",
        "aws_access_key_id",
        "aws_session_token",
        "google_application_credentials",
    }
)

_PEM_BLOCK_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"-----BEGIN [A-Z0-9 ]{0,32}PRIVATE KEY-----[\s\S]*?(?:-----END [A-Z0-9 ]{0,32}PRIVATE KEY-----|$)",
    re.IGNORECASE,
)

_RAW_SECRET_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(?:"
    r"Bearer\s+[A-Za-z0-9._\-+/=]{8,}"
    r"|Basic\s+[A-Za-z0-9+/=]{8,}"
    r"|sk-[A-Za-z0-9._\-]{8,}"
    r"|ya29\.[A-Za-z0-9._\-]{8,}"
    r"|AIza[0-9A-Za-z_\-]{16,}"
    r"|(?:AKIA|ASIA)[0-9A-Z]{16}"
    r"|(?:ghp|gho|ghs|hf|gsk)_[A-Za-z0-9_]{12,}"
    r"|github_pat_[A-Za-z0-9_]{16,}"
    r"|xai-[A-Za-z0-9._\-]{12,}"
    r"|eyJ[A-Za-z0-9_\-]{10,}\.eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}"
    r")",
    re.IGNORECASE,
)

_SAFE_ID_PREFIXES: Final[tuple[str, ...]] = (
    "req_",
    "resp_",
    "chatcmpl-",
    "cmpl-",
    "msg_",
    "proj_",
    "org-",
    "batch_",
    "file-",
    "ftjob-",
    "asst_",
    "thread_",
    "run_",
    "call_",
)

_HEX_OR_B64_SECRET_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\b(?=[A-Za-z0-9+/=_-]{28,}\b)(?=[A-Za-z0-9+/=_-]*[0-9])(?=[A-Za-z0-9+/=_-]*[A-Za-z])[A-Za-z0-9+/=_-]{28,}\b"
)

_SAFE_MODEL_NAME_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9._:/@-]{1,128}$")


def _sanitize_model_name(
    raw_model: str,
    sanitized_model: str | None = None,
    sanitized_model_group: str | None = None,
) -> str:
    fallback = sanitized_model_group or sanitized_model
    if not raw_model:
        return fallback or ""
    if not _SAFE_MODEL_NAME_PATTERN.match(raw_model):
        return fallback if isinstance(fallback, str) and fallback else MASKED_VALUE
    if isinstance(fallback, str) and fallback:
        raw_bare = raw_model.rsplit("/", 1)[-1]
        fallback_bare_model = fallback.rsplit("/", 1)[-1]
        if fallback != raw_model and fallback_bare_model != raw_bare:
            if not _SAFE_MODEL_NAME_PATTERN.match(fallback) or fallback in ("invalid-model", "unknown", "redacted"):
                return fallback
            if sanitized_model and sanitized_model == sanitized_model_group and raw_bare != fallback_bare_model:
                return fallback
    return raw_model


def _summarize_response_format(value: Any) -> dict[str, Any]:
    out: dict[str, Any] = {"bytes": _estimate_size(value)}
    if isinstance(value, Mapping):
        response_format_type = value.get("type")
        if isinstance(response_format_type, str):
            out["type"] = _truncate_str(response_format_type, 32)
        json_string = value.get("json_schema")
        if isinstance(json_string, Mapping) and isinstance(json_string.get("strict"), bool):
            out["strict"] = json_string["strict"]
    return out


def _truncate_str(value: str, max_len: int) -> str:
    if len(value) <= max_len:
        return value
    return value[:max_len] + "...(truncated)"


def _extract_auth_token_secret(raw_value: str) -> str | None:
    s = raw_value.strip()
    if s.lower().startswith("bearer "):
        s = s[7:].strip()
    elif s.lower().startswith("basic "):
        s = s[6:].strip()
    return s if len(s) >= 8 else None


def _collect_runtime_secrets(
    api_key: Any,
    litellm_params: Mapping[str, Any],
    raw_headers: Any = None,
) -> tuple[str, ...]:
    secrets: list[str] = []

    def _add(val: Any) -> None:
        if isinstance(val, str):
            s = val.strip()
            if len(s) >= 8 and s not in secrets:
                secrets.append(s)
                encoded_url = quote(s, safe="")
                if encoded_url != s and encoded_url not in secrets:
                    secrets.append(encoded_url)
                if len(s) <= 256:
                    try:
                        b64 = base64.b64encode(s.encode("utf-8")).decode("ascii")
                        if len(b64) >= 12 and b64 not in secrets:
                            secrets.append(b64)
                    except Exception:
                        pass

    _add(api_key)
    for k in ("api_key", "aws_secret_access_key", "aws_session_token"):
        _add(litellm_params.get(k))

    vertex_credentials = litellm_params.get("vertex_credentials")
    if isinstance(vertex_credentials, str):
        _add(vertex_credentials)
        if vertex_credentials.lstrip().startswith("{"):
            try:
                parsed_vertex_credentials = json.loads(vertex_credentials)
                if isinstance(parsed_vertex_credentials, Mapping):
                    _add(parsed_vertex_credentials.get("private_key"))
                    _add(parsed_vertex_credentials.get("private_key_id"))
            except Exception:
                pass
    elif isinstance(vertex_credentials, Mapping):
        _add(vertex_credentials.get("private_key"))
        _add(vertex_credentials.get("private_key_id"))

    if isinstance(raw_headers, Mapping):
        for header_key, hv in raw_headers.items():
            if str(header_key).lower() in AUTH_HEADER_NAMES and isinstance(hv, str):
                token_value = _extract_auth_token_secret(hv)
                if token_value:
                    _add(token_value)
    return tuple(secrets)


def _get_active_ip_redactor(
    ip_redactor: Callable[[str], str] | None = None,
) -> Callable[[str], str] | None:
    if ip_redactor is not None:
        return ip_redactor
    try:
        from litellm.proxy.spend_tracking.spend_tracking_utils import (
            _masked_ip_text,
            requester_ip_address_logging_disabled,
        )

        if requester_ip_address_logging_disabled() or bool(getattr(litellm, "disable_requester_ip_address_logging", False)):
            return _masked_ip_text
    except Exception:
        return None
    return None


def _scrub_text(
    value: str,
    *,
    runtime_secrets: tuple[str, ...] = (),
    credential_name: str | None = None,
    scrub_high_entropy: bool = False,
    ip_redactor: Callable[[str], str] | None = None,
) -> tuple[str, bool]:
    changed = False
    out = value[:MAX_REGEX_SCAN_LEN] if len(value) > MAX_REGEX_SCAN_LEN else value
    if runtime_secrets:
        repl = f"<credential:{credential_name}>" if credential_name else MASKED_VALUE
        for secret_value in runtime_secrets:
            if secret_value in out:
                out = out.replace(secret_value, repl)
                changed = True
    if "PRIVATE KEY-----" in out.upper():
        new_out = _PEM_BLOCK_PATTERN.sub(MASKED_VALUE, out)
        if new_out != out:
            out = new_out
            changed = True
    if len(out) >= 10 and _RAW_SECRET_PATTERN.search(out):
        out = _RAW_SECRET_PATTERN.sub(MASKED_VALUE, out)
        changed = True
    if scrub_high_entropy and len(out) >= 28:
        def _replace_entropy_match(m: re.Match[str]) -> str:
            token_value = m.group(0)
            if token_value.lower().startswith(_SAFE_ID_PREFIXES):
                return token_value
            return MASKED_VALUE

        new_out = _HEX_OR_B64_SECRET_PATTERN.sub(_replace_entropy_match, out)
        if new_out != out:
            out = new_out
            changed = True
    if ip_redactor is not None and out:
        redacted = ip_redactor(out)
        if redacted != out:
            out = redacted
            changed = True
    return out, changed


def _estimate_size(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, (bytes, bytearray)):
        return len(value)
    if isinstance(value, str):
        return len(value)
    total = 0
    nodes = 0
    stack: list[tuple[Any, int]] = [(value, 0)]
    while stack and nodes < MAX_ESTIMATE_NODES:
        cur, depth = stack.pop()
        nodes += 1
        if cur is None:
            total += 4
        elif isinstance(cur, bool):
            total += 5
        elif isinstance(cur, (int, float)):
            total += 8
        elif isinstance(cur, (bytes, bytearray)):
            total += len(cur)
        elif isinstance(cur, str):
            total += len(cur) + 2
        elif depth >= MAX_ESTIMATE_DEPTH:
            total += 16
        elif isinstance(cur, Mapping):
            total += 2
            for k, v in cur.items():
                total += len(str(k)) + 4
                stack.append((v, depth + 1))
                if len(stack) >= MAX_ESTIMATE_NODES:
                    break
        elif isinstance(cur, (list, tuple)):
            total += 2
            for item in cur:
                stack.append((item, depth + 1))
                if len(stack) >= MAX_ESTIMATE_NODES:
                    break
        else:
            total += len(str(cur))
    return total


def _sanitize_url(
    url: str,
    credential_name: str | None = None,
    runtime_secrets: tuple[str, ...] = (),
    ip_redactor: Callable[[str], str] | None = None,
) -> str:
    if not url:
        return ""
    try:
        parts = urlsplit(url)
        netloc = parts.netloc
        if "@" in netloc:
            _, hostpart = netloc.rsplit("@", 1)
            netloc = f"{MASKED_VALUE}@{hostpart}"
        query = parts.query
        if query:
            sanitized_query: list[tuple[str, str]] = []
            for k, v in parse_qsl(query, keep_blank_values=True):
                lower_key = k.lower()
                if lower_key in SAFE_QUERY_PARAMS:
                    scrubbed_v, _ = _scrub_text(v, runtime_secrets=runtime_secrets, credential_name=credential_name)
                    sanitized_query.append((k, _truncate_str(scrubbed_v, 64)))
                elif lower_key in SENSITIVE_QUERY_PARAMS and credential_name:
                    sanitized_query.append((k, f"<credential:{credential_name}>"))
                else:
                    sanitized_query.append((k, MASKED_VALUE))
            query = urlencode(sanitized_query)
        rebuilt = urlunsplit((parts.scheme, netloc, parts.path, query, ""))
        rebuilt, _ = _scrub_text(rebuilt, runtime_secrets=runtime_secrets, credential_name=credential_name, ip_redactor=ip_redactor)
        return _truncate_str(rebuilt, MAX_HEADER_VALUE_LEN)
    except Exception:
        return MASKED_VALUE


def _is_safe_header(lower_name: str) -> bool:
    if lower_name in SAFE_HEADER_NAMES:
        return True
    return lower_name.startswith(SAFE_HEADER_PREFIXES)


def _format_credential_replacement(raw_value: str, label: str) -> str:
    stripped = raw_value.strip()
    if stripped.lower().startswith("bearer "):
        return f"Bearer <credential:{label}>"
    return f"<credential:{label}>"


def _sanitize_safe_nested(
    value: Any,
    *,
    runtime_secrets: tuple[str, ...] = (),
    credential_name: str | None = None,
    ip_redactor: Callable[[str], str] | None = None,
    depth: int = 0,
) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        scrubbed, _ = _scrub_text(value, runtime_secrets=runtime_secrets, credential_name=credential_name, ip_redactor=ip_redactor)
        return _truncate_str(scrubbed, MAX_BODY_SCALAR_LEN)
    if depth >= MAX_NESTED_SANITIZE_DEPTH:
        return MASKED_VALUE
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for k, v in value.items():
            key_string = str(k)
            out[key_string] = _sanitize_safe_nested(
                v,
                runtime_secrets=runtime_secrets,
                credential_name=credential_name,
                ip_redactor=ip_redactor,
                depth=depth + 1,
            )
        return out
    if isinstance(value, (list, tuple)):
        return [
            _sanitize_safe_nested(
                item,
                runtime_secrets=runtime_secrets,
                credential_name=credential_name,
                ip_redactor=ip_redactor,
                depth=depth + 1,
            )
            for item in value[:16]
        ]
    return _truncate_str(str(value), MAX_BODY_SCALAR_LEN)


def _sanitize_headers(
    raw_headers: Any,
    *,
    credential_label: str | None = None,
    replace_action: str = "replace:credential_name",
    strip_provider_prefix: bool = False,
    runtime_secrets: tuple[str, ...] = (),
    ip_redactor: Callable[[str], str] | None = None,
) -> tuple[dict[str, str], dict[str, str]]:
    if not isinstance(raw_headers, Mapping):
        return {}, {}
    headers_out: dict[str, str] = {}
    process_header: dict[str, str] = {}
    for idx, (k, v) in enumerate(raw_headers.items()):
        if idx >= MAX_HEADER_COUNT:
            break
        key_str = str(k)
        if strip_provider_prefix and key_str.startswith("llm_provider-"):
            key_str = key_str[len("llm_provider-") :]
        lower_key = key_str.lower()
        val_str = "" if v is None else str(v)
        if lower_key in AUTH_HEADER_NAMES:
            if credential_label:
                headers_out[key_str] = _format_credential_replacement(val_str, credential_label)
                process_header[key_str] = replace_action
            else:
                headers_out[key_str] = MASKED_VALUE
                process_header[key_str] = "mask"
        elif _is_safe_header(lower_key):
            scrubbed, was_scrubbed = _scrub_text(
                val_str,
                runtime_secrets=runtime_secrets,
                credential_name=credential_label,
                ip_redactor=ip_redactor,
            )
            headers_out[key_str] = _truncate_str(scrubbed, MAX_HEADER_VALUE_LEN)
            if was_scrubbed:
                process_header[key_str] = "mask"
        else:
            headers_out[key_str] = MASKED_VALUE
            process_header[key_str] = "mask"
    return headers_out, process_header


def _summarize_body(
    body: Any,
    *,
    credential_name: str | None = None,
    runtime_secrets: tuple[str, ...] = (),
) -> tuple[dict[str, Any], dict[str, str], list[str]]:
    if not isinstance(body, Mapping):
        return {}, {}, []
    params: dict[str, Any] = {}
    process_body: dict[str, str] = {}
    raw_keys: list[str] = []
    for k, v in body.items():
        key_str = str(k)
        raw_keys.append(key_str)
        lower_key = key_str.lower()
        if lower_key in SECRET_BODY_FIELDS:
            if credential_name:
                params[key_str] = f"<credential:{credential_name}>"
                process_body[key_str] = "replace:credential_name"
            else:
                params[key_str] = MASKED_VALUE
                process_body[key_str] = "mask"
        elif lower_key == "response_format":
            params[key_str] = _summarize_response_format(v)
        elif lower_key in SAFE_BODY_FIELDS:
            if isinstance(v, (bool, int, float)) or v is None:
                params[key_str] = v
            elif isinstance(v, str):
                scrubbed, was_scrubbed = _scrub_text(
                    v,
                    runtime_secrets=runtime_secrets,
                    credential_name=credential_name,
                )
                params[key_str] = _truncate_str(scrubbed, MAX_BODY_SCALAR_LEN)
                if was_scrubbed:
                    process_body[key_str] = "mask"
            elif isinstance(v, (Mapping, list, tuple)):
                estimated_size = _estimate_size(v)
                if estimated_size <= MAX_BODY_SCALAR_LEN:
                    params[key_str] = _sanitize_safe_nested(
                        v,
                        runtime_secrets=runtime_secrets,
                        credential_name=credential_name,
                    )
                else:
                    params[f"{key_str}_bytes"] = estimated_size
                    if isinstance(v, (list, tuple)):
                        params[f"{key_str}_count"] = len(v)
            else:
                params[f"{key_str}_bytes"] = _estimate_size(v)
        else:
            params[f"{key_str}_bytes"] = _estimate_size(v)
            if isinstance(v, (list, tuple)):
                params[f"{key_str}_count"] = len(v)
    return params, process_body, raw_keys


def _resolve_credential_info(litellm_params: Mapping[str, Any]) -> tuple[str | None, str]:
    internal_meta = litellm_params.get("litellm_metadata")
    if isinstance(internal_meta, Mapping):
        override_name = internal_meta.get("_gateway_credential_name")
        if isinstance(override_name, str) and override_name:
            src = internal_meta.get("_gateway_credential_source")
            return override_name, str(src) if isinstance(src, str) and src else "model_config"
    deployment_credential = litellm_params.get("litellm_credential_name")
    if isinstance(deployment_credential, str) and deployment_credential:
        return deployment_credential, "deployment_default"
    return None, "unspecified"


def capture_external_egress(
    *,
    additional_args: Any,
    model_call_details: Mapping[str, Any],
    api_key: Any = None,
    ip_redactor: Callable[[str], str] | None = None,
) -> dict[str, Any]:
    args = additional_args if isinstance(additional_args, Mapping) else {}
    litellm_params = model_call_details.get("litellm_params")
    litellm_params_value = litellm_params if isinstance(litellm_params, Mapping) else {}

    active_ip_redactor = _get_active_ip_redactor(ip_redactor)
    credential_name, credential_source = _resolve_credential_info(litellm_params_value)
    raw_headers = args.get("headers") or {}
    runtime_secrets = _collect_runtime_secrets(api_key, litellm_params_value, raw_headers=raw_headers)
    headers_out, process_header = _sanitize_headers(
        raw_headers,
        credential_label=credential_name,
        replace_action="replace:credential_name",
        runtime_secrets=runtime_secrets,
        ip_redactor=active_ip_redactor,
    )
    body_params, process_body, raw_body_keys = _summarize_body(
        args.get("complete_input_dict") or {},
        credential_name=credential_name,
        runtime_secrets=runtime_secrets,
    )
    return {
        "api_base": _sanitize_url(
            str(args.get("api_base") or ""),
            credential_name=credential_name,
            runtime_secrets=runtime_secrets,
            ip_redactor=active_ip_redactor,
        ),
        "credential_name": credential_name,
        "credential_source": credential_source,
        "headers": headers_out,
        "body_params": body_params,
        "_raw_body_keys": raw_body_keys,
        "_process_header": process_header,
        "_process_body": process_body,
    }


def _extract_response_headers(
    kwargs: Mapping[str, Any],
    metadata: Mapping[str, Any],
    standard_logging_payload: Mapping[str, Any] | None,
    response_obj: object = None,
) -> Mapping[str, Any]:
    candidates: list[Any] = [kwargs.get("response_headers"), metadata.get("response_headers")]
    if response_obj is not None:
        hidden = getattr(response_obj, "_hidden_params", None)
        if isinstance(hidden, Mapping):
            candidates.append(hidden.get("additional_headers"))
    if isinstance(standard_logging_payload, Mapping):
        logging_metadata = standard_logging_payload.get("metadata")
        if isinstance(logging_metadata, Mapping):
            candidates.append(logging_metadata.get("response_headers"))
    for c in candidates:
        if isinstance(c, Mapping) and len(c) > 0:
            return c
    return {}


def build_gateway_io_metadata(
    *,
    kwargs: Mapping[str, Any],
    metadata: Mapping[str, Any],
    litellm_params: Mapping[str, Any],
    response_obj_dict: Mapping[str, Any],
    standard_logging_payload: Mapping[str, Any] | None,
    custom_llm_provider: str | None,
    api_base: str | None,
    response_obj: object = None,
    sanitized_error_info: Mapping[str, Any] | None = None,
    sanitized_model: str | None = None,
    sanitized_model_group: str | None = None,
    ip_redactor: Callable[[str], str] | None = None,
) -> dict[str, Any]:
    proxy_server_request = litellm_params.get("proxy_server_request")
    proxy_server_request_map = proxy_server_request if isinstance(proxy_server_request, Mapping) else {}
    proxy_server_request_body = proxy_server_request_map.get("body")
    proxy_server_request_body_map = proxy_server_request_body if isinstance(proxy_server_request_body, Mapping) else {}
    proxy_server_request_headers = proxy_server_request_map.get("headers")

    raw_alias = metadata.get("user_api_key_alias")
    key_label = (
        raw_alias
        if isinstance(raw_alias, str) and raw_alias and not raw_alias.lower().startswith("sk-")
        else None
    )
    in_secrets = _collect_runtime_secrets(None, {}, raw_headers=proxy_server_request_headers)
    in_headers, internal_process_header = _sanitize_headers(
        proxy_server_request_headers or {},
        credential_label=key_label,
        replace_action="replace:key_alias",
        runtime_secrets=in_secrets,
        ip_redactor=ip_redactor,
    )
    raw_route = str(proxy_server_request_map.get("url") or metadata.get("user_api_key_request_route") or "")
    raw_in_model = str(proxy_server_request_body_map.get("model") or metadata.get("model_group") or "")
    internal_ingress: dict[str, Any] = {
        "route": _sanitize_url(
            raw_route,
            credential_name=key_label,
            runtime_secrets=in_secrets,
            ip_redactor=ip_redactor,
        ),
        "model": _sanitize_model_name(
            raw_in_model,
            sanitized_model=sanitized_model,
            sanitized_model_group=sanitized_model_group,
        ),
        "headers": in_headers,
        "body_bytes": _estimate_size(proxy_server_request_body),
    }
    if internal_process_header:
        internal_ingress["litellm_process"] = {"header": internal_process_header}

    exc = kwargs.get("exception")
    is_failure = (
        metadata.get("status") == "failure"
        or exc is not None
        or isinstance(response_obj, Exception)
    )
    status_str = "failure" if is_failure else "success"
    http_status: int | None = None
    err_info = metadata.get("error_information")
    err_map = err_info if isinstance(err_info, Mapping) else {}
    if is_failure:
        raw_code = err_map.get("error_code")
        if raw_code is None and exc is not None:
            raw_code = getattr(exc, "status_code", None)
        if raw_code is None and isinstance(response_obj, Exception):
            raw_code = getattr(response_obj, "status_code", None)
        try:
            http_status = int(raw_code) if raw_code is not None and str(raw_code).isdigit() else None
        except Exception:
            http_status = None
    else:
        http_status = 200

    internal_egress: dict[str, Any] = {
        "status": status_str,
        "http_status": http_status,
        "body_bytes": _estimate_size(response_obj_dict if response_obj_dict else response_obj),
    }

    runtime_secrets = _collect_runtime_secrets(kwargs.get("api_key"), litellm_params)
    captured = kwargs.get("_gateway_external_egress")
    if isinstance(captured, Mapping):
        external_headers = dict(captured.get("headers") or {})
        external_process_header = dict(captured.get("_process_header") or {})
        external_body_params = dict(captured.get("body_params") or {})
        external_process_body = dict(captured.get("_process_body") or {})
        external_raw_keys = list(captured.get("_raw_body_keys") or [])
        external_api_base = str(captured.get("api_base") or "")
        credential_name = captured.get("credential_name")
        credential_source = str(captured.get("credential_source") or "unspecified")
    else:
        credential_name, credential_source = _resolve_credential_info(litellm_params)
        external_headers = {}
        external_process_header = {}
        external_body_params = {}
        external_process_body = {}
        external_raw_keys = []
        external_api_base = _sanitize_url(
            str(api_base or ""),
            credential_name=credential_name,
            runtime_secrets=runtime_secrets,
        )

    in_header_lower = {k.lower() for k in in_headers}
    for header_key in external_headers:
        if header_key.lower() not in in_header_lower and header_key not in external_process_header:
            external_process_header[header_key] = "add"

    if isinstance(external_body_params.get("model"), str):
        external_body_params["model"] = _sanitize_model_name(
            external_body_params["model"],
            sanitized_model=sanitized_model,
            sanitized_model_group=sanitized_model_group,
        )
    if ip_redactor is not None:
        if external_api_base:
            external_api_base = ip_redactor(external_api_base)
        for header_key, hv in list(external_headers.items()):
            redacted_header_value = ip_redactor(hv)
            if redacted_header_value != hv:
                external_headers[header_key] = redacted_header_value
                external_process_header[header_key] = "mask"

    in_body_keys = {str(k) for k in proxy_server_request_body_map}
    if in_body_keys:
        for body_key in external_raw_keys:
            if body_key not in in_body_keys and body_key not in external_process_body:
                external_process_body[body_key] = "add"
        in_model = internal_ingress.get("model")
        external_model = external_body_params.get("model")
        if isinstance(in_model, str) and isinstance(external_model, str) and in_model != external_model:
            external_process_body["model"] = "replace:model"

    external_egress: dict[str, Any] = {
        "provider": str(custom_llm_provider or ""),
        "api_base": external_api_base,
        "credential_name": credential_name,
        "credential_source": credential_source,
        "headers": external_headers,
        "body_params": external_body_params,
    }
    external_egress_process: dict[str, Any] = {}
    if external_process_header:
        external_egress_process["header"] = external_process_header
    if external_process_body:
        external_egress_process["body"] = external_process_body
    if external_egress_process:
        external_egress["litellm_process"] = external_egress_process

    raw_response_headers = _extract_response_headers(
        kwargs=kwargs,
        metadata=metadata,
        standard_logging_payload=standard_logging_payload,
        response_obj=response_obj,
    )
    resp_headers, response_process_header = _sanitize_headers(
        raw_response_headers,
        credential_label=None,
        strip_provider_prefix=True,
        runtime_secrets=runtime_secrets,
        ip_redactor=ip_redactor,
    )
    external_ingress: dict[str, Any] = {
        "status": status_str,
        "http_status": http_status,
        "headers": resp_headers,
    }
    response_model = response_obj_dict.get("model") if response_obj_dict else getattr(response_obj, "model", None)
    if isinstance(response_model, str):
        external_ingress["model"] = _sanitize_model_name(
            response_model,
            sanitized_model=sanitized_model,
            sanitized_model_group=sanitized_model_group,
        )
    response_access_programs = (
        response_obj_dict.get("access_programs")
        if response_obj_dict
        else getattr(response_obj, "access_programs", None)
    )
    if response_access_programs is not None:
        external_ingress["access_programs"] = _sanitize_safe_nested(
            response_access_programs,
            runtime_secrets=runtime_secrets,
            credential_name=credential_name,
            ip_redactor=ip_redactor,
        )
    if is_failure:
        sanitized_error = sanitized_error_info if isinstance(sanitized_error_info, Mapping) else {}
        err_cls = (
            sanitized_error.get("error_class")
            or err_map.get("error_class")
            or (type(exc).__name__ if exc is not None else None)
        )
        if err_cls:
            external_ingress["error_type"] = str(err_cls)
        err_msg = sanitized_error.get("error_message")
        if isinstance(err_msg, str) and err_msg:
            scrubbed_err, _ = _scrub_text(
                err_msg,
                runtime_secrets=runtime_secrets,
                credential_name=credential_name,
                scrub_high_entropy=True,
                ip_redactor=ip_redactor,
            )
            external_ingress["error_message"] = _truncate_str(scrubbed_err, MAX_ERROR_MSG_LEN)
    if response_process_header:
        external_ingress["litellm_process"] = {"header": response_process_header}

    return {
        "internal": {
            "ingress": internal_ingress,
            "egress": internal_egress,
        },
        "external": {
            "egress": external_egress,
            "ingress": external_ingress,
        },
    }


def attach_gateway_io_to_spend_log_payload(
    payload: Any,
    *,
    kwargs: Mapping[str, Any] | None,
    response_obj: object,
    ip_redactor: Callable[[str], str] | None = None,
) -> Any:
    if not isinstance(payload, dict):
        return payload
    keyword_args = kwargs if isinstance(kwargs, Mapping) else {}
    litellm_params_value = keyword_args.get("litellm_params")
    litellm_params_map = litellm_params_value if isinstance(litellm_params_value, Mapping) else {}
    meta = litellm_params_map.get("litellm_metadata")
    if not isinstance(meta, Mapping):
        meta = litellm_params_map.get("metadata")
    if not isinstance(meta, Mapping):
        meta = keyword_args.get("metadata")
    meta_map = meta if isinstance(meta, Mapping) else {}
    response_dict: Mapping[str, Any] = response_obj if isinstance(response_obj, Mapping) else {}
    logging_payload = keyword_args.get("standard_logging_object")
    logging_payload_map = logging_payload if isinstance(logging_payload, Mapping) else None
    raw_meta = payload.get("metadata")
    parsed_meta: dict[str, Any] | None = None
    if isinstance(raw_meta, str):
        try:
            loaded = json.loads(raw_meta)
            parsed_meta = loaded if isinstance(loaded, dict) else {}
        except Exception:
            parsed_meta = {}
    elif isinstance(raw_meta, dict):
        parsed_meta = raw_meta
    sanitized_err = parsed_meta.get("error_information") if isinstance(parsed_meta, dict) else None
    gateway_io = build_gateway_io_metadata(
        kwargs=keyword_args,
        metadata=meta_map,
        litellm_params=litellm_params_map,
        response_obj_dict=response_dict,
        standard_logging_payload=logging_payload_map,
        custom_llm_provider=str(payload.get("custom_llm_provider") or litellm_params_map.get("custom_llm_provider") or ""),
        api_base=str(payload.get("api_base") or ""),
        response_obj=response_obj,
        sanitized_error_info=sanitized_err if isinstance(sanitized_err, Mapping) else None,
        sanitized_model=str(payload["model"]) if isinstance(payload.get("model"), str) else None,
        sanitized_model_group=str(payload["model_group"]) if isinstance(payload.get("model_group"), str) else None,
        ip_redactor=ip_redactor,
    )
    if isinstance(parsed_meta, dict):
        parsed_meta.pop("_gateway_credential_name", None)
        parsed_meta.pop("_gateway_credential_source", None)
        parsed_meta["gateway_io"] = gateway_io
        if isinstance(raw_meta, str):
            payload["metadata"] = json.dumps(parsed_meta, default=str, ensure_ascii=False)
    return payload
