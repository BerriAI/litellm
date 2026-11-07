# SPDX-License-Identifier: MIT
"""LiteLLM CustomLogger integration for Silicium AI Ops (`silicium-ai-ops` == `1.0.2`).

Delivers four capabilities in a single in-process C-ABI pass for LiteLLM Proxy (`v1.108+`):
  1. In-Process HNSW Semantic Cache (`< 50 us`, `369,836 QPS` SIFT-1M, `4.0x` SQ8 RAM downsizing, zero Redis/Qdrant hops).
  2. 1-Pass KV-Cache Prefix Affinity Routing (`64`-bit FNV-1a `x-silicium-kv-prefix-hash`, `x-silicium-worker-id`, `estimated_tokens`).
  3. Zero-Malloc SIMD JSON Streaming (`7.58-9.42 GB/s` 1-core, `276-390 GB/s` multi-core, `0.46 cycles/byte`).
  4. 1-Pass 12-Flag SWAR PII & Secret Guardrails (evaluated in the same SIMD register sweep at zero extra memory pass).
"""

from __future__ import annotations

import contextlib
import importlib
from collections.abc import Mapping, Sequence
from typing import Final, NoReturn, Protocol

PINNED_SILICIUM_AI_OPS_VERSION: Final = "1.0.2"
_MAX_TENANT_SCOPES: Final = 256
_TENANT_META_KEYS: Final = (
    "user_api_key_org_id",
    "user_api_key_team_id",
    "user_api_key_user_id",
    "user_api_key_hash",
    "user_api_key",
)
_SAFE_LOGGING_KEYS: Final = (
    "model",
    "messages",
    "prompt",
    "call_type",
    "cache_hit",
    "custom_llm_provider",
)
_SUPPORTED_TEXT_BLOCK_TYPES: Final = ("text", "input_text", "output_text")
_ALLOWED_MESSAGE_KEYS: Final = ("role", "content", "name", "tool_call_id")
_CONTENT_CARRYING_KEYS: Final = ("messages", "prompt", "input", "instructions")


class _SiliciumGatewayProtocol(Protocol):
    def __call__(self, *args: object) -> object: ...

    def pre_call_hook(
        self,
        request_payload: object = None,
        raise_on_block: bool = False,
        data: object = None,
        call_type: object = None,
    ) -> Mapping[str, object]: ...

    async def async_pre_call_hook(
        self,
        request_payload: object = None,
        raise_on_block: bool = False,
        data: object = None,
        call_type: object = None,
    ) -> Mapping[str, object]: ...

    def post_call_hook(
        self,
        request_payload: object,
        response_obj: object = None,
        prompt: str = "",
        end_time: object = None,
    ) -> Mapping[str, object]: ...

    async def async_log_success_event(
        self,
        kwargs: object,
        response_obj: object = None,
        prompt: str = "",
        end_time: object = None,
    ) -> Mapping[str, object]: ...


def _resolve_custom_logger_base() -> type:
    with contextlib.suppress(Exception):
        mod: Final = importlib.import_module("litellm.integrations.custom_logger")
        custom_cls: Final = getattr(mod, "CustomLogger", object)
        if isinstance(custom_cls, type):
            return custom_cls
    return object


def _raise_unsupported_content(field_name: str) -> NoReturn:
    raise PermissionError(
        f"SiliciumAiOpsLogger rejected request: unsupported `{field_name}` shape for PII masking."
    )


def _extract_mapping_parts(src: Mapping[str, object]) -> tuple[str, ...]:
    return tuple(
        f"{key}={val.strip()}"
        for key in _TENANT_META_KEYS
        if isinstance((val := src.get(key)), str) and val.strip()
    )


def _extract_model_part(data: object) -> str | None:
    if not isinstance(data, Mapping):
        return None
    direct_model: Final = data.get("model")
    if isinstance(direct_model, str) and direct_model.strip():
        return f"model={direct_model.strip()}"
    std_obj: Final = data.get("standard_logging_object")
    if isinstance(std_obj, Mapping):
        std_model: Final = std_obj.get("model")
        if isinstance(std_model, str) and std_model.strip():
            return f"model={std_model.strip()}"
    return None


def _extract_tenant_scope(
    explicit_scope: str | None,
    user_api_key_dict: object,
    data: object,
    is_proxy: bool = False,
) -> str | None:
    clean_explicit: Final = (
        explicit_scope.strip() if isinstance(explicit_scope, str) and explicit_scope.strip() else None
    )
    key_parts: Final[tuple[str, ...]] = (
        _extract_mapping_parts(user_api_key_dict) if isinstance(user_api_key_dict, Mapping) else ()
    )
    meta: Final = data.get("metadata") if isinstance(data, Mapping) else None
    meta_parts: Final[tuple[str, ...]] = (
        _extract_mapping_parts(meta) if isinstance(meta, Mapping) else ()
    )
    std_obj: Final = data.get("standard_logging_object") if isinstance(data, Mapping) else None
    std_parts: Final[tuple[str, ...]] = (
        _extract_mapping_parts(std_obj) if isinstance(std_obj, Mapping) else ()
    )
    litellm_params: Final = data.get("litellm_params") if isinstance(data, Mapping) else None
    lp_meta: Final = litellm_params.get("metadata") if isinstance(litellm_params, Mapping) else None
    lp_parts: Final[tuple[str, ...]] = (
        _extract_mapping_parts(lp_meta) if isinstance(lp_meta, Mapping) else ()
    )
    attr_parts: Final[tuple[str, ...]] = (
        tuple(
            f"{attr}={attr_val.strip()}"
            for attr in ("org_id", "team_id", "user_id", "token")
            if isinstance((attr_val := getattr(user_api_key_dict, attr, None)), str) and attr_val.strip()
        )
        if user_api_key_dict is not None and not isinstance(user_api_key_dict, Mapping)
        else ()
    )
    req_parts: Final[tuple[str, ...]] = (*key_parts, *meta_parts, *std_parts, *lp_parts, *attr_parts)
    if is_proxy and not req_parts:
        return None
    scope_prefix: Final[tuple[str, ...]] = (f"scope={clean_explicit}",) if clean_explicit else ()
    identity_parts: Final[tuple[str, ...]] = (*scope_prefix, *req_parts)
    if not identity_parts:
        return None
    model_part: Final = _extract_model_part(data)
    all_parts: Final[tuple[str, ...]] = (
        (*identity_parts, model_part) if model_part is not None else identity_parts
    )
    return "|".join(all_parts)


def _sanitize_logging_payload(
    kwargs: object,
    tenant_scope: str | None,
) -> Mapping[str, object]:
    std_obj: Final = kwargs.get("standard_logging_object") if isinstance(kwargs, Mapping) else None
    src: Final[Mapping[str, object]] = (
        std_obj
        if isinstance(std_obj, Mapping)
        else (kwargs if isinstance(kwargs, Mapping) else {})
    )
    base_pairs: Final[Mapping[str, object]] = {
        key: src[key] for key in _SAFE_LOGGING_KEYS if key in src
    }
    if tenant_scope:
        return {**base_pairs, "tenant_scope": tenant_scope}
    return base_pairs


def _extract_redacted_text(res: Mapping[str, object], fallback: str) -> str:
    msgs: Final = res.get("messages")
    if isinstance(msgs, Sequence) and not isinstance(msgs, (str, bytes)) and len(msgs) > 0:
        first_msg: Final = msgs[0]
        if isinstance(first_msg, Mapping):
            content: Final = first_msg.get("content")
            if isinstance(content, str):
                return content
    prompt_val: Final = res.get("prompt")
    if isinstance(prompt_val, str):
        return prompt_val
    sanitized_val: Final = res.get("sanitized_payload_str")
    if isinstance(sanitized_val, str):
        return sanitized_val
    return fallback


def _redact_string_leaf(delegate: _SiliciumGatewayProtocol, text: str) -> str:
    probe_data: Final[Mapping[str, object]] = {
        "model": "silicium-redact-leaf",
        "messages": [{"role": "user", "content": text}],
        "prompt": text,
    }
    res: Final = delegate.pre_call_hook(
        request_payload=probe_data,
        raise_on_block=False,
        data=probe_data,
        call_type="completion",
    )
    return _extract_redacted_text(res, text)


def _redact_content_block(
    delegate: _SiliciumGatewayProtocol,
    block: object,
) -> Mapping[str, object]:
    if not isinstance(block, Mapping):
        _raise_unsupported_content("content_block")
    block_type: Final = block.get("type")
    text_val: Final = block.get("text")
    extra_keys: Final[tuple[str, ...]] = tuple(
        str(k) for k in block if k not in ("type", "text")
    )
    if block_type in _SUPPORTED_TEXT_BLOCK_TYPES and isinstance(text_val, str) and not extra_keys:
        return {**block, "text": _redact_string_leaf(delegate, text_val)}
    _raise_unsupported_content("content_block")


def _redact_message_item(
    delegate: _SiliciumGatewayProtocol,
    msg: object,
) -> Mapping[str, object]:
    if not isinstance(msg, Mapping):
        _raise_unsupported_content("messages")
    unexpected_keys: Final[tuple[str, ...]] = tuple(
        str(k) for k in msg if k not in _ALLOWED_MESSAGE_KEYS
    )
    if unexpected_keys:
        _raise_unsupported_content("messages")
    content: Final = msg.get("content")
    if isinstance(content, str):
        return {**msg, "content": _redact_string_leaf(delegate, content)}
    if isinstance(content, Sequence) and not isinstance(content, (str, bytes)):
        redacted_blocks: Final[Sequence[Mapping[str, object]]] = [
            _redact_content_block(delegate, blk) for blk in content
        ]
        return {**msg, "content": redacted_blocks}
    _raise_unsupported_content("messages")


def _redact_input_item(
    delegate: _SiliciumGatewayProtocol,
    item: object,
) -> object:
    if isinstance(item, str):
        return _redact_string_leaf(delegate, item)
    if isinstance(item, Mapping):
        if "content" in item:
            return _redact_message_item(delegate, item)
        return _redact_content_block(delegate, item)
    _raise_unsupported_content("input")


def _redact_input_field(
    delegate: _SiliciumGatewayProtocol,
    input_val: object,
) -> object:
    if isinstance(input_val, str):
        return _redact_string_leaf(delegate, input_val)
    if isinstance(input_val, Sequence) and not isinstance(input_val, (str, bytes)):
        return [_redact_input_item(delegate, item) for item in input_val]
    _raise_unsupported_content("input")


def _redact_prompt_field(
    delegate: _SiliciumGatewayProtocol,
    prompt_val: object,
) -> object:
    if isinstance(prompt_val, str):
        return _redact_string_leaf(delegate, prompt_val)
    if (
        isinstance(prompt_val, Sequence)
        and not isinstance(prompt_val, (str, bytes))
        and all(isinstance(item, str) for item in prompt_val)
    ):
        return [_redact_string_leaf(delegate, str(item)) for item in prompt_val]
    _raise_unsupported_content("prompt")


def _extract_pii_flags(res: Mapping[str, object]) -> int:
    res_meta: Final = res.get("metadata")
    insp_raw: Final = (
        res_meta.get("silicium_inspection")
        if isinstance(res_meta, Mapping) and "silicium_inspection" in res_meta
        else res
    )
    flags_val: Final = (
        insp_raw.get("pii_flags", res.get("pii_flags", 0))
        if isinstance(insp_raw, Mapping)
        else res.get("pii_flags", 0)
    )
    return flags_val if isinstance(flags_val, int) else 0


def _apply_structured_redaction(
    delegate: _SiliciumGatewayProtocol,
    payload: Mapping[str, object],
) -> Mapping[str, object]:
    has_supported_target: Final = any(k in payload for k in _CONTENT_CARRYING_KEYS)
    if not has_supported_target:
        _raise_unsupported_content("payload")
    msgs_val: Final = payload.get("messages")
    msgs_patch: Final[Mapping[str, object]] = (
        {"messages": [_redact_message_item(delegate, m) for m in msgs_val]}
        if isinstance(msgs_val, Sequence) and not isinstance(msgs_val, (str, bytes))
        else ({} if msgs_val is None else _raise_unsupported_content("messages"))
    )
    prompt_val: Final = payload.get("prompt")
    prompt_patch: Final[Mapping[str, object]] = (
        {"prompt": _redact_prompt_field(delegate, prompt_val)}
        if prompt_val is not None
        else {}
    )
    input_val: Final = payload.get("input")
    input_patch: Final[Mapping[str, object]] = (
        {"input": _redact_input_field(delegate, input_val)}
        if input_val is not None
        else {}
    )
    instr_val: Final = payload.get("instructions")
    instr_patch: Final[Mapping[str, object]] = (
        {"instructions": _redact_string_leaf(delegate, instr_val)}
        if isinstance(instr_val, str)
        else ({} if instr_val is None else _raise_unsupported_content("instructions"))
    )
    redacted: Final[Mapping[str, object]] = {
        **payload,
        **msgs_patch,
        **prompt_patch,
        **input_patch,
        **instr_patch,
    }
    verify_body: Final[Mapping[str, object]] = {
        k: v for k, v in redacted.items() if k != "metadata"
    }
    verify_res: Final = delegate.pre_call_hook(
        request_payload=verify_body,
        raise_on_block=False,
        data=verify_body,
        call_type="completion",
    )
    if _extract_pii_flags(verify_res) != 0:
        raise PermissionError(
            "SiliciumAiOpsLogger blocked request: unredacted PII remained after structured content masking."
        )
    return redacted


class SiliciumAiOpsLogger(_resolve_custom_logger_base()):
    """LiteLLM `CustomLogger` callback delegating to `silicium_ai_ops.SiliciumLiteLLMGateway`."""

    enforces_request_content: bool = True

    def __init__(
        self,
        num_workers: int = 8,
        max_tokens: int = 262144,
        pii_policy: str = "block",
        enable_semantic_cache: bool = False,
        tenant_scope: str | None = None,
        cache_dim: int = 8,
        cache_distance_threshold: float = 0.15,
        residency_zone: str = "on-prem-sovereign",
        delegate: _SiliciumGatewayProtocol | None = None,
    ) -> None:
        with contextlib.suppress(Exception):
            super().__init__()
        self.enforces_request_content = True
        eff_workers: Final = max(1, int(num_workers))
        eff_tokens: Final = max(1, int(max_tokens))
        eff_policy: Final = str(pii_policy).strip().lower() or "block"
        eff_cache: Final = bool(enable_semantic_cache)
        eff_scope: Final = tenant_scope.strip() if isinstance(tenant_scope, str) and tenant_scope.strip() else None
        if eff_cache and eff_scope is None:
            raise ValueError(
                "SiliciumAiOpsLogger requires a non-empty `tenant_scope` when `enable_semantic_cache=True`."
            )
        eff_dim: Final = max(1, int(cache_dim))
        eff_thresh: Final = float(cache_distance_threshold)
        eff_zone: Final = str(residency_zone)
        self.num_workers: int = eff_workers
        self.max_tokens: int = eff_tokens
        self.pii_policy: str = eff_policy
        self.enable_semantic_cache: bool = eff_cache
        self.tenant_scope: str | None = eff_scope
        self.cache_dim: int = eff_dim
        self.cache_distance_threshold: float = eff_thresh
        self.residency_zone: str = eff_zone
        self._delegate: _SiliciumGatewayProtocol | None = delegate
        self._unscoped_delegate: _SiliciumGatewayProtocol | None = None
        self._tenant_delegates: Mapping[str, _SiliciumGatewayProtocol] = {}
        self._populated_tenants: frozenset[str] = frozenset()

    def _instantiate_native_gateway(self, cache_enabled: bool) -> _SiliciumGatewayProtocol:
        with contextlib.suppress(ImportError, AttributeError, TypeError, ValueError, RuntimeError):
            mod: Final = importlib.import_module("silicium_ai_ops")
            ver: Final = str(getattr(mod, "__version__", PINNED_SILICIUM_AI_OPS_VERSION))
            gw_cls: Final = getattr(mod, "SiliciumLiteLLMGateway", None)
            if gw_cls is not None and ver.startswith("1.0."):
                instance: Final[_SiliciumGatewayProtocol] = gw_cls(
                    num_workers=self.num_workers,
                    max_tokens=self.max_tokens,
                    pii_policy=self.pii_policy,
                    enable_semantic_cache=cache_enabled,
                    cache_dim=self.cache_dim,
                    cache_distance_threshold=self.cache_distance_threshold,
                    residency_zone=self.residency_zone,
                )
                return instance
        raise RuntimeError(
            f"SiliciumAiOpsLogger requires `silicium-ai-ops=={PINNED_SILICIUM_AI_OPS_VERSION}` to be installed."
        )

    def _can_retain_tenant(self, tenant_key: str) -> bool:
        if self._delegate is not None:
            return (
                tenant_key in self._populated_tenants
                or len(self._populated_tenants) < _MAX_TENANT_SCOPES
            )
        return (
            tenant_key in self._tenant_delegates
            or len(self._tenant_delegates) < _MAX_TENANT_SCOPES
        )

    def _record_populated_tenant(self, tenant_key: str) -> None:
        if tenant_key in self._populated_tenants:
            return
        if self._delegate is None and tenant_key not in self._tenant_delegates:
            return
        if len(self._populated_tenants) < _MAX_TENANT_SCOPES:
            self._populated_tenants = frozenset((*self._populated_tenants, tenant_key))

    def _get_delegate(self, tenant_key: str | None = None) -> _SiliciumGatewayProtocol:
        if self._delegate is not None:
            return self._delegate
        if (
            not self.enable_semantic_cache
            or tenant_key is None
            or not self._can_retain_tenant(tenant_key)
        ):
            if self._unscoped_delegate is not None:
                return self._unscoped_delegate
            unscoped: Final = self._instantiate_native_gateway(cache_enabled=False)
            self._unscoped_delegate = unscoped
            return unscoped
        existing: Final = self._tenant_delegates.get(tenant_key)
        if existing is not None:
            return existing
        scoped: Final = self._instantiate_native_gateway(cache_enabled=True)
        self._tenant_delegates = {**self._tenant_delegates, tenant_key: scoped}
        return scoped

    def _should_raise_on_block(self, cache: object, is_proxy_call: bool) -> bool:
        if isinstance(cache, bool):
            return cache or (self.pii_policy == "block" and is_proxy_call)
        return self.pii_policy == "block" and is_proxy_call

    def _finalize_pre_call_result(
        self,
        res: Mapping[str, object],
        user_api_key_dict: object,
        data: object,
        call_type: object,
        tenant_key: str | None,
        raise_on_block: bool,
        delegate: _SiliciumGatewayProtocol,
    ) -> Mapping[str, object]:
        res_meta: Final = res.get("metadata")
        insp_raw: Final = (
            res_meta.get("silicium_inspection")
            if isinstance(res_meta, Mapping) and "silicium_inspection" in res_meta
            else res
        )
        base_insp: Final[Mapping[str, object]] = dict(insp_raw) if isinstance(insp_raw, Mapping) else {}
        allowed: Final = base_insp.get("allowed", res.get("allowed", True))
        pii_flags: Final = _extract_pii_flags(res)
        if allowed is False or (self.pii_policy == "block" and pii_flags != 0 and raise_on_block):
            raise PermissionError("SiliciumAiOpsLogger blocked request due to PII/secret policy.")
        if pii_flags != 0 and self.pii_policy not in ("mask", "monitor") and raise_on_block:
            raise PermissionError("SiliciumAiOpsLogger blocked request due to PII/secret policy.")

        cache_permitted: Final = bool(
            self.enable_semantic_cache
            and tenant_key is not None
            and tenant_key in self._populated_tenants
        )
        insp_dict: Final[Mapping[str, object]] = (
            {**base_insp, "tenant_scope": tenant_key}
            if cache_permitted
            else (
                {
                    **{k: v for k, v in base_insp.items() if k != "cached_response"},
                    "cache_hit": False,
                    "tenant_scope": tenant_key,
                }
                if self.enable_semantic_cache and tenant_key is not None
                else {
                    **{k: v for k, v in base_insp.items() if k != "cached_response"},
                    "cache_hit": False,
                }
            )
        )

        if isinstance(data, Mapping) and call_type is not None:
            effective_data: Final[Mapping[str, object]] = (
                _apply_structured_redaction(delegate, data)
                if pii_flags != 0 and self.pii_policy == "mask"
                else data
            )
            orig_meta: Final = data.get("metadata")
            base_meta: Final[Mapping[str, object]] = (
                dict(orig_meta) if isinstance(orig_meta, Mapping) else {}
            )
            merged_meta: Final[Mapping[str, object]] = {
                **base_meta,
                "silicium_inspection": insp_dict,
            }
            return {
                **effective_data,
                "metadata": merged_meta,
            }

        redacted_1arg: Final[Mapping[str, object]] = (
            _apply_structured_redaction(delegate, user_api_key_dict)
            if pii_flags != 0 and self.pii_policy == "mask" and isinstance(user_api_key_dict, Mapping)
            else {}
        )
        if pii_flags != 0 and self.pii_policy == "mask" and not isinstance(user_api_key_dict, Mapping):
            if isinstance(user_api_key_dict, str):
                redacted_str: Final = _redact_string_leaf(delegate, user_api_key_dict)
                verify_str_res: Final = delegate.pre_call_hook(
                    request_payload=redacted_str,
                    raise_on_block=False,
                )
                if _extract_pii_flags(verify_str_res) != 0:
                    raise PermissionError("SiliciumAiOpsLogger blocked request: unredacted string PII.")
            else:
                _raise_unsupported_content("request_payload")

        base_1arg: Final[Mapping[str, object]] = {**res, **redacted_1arg}
        if cache_permitted:
            return {**base_1arg, "tenant_scope": tenant_key}
        if self.enable_semantic_cache and tenant_key is not None:
            return {
                **{k: v for k, v in base_1arg.items() if k != "cached_response"},
                "cache_hit": False,
                "tenant_scope": tenant_key,
            }
        return {
            **{k: v for k, v in base_1arg.items() if k != "cached_response"},
            "cache_hit": False,
        }

    def __call__(self, *args: object) -> object:
        if not args:
            return self
        delegate: Final = self._get_delegate(None)
        return delegate(*args)

    def pre_call_hook(
        self,
        user_api_key_dict: object = None,
        cache: object = None,
        data: object = None,
        call_type: object = None,
    ) -> Mapping[str, object]:
        is_proxy: Final = data is not None or call_type is not None
        eff_target: Final = data if is_proxy else user_api_key_dict
        tenant_key: Final = _extract_tenant_scope(
            self.tenant_scope, user_api_key_dict, eff_target, is_proxy=is_proxy
        )
        delegate: Final = self._get_delegate(tenant_key)
        raise_block: Final = self._should_raise_on_block(cache, is_proxy)
        raw_res: Final = delegate.pre_call_hook(
            request_payload=user_api_key_dict,
            raise_on_block=raise_block,
            data=data,
            call_type=call_type,
        )
        return self._finalize_pre_call_result(
            raw_res, user_api_key_dict, data, call_type, tenant_key, raise_block, delegate
        )

    async def async_pre_call_hook(
        self,
        user_api_key_dict: object = None,
        cache: object = None,
        data: object = None,
        call_type: object = None,
    ) -> Mapping[str, object]:
        is_proxy: Final = data is not None or call_type is not None
        eff_target: Final = data if is_proxy else user_api_key_dict
        tenant_key: Final = _extract_tenant_scope(
            self.tenant_scope, user_api_key_dict, eff_target, is_proxy=is_proxy
        )
        delegate: Final = self._get_delegate(tenant_key)
        raise_block: Final = self._should_raise_on_block(cache, is_proxy)
        raw_res: Final = await delegate.async_pre_call_hook(
            request_payload=user_api_key_dict,
            raise_on_block=raise_block,
            data=data,
            call_type=call_type,
        )
        return self._finalize_pre_call_result(
            raw_res, user_api_key_dict, data, call_type, tenant_key, raise_block, delegate
        )

    def log_success_event(
        self,
        kwargs: object,
        response_obj: object = None,
        start_time: object = "",
        end_time: object = None,
    ) -> Mapping[str, object]:
        is_proxy_log: Final = isinstance(kwargs, Mapping) and (
            "litellm_params" in kwargs or "standard_logging_object" in kwargs
        )
        tenant_key: Final = _extract_tenant_scope(
            self.tenant_scope, None, kwargs, is_proxy=is_proxy_log
        )
        if not (
            self.enable_semantic_cache
            and tenant_key is not None
            and self._can_retain_tenant(tenant_key)
        ):
            return {"cached": False, "reason": "semantic_cache_disabled_or_unscoped"}
        delegate: Final = self._get_delegate(tenant_key)
        safe_payload: Final = _sanitize_logging_payload(kwargs, tenant_key)
        eff_prompt: Final = start_time if isinstance(start_time, str) else ""
        res: Final = delegate.post_call_hook(
            safe_payload,
            response_obj=response_obj,
            prompt=eff_prompt,
            end_time=end_time,
        )
        self._record_populated_tenant(tenant_key)
        return res

    async def async_log_success_event(
        self,
        kwargs: object,
        response_obj: object = None,
        start_time: object = "",
        end_time: object = None,
    ) -> Mapping[str, object]:
        is_proxy_log: Final = isinstance(kwargs, Mapping) and (
            "litellm_params" in kwargs or "standard_logging_object" in kwargs
        )
        tenant_key: Final = _extract_tenant_scope(
            self.tenant_scope, None, kwargs, is_proxy=is_proxy_log
        )
        if not (
            self.enable_semantic_cache
            and tenant_key is not None
            and self._can_retain_tenant(tenant_key)
        ):
            return {"cached": False, "reason": "semantic_cache_disabled_or_unscoped"}
        delegate: Final = self._get_delegate(tenant_key)
        safe_payload: Final = _sanitize_logging_payload(kwargs, tenant_key)
        eff_prompt: Final = start_time if isinstance(start_time, str) else ""
        res: Final = await delegate.async_log_success_event(
            safe_payload,
            response_obj=response_obj,
            prompt=eff_prompt,
            end_time=end_time,
        )
        self._record_populated_tenant(tenant_key)
        return res


silicium_ai_ops_logger: Final = SiliciumAiOpsLogger()