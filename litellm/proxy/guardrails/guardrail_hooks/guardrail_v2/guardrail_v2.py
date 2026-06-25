import asyncio
import json
from typing import TYPE_CHECKING, Callable, Literal, Optional

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


def _load_bridge():
    """Return the compiled litellm_python_bridge module, or raise ImportError when
    the optional extension is not installed so callers can fall back to Python."""
    import litellm_python_bridge

    return litellm_python_bridge


def config_supported(guardrail_type: str, params: dict) -> bool:
    """Whether the Rust engine can handle this guardrail type and params. Raises
    ImportError when the extension is absent."""
    bridge = _load_bridge()
    return bridge.guardrail_config_supported(
        guardrail_type, json.dumps(params, default=str)
    )


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


def _get_optional_param(litellm_params, name: str) -> object:
    """Read a guardrail param from optional_params, then litellm_params itself."""
    optional_params = getattr(litellm_params, "optional_params", None)
    if optional_params is not None:
        value = getattr(optional_params, name, None)
        if value is not None:
            return value
    return getattr(litellm_params, name, None)


class GuardrailV2(CustomGuardrail):
    def __init__(
        self,
        guardrail_type: str,
        params: dict,
        extra_headers: Optional[list] = None,
        streaming_end_of_stream_only: Optional[bool] = None,
        streaming_sampling_rate: Optional[int] = None,
        **kwargs: object,
    ):
        self.guardrail_type = guardrail_type
        # Raw guardrail params; the Rust engine builds the provider config from
        # these per request (resolving secrets and files), so Python never builds
        # or round-trips a provider config.
        self.params = params
        # Injection point for tests; set the attribute directly to bypass the
        # compiled bridge. None means call the real litellm_python_bridge.
        self._apply_guardrail_fn: Optional[Callable[[str], str]] = None
        self.extra_header_allowlist = [
            h for h in (extra_headers or []) if isinstance(h, str)
        ]
        # Read by UnifiedLLMGuardrails.async_post_call_streaming_iterator_hook to
        # drive streaming moderation. Defaults match the Python moderation guardrail.
        self.streaming_end_of_stream_only: bool = (
            False
            if streaming_end_of_stream_only is None
            else streaming_end_of_stream_only
        )
        self.streaming_sampling_rate: int = (
            5 if streaming_sampling_rate is None else streaming_sampling_rate
        )

        if "supported_event_hooks" not in kwargs:
            kwargs["supported_event_hooks"] = [
                GuardrailEventHooks.pre_call,
                GuardrailEventHooks.post_call,
                GuardrailEventHooks.during_call,
            ]

        super().__init__(**kwargs)

    def _apply(self, request_json: str) -> str:
        if self._apply_guardrail_fn is not None:
            return self._apply_guardrail_fn(request_json)
        return _load_bridge().apply_guardrails(request_json)

    def _collect_batch(
        self, request_data: dict, input_type: str
    ) -> "list[GuardrailV2]":
        """All Rust-backed guardrails active for this request at the same hook as
        self, so the engine evaluates the whole set at once (block-wins precedence,
        mask composition) instead of letting Python run them sequentially where a
        mask can hide content from a later block."""
        import litellm
        from litellm.types.guardrails import GuardrailEventHooks

        events = (
            [GuardrailEventHooks.post_call]
            if input_type == "response"
            else [GuardrailEventHooks.pre_call, GuardrailEventHooks.during_call]
        )
        batch: list[GuardrailV2] = []
        for cb in litellm.callbacks:
            if not isinstance(cb, GuardrailV2) or cb.event_hook != self.event_hook:
                continue
            try:
                active = any(
                    cb.should_run_guardrail(data=request_data, event_type=event)
                    for event in events
                )
            except Exception:
                active = cb is self
            if active:
                batch.append(cb)
        if self not in batch:
            batch.append(self)
        return batch

    @log_guardrail_information
    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: Optional["LiteLLMLoggingObj"] = None,
    ) -> GenericGuardrailAPIInputs:
        rd = request_data if request_data is not None else {}
        # The first Rust guardrail invoked for this request/input_type runs the
        # whole set; the rest are no-ops so the batch is applied exactly once. The
        # marker lives in litellm metadata (not the top-level request dict) so it
        # is never forwarded to the upstream provider as a request parameter.
        metadata = rd.get("metadata")
        if metadata is None:
            metadata = rd.get("litellm_metadata")
        if metadata is None:
            metadata = {}
            rd["metadata"] = metadata
        marker = f"_litellm_rust_guardrail_batch_{input_type}"
        if metadata.get(marker):
            out: GenericGuardrailAPIInputs = {}
            out.update(inputs)
            return out
        metadata[marker] = True

        batch = self._collect_batch(rd, input_type)
        request_body = rd.get("body") or {}
        dynamic_params = self.get_guardrail_dynamic_request_body_params(request_body)

        request = {
            "guardrails": [
                {"guardrail_type": g.guardrail_type, "params": g.params} for g in batch
            ],
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
                "user_api_key_metadata": _extract_user_metadata(rd),
                "request_headers": _extract_raw_headers(rd, logging_obj),
                "dynamic_params": dynamic_params or {},
                "litellm_version": litellm_version,
                "extra_header_allowlist": self.extra_header_allowlist,
            },
        }

        request_json = json.dumps(request, default=str)
        # The engine releases the GIL during the provider calls, so running it in a
        # worker thread keeps the event loop responsive.
        result_json = await asyncio.get_running_loop().run_in_executor(
            None, self._apply, request_json
        )
        result = json.loads(result_json)
        verdict = result.get("verdict", {})
        action = verdict.get("action")

        if action == "block":
            raise GuardrailRaisedException(
                guardrail_name=self.guardrail_name or self.guardrail_type,
                message=verdict.get("violation_message", "Content violates policy"),
                should_wrap_with_default_message=False,
            )

        out = {}
        out.update(inputs)
        if action == "mask":
            masked = verdict.get("texts")
            if masked is not None:
                out["texts"] = masked
        return out
