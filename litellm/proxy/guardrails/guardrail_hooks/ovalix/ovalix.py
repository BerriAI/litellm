"""Ovalix guardrail integration: pre- and post-call checks via the Tracker service.

Use Ovalix Guardrails for your LLM calls. Supports pre_call (user input) and
post_call (model output) checkpoints with optional correction/blocking.
"""

import asyncio
import base64
import datetime
import gzip
import hashlib
import mimetypes
import os
import re
import time
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, List, Literal, NamedTuple, Optional, Type

import httpx

from litellm._logging import verbose_proxy_logger
from litellm.exceptions import GuardrailRaisedException
from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    log_guardrail_information,
)
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.proxy.guardrails.guardrail_hooks.ovalix.ovalix_extraction import (
    FilePart,
    extract_file_parts_from_images,
    extract_file_parts_from_messages,
    extract_tool_results,
    make_tool_data,
    tool_call_to_tool_data,
)
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.utils import GenericGuardrailAPIInputs

if TYPE_CHECKING:
    from litellm.types.proxy.guardrails.guardrail_hooks.base import GuardrailConfigModel


BLOCKED_BY_OVALIX_FALLBACK_MESSAGE = "This message was blocked by Ovalix"
BLOCKED_ACTION_TYPE = "block"
_MODIFY_ACTION_TYPES = ("anonymize", "sanitize")
_ROUTING_CACHE_TTL_SECONDS = 3600
_ROUTING_CACHE_MAX_SIZE = 1000
_DEFAULT_FILE_SIZE_LIMIT = 64 * 1024 * 1024
_FILE_BLOCK_ESCALATION_REASON = (
    "This message was blocked by Ovalix because file content anonymization isn't possible via LiteLLM"
)
_TOOL_BLOCK_ESCALATION_REASON = (
    "This message was blocked by Ovalix because tool call anonymization isn't possible via LiteLLM"
)
_TOOL_RESULT_BLOCK_ESCALATION_REASON = (
    "This message was blocked by Ovalix because tool result anonymization isn't possible via LiteLLM"
)


def _encode_file_wire_format(raw: bytes) -> str:
    return base64.b64encode(gzip.compress(raw)).decode()


class ResolvedRouting(NamedTuple):
    application_id: str
    checkpoint_id_pre: str | None
    checkpoint_id_post: str | None
    checkpoint_id_pre_file: str | None
    checkpoint_id_post_file: str | None


def _coerce_bool(value: bool | str) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "on")


class OvalixGuardrailMissingSecrets(Exception):
    """Raised when required Ovalix config (API base, key, application/checkpoint IDs) is missing."""

    pass


class OvalixGuardrailBlockedException(GuardrailRaisedException):
    """
    Raised when Ovalix blocks a message. Sets status_code=400 so the proxy
    returns 400 and HTTP clients do not retry (they retry on 5xx).
    """

    status_code = 400

    def __init__(
        self,
        guardrail_name: Optional[str] = None,
        message: str = "",
        should_wrap_with_default_message: bool = True,
    ):
        super().__init__(
            guardrail_name=guardrail_name,
            message=message,
            should_wrap_with_default_message=should_wrap_with_default_message,
        )


class OvalixGuardrail(CustomGuardrail):
    """
    Ovalix guardrail: pre-prompt (pre_call) and post-prompt (post_call) checks
    via the Tracker service, with application and checkpoint resolution from the
    Monolith backend.
    """

    @classmethod
    def get_supported_event_hooks(cls) -> List[GuardrailEventHooks]:
        return [
            GuardrailEventHooks.pre_call,
            GuardrailEventHooks.post_call,
        ]

    def __init__(
        self,
        tracker_api_base: Optional[str] = None,
        tracker_api_key: Optional[str] = None,
        application_id: Optional[str] = None,
        pre_checkpoint_id: Optional[str] = None,
        post_checkpoint_id: Optional[str] = None,
        file_checkpoint_id: str | None = None,
        enable_routing_cache: bool | None = None,
        **kwargs: Any,
    ):
        self._tracker_api_base = tracker_api_base or os.environ.get("OVALIX_TRACKER_API_BASE")
        self._tracker_api_key = tracker_api_key or os.environ.get("OVALIX_TRACKER_API_KEY")
        self._application_id = application_id or os.environ.get("OVALIX_APPLICATION_ID")
        self._pre_checkpoint_id = pre_checkpoint_id or os.environ.get("OVALIX_PRE_CHECKPOINT_ID")
        self._post_checkpoint_id = post_checkpoint_id or os.environ.get("OVALIX_POST_CHECKPOINT_ID")
        self._file_checkpoint_id = file_checkpoint_id or os.environ.get("OVALIX_FILE_CHECKPOINT_ID")
        env_enable_routing_cache = os.environ.get("OVALIX_ENABLE_ROUTING_CACHE")
        resolved_enable_routing_cache = (
            enable_routing_cache if enable_routing_cache is not None else env_enable_routing_cache
        )
        self._enable_routing_cache = (
            True if resolved_enable_routing_cache is None else _coerce_bool(resolved_enable_routing_cache)
        )
        self._routing_cache: OrderedDict[str, tuple[float, ResolvedRouting]] = OrderedDict()
        self._app_name_regex: re.Pattern[str] | None = None

        if "supported_event_hooks" not in kwargs:
            kwargs["supported_event_hooks"] = []

        self._validate_config(kwargs["supported_event_hooks"])

        self._tracker_headers = httpx.Headers(
            {
                "Authorization": f"Bearer {self._tracker_api_key}",
                "Content-Type": "application/json",
            },
            encoding="utf-8",
        )

        self._async_handler = get_async_httpx_client(llm_provider=httpxSpecialProvider.GuardrailCallback)

        super().__init__(**kwargs)
        verbose_proxy_logger.debug(
            "Ovalix Guardrail initialized: tracker=%s, application_id=%s, pre_checkpoint_id=%s, post_checkpoint_id=%s",
            self._tracker_api_base,
            self._application_id,
            self._pre_checkpoint_id,
            self._post_checkpoint_id,
        )

    def _validate_config(self, supported_event_hooks: List[GuardrailEventHooks]) -> None:
        """Ensure required Tracker secrets are set; register the pre/post hooks this config can serve (both in discovery mode; only configured-checkpoint directions in static mode)."""
        errors: List[str] = []

        if not self._tracker_api_base:
            errors.append("Tracker API base, set OVALIX_TRACKER_API_BASE or pass tracker_api_base")
        if not self._tracker_api_key:
            errors.append("Tracker API key, set OVALIX_TRACKER_API_KEY or pass tracker_api_key")
        if self._application_id and not self._pre_checkpoint_id and not self._post_checkpoint_id:
            errors.append("With application_id set, provide OVALIX_PRE_CHECKPOINT_ID and/or OVALIX_POST_CHECKPOINT_ID")

        if errors:
            raise OvalixGuardrailMissingSecrets("Missing Ovalix guardrail configuration errors: " + ". ".join(errors))

        supports_pre = not self._application_id or bool(self._pre_checkpoint_id)
        supports_post = not self._application_id or bool(self._post_checkpoint_id)
        if supports_pre and GuardrailEventHooks.pre_call not in supported_event_hooks:
            supported_event_hooks.append(GuardrailEventHooks.pre_call)
        if supports_post and GuardrailEventHooks.post_call not in supported_event_hooks:
            supported_event_hooks.append(GuardrailEventHooks.post_call)

    def _get_actor(self, data: dict) -> str:
        """Return a stable actor identifier from request metadata (e.g. user email or id)."""
        metadata = data.get("metadata") or data.get("litellm_metadata") or {}
        if metadata.get("user_api_key_user_email"):
            return metadata["user_api_key_user_email"]
        if metadata.get("user_api_key_user_id"):
            return metadata["user_api_key_user_id"]
        return ""

    def _get_tracker_actor_id(self, data: dict) -> str:
        """Normalize the actor string into a short, stable id for Tracker API payloads."""
        # NOTE: this hash is purely for normalization — it collapses an arbitrary actor
        # string (email, user id, or empty) into a compact, fixed-length, consistent
        # key. It is not a privacy/security measure and the actor value is not sensitive,
        # so a plain SHA-256 (truncated) is sufficient; no salting/KDF is needed here.
        actor_id = self._get_actor(data).encode()
        normalized_actor_id = hashlib.sha256(actor_id).hexdigest()[:8]
        return normalized_actor_id

    def _get_session_id(self, data: dict) -> str:
        """Return a unique identifier for the chat/session (actor + date + application_id)."""
        return self._get_session_id_for_application(data, self._application_id)

    async def _call_checkpoint(
        self,
        data_type: str,
        data: dict[str, Any],
        checkpoint_id: str,
        actor: str,
        session_id: str,
        application_id: str,
    ) -> dict[str, Any]:
        """Call the Ovalix Tracker checkpoint API and return the JSON response."""
        if not application_id or not checkpoint_id:
            raise ValueError("Ovalix: application_id or checkpoint_id not resolved")

        url = f"{self._tracker_api_base}/tracking/custom_application/checkpoint"
        payload = {
            "application_id": application_id,
            "checkpoint_id": checkpoint_id,
            "actor": actor,
            "session_id": session_id,
            "data_type": data_type,
            "data": data,
            "tool": "LiteLLM",
        }
        response = await self._async_handler.post(url, headers=dict(self._tracker_headers), json=payload)
        response.raise_for_status()
        return response.json()

    def _verdict(self, resp: dict[str, Any]) -> tuple[str, str | None]:
        return (resp.get("action_type") or "").lower(), self._get_trackers_corrected_message(resp)

    async def _block_reason_for_item(
        self,
        data_type: str,
        data: dict[str, Any],
        checkpoint_id: str,
        actor: str,
        session_id: str,
        application_id: str,
        escalation_reason: str,
    ) -> str | None:
        try:
            resp = await self._call_checkpoint(data_type, data, checkpoint_id, actor, session_id, application_id)
        except Exception as e:
            verbose_proxy_logger.exception("Ovalix checkpoint call failed: %s", e)
            raise GuardrailRaisedException(
                guardrail_name=self.guardrail_name,
                message=f"Ovalix guardrail error: {e!s}",
                should_wrap_with_default_message=False,
            ) from e
        action, corrected = self._verdict(resp)
        if action == BLOCKED_ACTION_TYPE:
            return corrected or BLOCKED_BY_OVALIX_FALLBACK_MESSAGE
        if action in _MODIFY_ACTION_TYPES:
            return escalation_reason
        return None

    async def _check_items_block_only(
        self,
        items: list[tuple[str, dict[str, Any]]],
        checkpoint_id: str,
        actor: str,
        session_id: str,
        application_id: str,
        escalation_reason: str,
    ) -> str | None:
        for data_type, data in items:
            reason = await self._block_reason_for_item(
                data_type, data, checkpoint_id, actor, session_id, application_id, escalation_reason
            )
            if reason is not None:
                return reason
        return None

    async def _check_files_for_block(
        self,
        file_parts: list[FilePart],
        checkpoint_id: str,
        actor: str,
        session_id: str,
        application_id: str,
    ) -> str | None:
        for part in sorted(file_parts, key=lambda p: p.message_index, reverse=True):
            data = await self._file_part_to_data(part)
            reason = await self._block_reason_for_item(
                "FILE", data, checkpoint_id, actor, session_id, application_id, _FILE_BLOCK_ESCALATION_REASON
            )
            if reason is not None:
                return reason
        return None

    @log_guardrail_information
    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: Optional[Any] = None,
    ) -> GenericGuardrailAPIInputs:
        routing = await self._resolve_routing(request_data)
        actor = self._get_actor(request_data)
        session_id = self._get_session_id_for_application(request_data, routing.application_id)
        is_response = input_type == "response"

        prompt_checkpoint = routing.checkpoint_id_post if is_response else routing.checkpoint_id_pre
        file_checkpoint = (
            routing.checkpoint_id_post_file if is_response else routing.checkpoint_id_pre_file
        ) or prompt_checkpoint
        if not prompt_checkpoint:
            raise GuardrailRaisedException(
                guardrail_name=self.guardrail_name,
                message="Ovalix guardrail error: no checkpoint resolved for input_type",
                should_wrap_with_default_message=False,
            )

        structured_messages = inputs.get("structured_messages") or []
        file_parts = (
            extract_file_parts_from_images(inputs.get("images"), size_limit=_DEFAULT_FILE_SIZE_LIMIT)
            if is_response
            else extract_file_parts_from_messages(structured_messages, size_limit=_DEFAULT_FILE_SIZE_LIMIT)
        )
        file_block = await self._check_files_for_block(
            file_parts, file_checkpoint, actor, session_id, routing.application_id
        )
        if file_block is not None:
            self._block_current_message(file_block)

        tool_call_items = [
            ("TOOL", td) for td in (tool_call_to_tool_data(tc) for tc in (inputs.get("tool_calls") or [])) if td
        ]
        tool_block = await self._check_items_block_only(
            tool_call_items,
            prompt_checkpoint,
            actor,
            session_id,
            routing.application_id,
            _TOOL_BLOCK_ESCALATION_REASON,
        )
        if tool_block is not None:
            self._block_current_message(tool_block)

        tool_results = extract_tool_results(structured_messages)
        tool_result_items = [("TOOL", make_tool_data(name, content)) for name, content, _ in tool_results]
        tool_result_block = await self._check_items_block_only(
            tool_result_items,
            prompt_checkpoint,
            actor,
            session_id,
            routing.application_id,
            _TOOL_RESULT_BLOCK_ESCALATION_REASON,
        )
        if tool_result_block is not None:
            self._block_current_message(tool_result_block)

        texts = inputs.get("texts") or []
        if not texts or not isinstance(texts, list):
            return inputs
        output_texts = await self._check_texts(texts, prompt_checkpoint, actor, session_id, routing.application_id)
        if output_texts is None:
            return inputs
        return {**inputs, "texts": output_texts}

    async def _file_part_to_data(self, part: FilePart) -> dict[str, Any]:
        extension = mimetypes.guess_extension(part.mime_hint) if part.mime_hint else None
        name = part.name or (f"file{extension}" if extension else "file")
        content = (
            await asyncio.get_event_loop().run_in_executor(None, _encode_file_wire_format, part.data)
            if part.data
            else None
        )
        return {"name": name, "content": content}

    async def _check_texts(
        self,
        texts: list[str],
        checkpoint_id: str,
        actor: str,
        session_id: str,
        application_id: str,
    ) -> list[str] | None:
        output = list(texts)
        changed = False
        count = len(texts)
        for reversed_index in range(count):
            original_index = count - 1 - reversed_index
            is_newest = reversed_index == 0
            content = texts[original_index]
            try:
                resp = await self._call_checkpoint(
                    "TEXT", {"content": content}, checkpoint_id, actor, session_id, application_id
                )
            except Exception as e:
                verbose_proxy_logger.exception("Ovalix checkpoint call failed: %s", e)
                raise GuardrailRaisedException(
                    guardrail_name=self.guardrail_name,
                    message=f"Ovalix guardrail error: {e!s}",
                    should_wrap_with_default_message=False,
                ) from e
            action, corrected = self._verdict(resp)
            if action == BLOCKED_ACTION_TYPE:
                block_message = corrected or BLOCKED_BY_OVALIX_FALLBACK_MESSAGE
                if is_newest:
                    self._block_current_message(block_message)
                if output[original_index] != block_message:
                    changed = True
                output[original_index] = block_message
                continue
            if action in _MODIFY_ACTION_TYPES and corrected is not None and corrected != content:
                changed = True
                output[original_index] = corrected
        return output if changed else None

    def _get_session_id_for_application(self, data: dict, application_id: str | None) -> str:
        actor_hash = self._get_tracker_actor_id(data)
        today = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
        return f"{actor_hash}_{today}_{application_id}"

    def _block_current_message(self, blocking_message: str) -> None:
        """Raise OvalixGuardrailBlockedException with the given message (no default wrapper)."""
        raise OvalixGuardrailBlockedException(
            guardrail_name=self.guardrail_name,
            message=blocking_message,
            should_wrap_with_default_message=False,
        )

    def _get_trackers_corrected_message(self, resp: dict) -> Optional[str]:
        """Extract corrected/blocking message content from Tracker checkpoint response."""
        modified = resp.get("modified_data")
        if isinstance(modified, dict) and "content" in modified:
            return modified["content"]
        return None

    def _get_key_alias(self, request_data: dict) -> str | None:
        metadata = {**(request_data.get("metadata") or {}), **(request_data.get("litellm_metadata") or {})}
        return metadata.get("user_api_key_alias") or metadata.get("user_api_key_key_alias")

    async def _get_app_name_regex(self) -> re.Pattern[str]:
        if self._app_name_regex is not None:
            return self._app_name_regex
        url = f"{self._tracker_api_base}/tracking/custom_application/litellm_app_name_regex"
        try:
            response = await self._async_handler.get(url, headers=dict(self._tracker_headers))
            response.raise_for_status()
            compiled = re.compile(response.json()["regex"])
        except Exception as e:
            verbose_proxy_logger.exception("Ovalix app-name regex fetch failed: %s", e)
            raise GuardrailRaisedException(
                guardrail_name=self.guardrail_name,
                message=f"Ovalix guardrail error: app-name regex fetch failed: {e!s}",
                should_wrap_with_default_message=False,
            ) from e
        self._app_name_regex = compiled
        return compiled

    def _extract_application_name(self, alias: str, regex: re.Pattern[str]) -> str | None:
        match = regex.search(alias)
        if not match:
            return None
        name = (match.group(1) if match.groups() else match.group(0)).strip()
        return name or None

    def _routing_cache_get(self, name: str) -> ResolvedRouting | None:
        entry = self._routing_cache.get(name)
        if entry is None:
            return None
        stored_at, routing = entry
        if time.monotonic() - stored_at >= _ROUTING_CACHE_TTL_SECONDS:
            del self._routing_cache[name]
            return None
        self._routing_cache.move_to_end(name)
        return routing

    def _routing_cache_put(self, name: str, routing: ResolvedRouting) -> None:
        self._routing_cache[name] = (time.monotonic(), routing)
        self._routing_cache.move_to_end(name)
        while len(self._routing_cache) > _ROUTING_CACHE_MAX_SIZE:
            self._routing_cache.popitem(last=False)

    async def _resolve_routing(self, request_data: dict) -> ResolvedRouting:
        if self._application_id:
            return ResolvedRouting(
                self._application_id,
                self._pre_checkpoint_id,
                self._post_checkpoint_id,
                self._file_checkpoint_id,
                self._file_checkpoint_id,
            )
        alias = self._get_key_alias(request_data)
        if not alias:
            raise GuardrailRaisedException(
                guardrail_name=self.guardrail_name,
                message="Ovalix guardrail error: no application_id configured and no user_api_key_alias to resolve by",
                should_wrap_with_default_message=False,
            )
        regex = await self._get_app_name_regex()
        name = self._extract_application_name(alias, regex)
        if not name:
            raise GuardrailRaisedException(
                guardrail_name=self.guardrail_name,
                message="Ovalix guardrail error: could not extract an application name from the api key alias",
                should_wrap_with_default_message=False,
            )
        if self._enable_routing_cache:
            cached = self._routing_cache_get(name)
            if cached is not None:
                return cached
        routing = await self._resolve_via_tracker(name)
        if self._enable_routing_cache:
            self._routing_cache_put(name, routing)
        return routing

    async def _resolve_via_tracker(self, application_name: str) -> ResolvedRouting:
        url = f"{self._tracker_api_base}/tracking/custom_application/resolve_litellm_application"
        try:
            response = await self._async_handler.post(
                url, headers=dict(self._tracker_headers), json={"application_name": application_name}
            )
            response.raise_for_status()
            body = response.json()
            routing = ResolvedRouting(
                application_id=str(body["application_id"]),
                checkpoint_id_pre=body.get("checkpoint_id_pre"),
                checkpoint_id_post=body.get("checkpoint_id_post"),
                checkpoint_id_pre_file=body.get("checkpoint_id_pre_file"),
                checkpoint_id_post_file=body.get("checkpoint_id_post_file"),
            )
        except Exception as e:
            verbose_proxy_logger.exception("Ovalix routing resolution failed: %s", e)
            raise GuardrailRaisedException(
                guardrail_name=self.guardrail_name,
                message=f"Ovalix guardrail error: routing resolution failed: {e!s}",
                should_wrap_with_default_message=False,
            ) from e
        return routing

    @staticmethod
    def get_config_model() -> Optional[Type["GuardrailConfigModel"]]:
        from litellm.types.proxy.guardrails.guardrail_hooks.ovalix import (
            OvalixGuardrailConfigModel,
        )

        return OvalixGuardrailConfigModel
