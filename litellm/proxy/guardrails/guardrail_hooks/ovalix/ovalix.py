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
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final, Literal, NamedTuple

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
    tool_result_text_indices,
)
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.utils import GenericGuardrailAPIInputs

if TYPE_CHECKING:
    from litellm.types.proxy.guardrails.guardrail_hooks.base import GuardrailConfigModel


BLOCKED_BY_OVALIX_FALLBACK_MESSAGE: Final = "This message was blocked by Ovalix"
BLOCKED_ACTION_TYPE: Final = "block"
_MODIFY_ACTION_TYPES: Final = ("anonymize", "sanitize")
_APPLICATION_NOT_FOUND_STATUS: Final = 404
_ROUTING_CACHE_TTL_SECONDS: Final = 3600
_ROUTING_CACHE_NEGATIVE_TTL_SECONDS: Final = 300
_ROUTING_CACHE_MAX_SIZE: Final = 1000
_DEFAULT_FILE_SIZE_LIMIT: Final = 64 * 1024 * 1024
_NO_METADATA: Final[Mapping[str, Any]] = MappingProxyType({})
_FILE_BLOCK_ESCALATION_REASON: Final = (
    "This message was blocked by Ovalix because file content anonymization isn't possible via LiteLLM"
)
_TOOL_BLOCK_ESCALATION_REASON: Final = (
    "This message was blocked by Ovalix because tool call anonymization isn't possible via LiteLLM"
)
_TOOL_RESULT_BLOCK_ESCALATION_REASON: Final = (
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

    @property
    def has_any_checkpoint(self) -> bool:
        return any(
            (
                self.checkpoint_id_pre,
                self.checkpoint_id_post,
                self.checkpoint_id_pre_file,
                self.checkpoint_id_post_file,
            )
        )


class CheckpointTarget(NamedTuple):
    """How a checkpoint call addresses the application.

    When application_name is set the call sends the name and the direction, and the tracker resolves
    and evaluates in one request; resolution can create the application, and ids from a separate
    resolve call may name one the tracker's config has not caught up with, which it reports as an
    uninspected allow rather than an error. application_id is what the session id groups on, and is
    what gets sent when the deployment pins the application in config and reads no alias.
    """

    application_id: str
    input_type: str
    application_name: str | None = None


def _coerce_bool(value: bool | str) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "on")


class OvalixGuardrailMissingSecrets(Exception):
    """Raised when required Ovalix config (API base, key, application/checkpoint IDs) is missing."""


class OvalixGuardrailBlockedException(GuardrailRaisedException):
    """
    Raised when Ovalix blocks a message. Sets status_code=400 so the proxy
    returns 400 and HTTP clients do not retry (they retry on 5xx).
    """

    status_code = 400

    def __init__(
        self,
        guardrail_name: str | None = None,
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
    def get_supported_event_hooks(cls) -> list[GuardrailEventHooks]:
        return [
            GuardrailEventHooks.pre_call,
            GuardrailEventHooks.post_call,
        ]

    def __init__(
        self,
        tracker_api_base: str | None = None,
        tracker_api_key: str | None = None,
        application_id: str | None = None,
        pre_checkpoint_id: str | None = None,
        post_checkpoint_id: str | None = None,
        file_checkpoint_id: str | None = None,
        enable_routing_cache: bool | None = None,
        fail_if_no_application: bool | None = None,
        **kwargs: Any,
    ):
        self._tracker_api_base = tracker_api_base or os.environ.get("OVALIX_TRACKER_API_BASE")
        self._tracker_api_key = tracker_api_key or os.environ.get("OVALIX_TRACKER_API_KEY")
        self._application_id = application_id or os.environ.get("OVALIX_APPLICATION_ID")
        self._pre_checkpoint_id = pre_checkpoint_id or os.environ.get("OVALIX_PRE_CHECKPOINT_ID")
        self._post_checkpoint_id = post_checkpoint_id or os.environ.get("OVALIX_POST_CHECKPOINT_ID")
        self._file_checkpoint_id = file_checkpoint_id or os.environ.get("OVALIX_FILE_CHECKPOINT_ID")
        env_enable_routing_cache: Final = os.environ.get("OVALIX_ENABLE_ROUTING_CACHE")
        resolved_enable_routing_cache: Final = (
            enable_routing_cache if enable_routing_cache is not None else env_enable_routing_cache
        )
        self._enable_routing_cache = (
            True if resolved_enable_routing_cache is None else _coerce_bool(resolved_enable_routing_cache)
        )
        env_fail_if_no_application: Final = os.environ.get("OVALIX_FAIL_IF_NO_APPLICATION")
        resolved_fail_if_no_application: Final = (
            fail_if_no_application if fail_if_no_application is not None else env_fail_if_no_application
        )
        self._fail_if_no_application = (
            True if resolved_fail_if_no_application is None else _coerce_bool(resolved_fail_if_no_application)
        )
        self._routing_cache: OrderedDict[str, tuple[float, ResolvedRouting | None]] = OrderedDict()
        self._app_name_regex: re.Pattern[str] | None = None

        supported_event_hooks: Final[list[GuardrailEventHooks]] = list(kwargs.get("supported_event_hooks") or ())

        self._validate_config(supported_event_hooks)

        self._tracker_headers = dict(
            httpx.Headers(
                MappingProxyType(
                    {
                        "Authorization": f"Bearer {self._tracker_api_key}",
                        "x-api-key": self._tracker_api_key or "",
                        "Content-Type": "application/json",
                    }
                ),
                encoding="utf-8",
            )
        )

        self._async_handler = get_async_httpx_client(llm_provider=httpxSpecialProvider.GuardrailCallback)

        super().__init__(**{**kwargs, "supported_event_hooks": supported_event_hooks})
        verbose_proxy_logger.debug(
            "Ovalix Guardrail initialized: tracker=%s, application_id=%s, pre_checkpoint_id=%s, post_checkpoint_id=%s",
            self._tracker_api_base,
            self._application_id,
            self._pre_checkpoint_id,
            self._post_checkpoint_id,
        )

    def _validate_config(self, supported_event_hooks: list[GuardrailEventHooks]) -> None:
        """Ensure required Tracker secrets are set; register the pre/post hooks this config can serve (both in discovery mode; only configured-checkpoint directions in static mode)."""
        errors: Final = tuple(
            message
            for present, message in (
                (not self._tracker_api_base, "Tracker API base, set OVALIX_TRACKER_API_BASE or pass tracker_api_base"),
                (not self._tracker_api_key, "Tracker API key, set OVALIX_TRACKER_API_KEY or pass tracker_api_key"),
                (
                    bool(self._application_id) and not self._pre_checkpoint_id and not self._post_checkpoint_id,
                    "With application_id set, provide OVALIX_PRE_CHECKPOINT_ID and/or OVALIX_POST_CHECKPOINT_ID",
                ),
            )
            if present
        )

        if errors:
            raise OvalixGuardrailMissingSecrets("Missing Ovalix guardrail configuration errors: " + ". ".join(errors))

        supports_pre: Final = not self._application_id or bool(self._pre_checkpoint_id)
        supports_post: Final = not self._application_id or bool(self._post_checkpoint_id)
        if supports_pre and GuardrailEventHooks.pre_call not in supported_event_hooks:
            supported_event_hooks.append(GuardrailEventHooks.pre_call)
        if supports_post and GuardrailEventHooks.post_call not in supported_event_hooks:
            supported_event_hooks.append(GuardrailEventHooks.post_call)

    def _get_actor(self, data: Mapping[str, Any]) -> str:
        """Return a stable actor identifier from request metadata (e.g. user email or id)."""
        metadata: Final = data.get("metadata") or data.get("litellm_metadata") or _NO_METADATA
        if metadata.get("user_api_key_user_email"):
            return metadata["user_api_key_user_email"]
        if metadata.get("user_api_key_user_id"):
            return metadata["user_api_key_user_id"]
        return ""

    def _get_tracker_actor_id(self, data: Mapping[str, Any]) -> str:
        """Normalize the actor string into a short, stable id for Tracker API payloads."""
        # NOTE: this hash is purely for normalization — it collapses an arbitrary actor
        # string (email, user id, or empty) into a compact, fixed-length, consistent
        # key. It is not a privacy/security measure and the actor value is not sensitive,
        # so a plain SHA-256 (truncated) is sufficient; no salting/KDF is needed here.
        actor_id: Final = self._get_actor(data).encode()
        normalized_actor_id: Final = hashlib.sha256(actor_id).hexdigest()[:8]
        return normalized_actor_id

    def _get_session_id(self, data: Mapping[str, Any]) -> str:
        """Return a unique identifier for the chat/session (actor + date + application_id)."""
        return self._get_session_id_for_application(data, self._application_id)

    async def _call_checkpoint(
        self,
        data_type: str,
        data: Mapping[str, Any],
        checkpoint_id: str,
        actor: str,
        session_id: str,
        target: CheckpointTarget,
    ) -> Mapping[str, Any]:
        """Call the Ovalix Tracker checkpoint API and return the JSON response.

        Both routes live on the tracker's /beta litellm router, which accepts the api key this
        guardrail already sends. The name and id routing forms are mutually exclusive, so exactly one
        reaches the wire; the name form also sends the direction, which is what the tracker selects
        the pre or post checkpoint by.
        """
        if not target.application_name and (not target.application_id or not checkpoint_id):
            raise ValueError("Ovalix: application_id or checkpoint_id not resolved")

        route: Final = "file_checkpoint" if data_type == "FILE" else "checkpoint"
        routing: Final = (
            {"application_name": target.application_name, "input_type": target.input_type}
            if target.application_name
            else {"application_id": target.application_id, "checkpoint_id": checkpoint_id}
        )
        payload: Final = {
            "actor": actor,
            "session_id": session_id,
            "data_type": data_type,
            "data": data,
            "tool": "LiteLLM",
            **routing,
        }
        response: Final = await self._async_handler.post(
            f"{self._tracker_api_base}/tracking/beta/{route}", headers=self._tracker_headers, json=payload
        )
        response.raise_for_status()
        return response.json()

    def _verdict(self, resp: Mapping[str, Any]) -> tuple[str, str | None]:
        return (resp.get("action_type") or "").lower(), self._get_trackers_corrected_message(resp)

    async def _block_reason_for_item(
        self,
        data_type: str,
        data: Mapping[str, Any],
        checkpoint_id: str,
        actor: str,
        session_id: str,
        target: CheckpointTarget,
        escalation_reason: str,
    ) -> str | None:
        try:
            resp: Final = await self._call_checkpoint(data_type, data, checkpoint_id, actor, session_id, target)
        except Exception as e:
            verbose_proxy_logger.exception("Ovalix checkpoint call failed: %s", e)
            raise GuardrailRaisedException(
                guardrail_name=self.guardrail_name,
                message=f"Ovalix guardrail error: {e}",
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
        items: Sequence[tuple[str, Mapping[str, Any]]],
        checkpoint_id: str,
        actor: str,
        session_id: str,
        target: CheckpointTarget,
        escalation_reason: str,
    ) -> str | None:
        for data_type, data in items:
            reason = await self._block_reason_for_item(
                data_type, data, checkpoint_id, actor, session_id, target, escalation_reason
            )
            if reason is not None:
                return reason
        return None

    async def _check_files_for_block(
        self,
        file_parts: Sequence[FilePart],
        checkpoint_id: str,
        actor: str,
        session_id: str,
        target: CheckpointTarget,
    ) -> str | None:
        for part in sorted(file_parts, key=lambda p: p.message_index, reverse=True):
            data = await self._file_part_to_data(part)
            reason = await self._block_reason_for_item(
                "FILE", data, checkpoint_id, actor, session_id, target, _FILE_BLOCK_ESCALATION_REASON
            )
            if reason is not None:
                return reason
        return None

    @log_guardrail_information
    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: Mapping[str, Any],
        input_type: Literal["request", "response"],
        logging_obj: Any | None = None,
    ) -> GenericGuardrailAPIInputs:
        routing: Final = await self._resolve_routing(request_data)
        if routing is None:
            return inputs
        actor: Final = self._get_actor(request_data)
        session_id: Final = self._get_session_id_for_application(request_data, routing.application_id)
        is_response: Final = input_type == "response"
        target: Final = CheckpointTarget(
            application_id=routing.application_id,
            input_type=input_type,
            application_name=await self._checkpoint_routing_name(request_data),
        )

        prompt_checkpoint: Final = routing.checkpoint_id_post if is_response else routing.checkpoint_id_pre
        file_checkpoint: Final = (
            routing.checkpoint_id_post_file if is_response else routing.checkpoint_id_pre_file
        ) or prompt_checkpoint
        if not routing.has_any_checkpoint:
            raise GuardrailRaisedException(
                guardrail_name=self.guardrail_name,
                message=f"Ovalix guardrail error: application {routing.application_id} has no checkpoints configured",
                should_wrap_with_default_message=False,
            )
        if not file_checkpoint:
            verbose_proxy_logger.debug(
                "Ovalix guardrail: application %s has no %s checkpoint, leaving this direction uninspected",
                routing.application_id,
                input_type,
            )
            return inputs

        structured_messages: Final = inputs.get("structured_messages") or ()
        file_parts: Final = (
            extract_file_parts_from_images(inputs.get("images"), size_limit=_DEFAULT_FILE_SIZE_LIMIT)
            if is_response
            else extract_file_parts_from_messages(structured_messages, size_limit=_DEFAULT_FILE_SIZE_LIMIT)
        )
        file_block: Final = await self._check_files_for_block(file_parts, file_checkpoint, actor, session_id, target)
        if file_block is not None:
            self._block_current_message(file_block)

        if not prompt_checkpoint:
            verbose_proxy_logger.debug(
                "Ovalix guardrail: application %s has only a %s file checkpoint, skipping text and tool inspection",
                routing.application_id,
                input_type,
            )
            return inputs

        tool_call_items: Final = tuple(
            ("TOOL", td) for td in (tool_call_to_tool_data(tc) for tc in (inputs.get("tool_calls") or ())) if td
        )
        tool_block: Final = await self._check_items_block_only(
            tool_call_items,
            prompt_checkpoint,
            actor,
            session_id,
            target,
            _TOOL_BLOCK_ESCALATION_REASON,
        )
        if tool_block is not None:
            self._block_current_message(tool_block)

        tool_results: Final = extract_tool_results(structured_messages)
        tool_result_items: Final = tuple(("TOOL", make_tool_data(name, content)) for name, content, _ in tool_results)
        tool_result_block: Final = await self._check_items_block_only(
            tool_result_items,
            prompt_checkpoint,
            actor,
            session_id,
            target,
            _TOOL_RESULT_BLOCK_ESCALATION_REASON,
        )
        if tool_result_block is not None:
            self._block_current_message(tool_result_block)

        texts: Final = inputs.get("texts") or ()
        if not texts:
            return inputs
        output_texts: Final = await self._check_texts(
            texts,
            prompt_checkpoint,
            actor,
            session_id,
            target,
            tool_result_text_indices(structured_messages, texts),
        )
        if output_texts is None:
            return inputs
        return {**inputs, "texts": output_texts}

    async def _file_part_to_data(self, part: FilePart) -> Mapping[str, Any]:
        extension: Final = mimetypes.guess_extension(part.mime_hint) if part.mime_hint else None
        name: Final = part.name or (f"file{extension}" if extension else "file")
        content: Final = (
            await asyncio.get_event_loop().run_in_executor(None, _encode_file_wire_format, part.data)
            if part.data
            else None
        )
        return {"name": name, "content": content}

    async def _check_texts(
        self,
        texts: Sequence[str],
        checkpoint_id: str,
        actor: str,
        session_id: str,
        target: CheckpointTarget,
        skip_indices: frozenset[int],
    ) -> list[str] | None:
        original: Final = tuple(texts)
        output: Final = list(texts)
        count: Final = len(texts)
        for reversed_index in range(count):
            original_index = count - 1 - reversed_index
            if original_index in skip_indices:
                continue
            is_newest = reversed_index == 0
            content = texts[original_index]
            try:
                resp = await self._call_checkpoint(
                    "TEXT", {"content": content}, checkpoint_id, actor, session_id, target
                )
            except Exception as e:
                verbose_proxy_logger.exception("Ovalix checkpoint call failed: %s", e)
                raise GuardrailRaisedException(
                    guardrail_name=self.guardrail_name,
                    message=f"Ovalix guardrail error: {e}",
                    should_wrap_with_default_message=False,
                ) from e
            action, corrected = self._verdict(resp)
            if action == BLOCKED_ACTION_TYPE:
                block_message = corrected or BLOCKED_BY_OVALIX_FALLBACK_MESSAGE
                if is_newest:
                    self._block_current_message(block_message)
                output[original_index] = block_message
                continue
            if action in _MODIFY_ACTION_TYPES and corrected is not None and corrected != content:
                output[original_index] = corrected
        return output if tuple(output) != original else None

    def _get_session_id_for_application(self, data: Mapping[str, Any], application_id: str | None) -> str:
        actor_hash: Final = self._get_tracker_actor_id(data)
        today: Final = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
        return f"{actor_hash}_{today}_{application_id}"

    def _block_current_message(self, blocking_message: str) -> None:
        """Raise OvalixGuardrailBlockedException with the given message (no default wrapper)."""
        raise OvalixGuardrailBlockedException(
            guardrail_name=self.guardrail_name,
            message=blocking_message,
            should_wrap_with_default_message=False,
        )

    def _get_trackers_corrected_message(self, resp: Mapping[str, Any]) -> str | None:
        """Extract corrected/blocking message content from Tracker checkpoint response."""
        modified: Final = resp.get("modified_data")
        if isinstance(modified, dict) and "content" in modified:
            return modified["content"]
        return None

    def _get_key_alias(self, request_data: Mapping[str, Any]) -> str | None:
        litellm_metadata: Final = request_data.get("litellm_metadata") or _NO_METADATA
        metadata: Final = request_data.get("metadata") or _NO_METADATA

        def _merged(key: str) -> object:
            return litellm_metadata.get(key) if key in litellm_metadata else metadata.get(key)

        alias: Final = _merged("user_api_key_alias") or _merged("user_api_key_key_alias")
        return alias if isinstance(alias, str) else None

    async def _get_app_name_regex(self) -> re.Pattern[str]:
        if self._app_name_regex is not None:
            return self._app_name_regex
        url: Final = f"{self._tracker_api_base}/tracking/beta/app_name_regex"
        try:
            response: Final = await self._async_handler.get(url, headers=self._tracker_headers)
            response.raise_for_status()
            compiled: Final = re.compile(response.json()["regex"])
        except Exception as e:
            verbose_proxy_logger.exception("Ovalix app-name regex fetch failed: %s", e)
            raise GuardrailRaisedException(
                guardrail_name=self.guardrail_name,
                message=f"Ovalix guardrail error: app-name regex fetch failed: {e}",
                should_wrap_with_default_message=False,
            ) from e
        self._app_name_regex = compiled
        return compiled

    def _extract_application_name(self, alias: str, regex: re.Pattern[str]) -> str | None:
        match: Final = regex.search(alias)
        if not match:
            return None
        name: Final = (match.group(1) if match.groups() else match.group(0)).strip()
        return name or None

    def _routing_cache_get(self, name: str) -> tuple[bool, ResolvedRouting | None]:
        entry: Final = self._routing_cache.get(name)
        if entry is None:
            return False, None
        expires_at, routing = entry
        if time.monotonic() >= expires_at:
            del self._routing_cache[name]
            return False, None
        self._routing_cache.move_to_end(name)
        return True, routing

    def _routing_cache_put(self, name: str, routing: ResolvedRouting | None) -> None:
        ttl: Final = _ROUTING_CACHE_TTL_SECONDS if routing is not None else _ROUTING_CACHE_NEGATIVE_TTL_SECONDS
        self._routing_cache[name] = (time.monotonic() + ttl, routing)
        self._routing_cache.move_to_end(name)
        while len(self._routing_cache) > _ROUTING_CACHE_MAX_SIZE:
            self._routing_cache.popitem(last=False)

    def _no_application(self, reason: str) -> None:
        if self._fail_if_no_application:
            raise GuardrailRaisedException(
                guardrail_name=self.guardrail_name,
                message=f"Ovalix guardrail error: {reason}",
                should_wrap_with_default_message=False,
            )
        verbose_proxy_logger.warning(
            "Ovalix guardrail passing the call through unguarded (fail_if_no_application=false): %s", reason
        )

    def _routing_error(self, error: Exception) -> GuardrailRaisedException:
        verbose_proxy_logger.exception("Ovalix routing resolution failed: %s", error)
        return GuardrailRaisedException(
            guardrail_name=self.guardrail_name,
            message=f"Ovalix guardrail error: routing resolution failed: {error}",
            should_wrap_with_default_message=False,
        )

    async def _checkpoint_routing_name(self, request_data: Mapping[str, Any]) -> str | None:
        """The application name to route checkpoints by, or None to route by resolved ids.

        None when the deployment pins an application in config, or when no name can be read from the
        api key alias. Only reached after _resolve_routing has already fetched and cached the regex.
        """
        if self._application_id:
            return None
        alias: Final = self._get_key_alias(request_data)
        if not alias:
            return None
        return self._extract_application_name(alias, await self._get_app_name_regex())

    async def _resolve_routing(self, request_data: Mapping[str, Any]) -> ResolvedRouting | None:
        if self._application_id:
            return ResolvedRouting(
                self._application_id,
                self._pre_checkpoint_id,
                self._post_checkpoint_id,
                self._file_checkpoint_id,
                self._file_checkpoint_id,
            )
        alias: Final = self._get_key_alias(request_data)
        if not alias:
            return self._no_application("no application_id configured and no user_api_key_alias to resolve by")
        regex: Final = await self._get_app_name_regex()
        name: Final = self._extract_application_name(alias, regex)
        if not name:
            return self._no_application("could not extract an application name from the api key alias")
        if self._enable_routing_cache:
            hit, cached = self._routing_cache_get(name)
            if hit:
                return cached if cached is not None else self._no_application(f"application '{name}' was not found")
        routing: Final = await self._resolve_via_tracker(name)
        if self._enable_routing_cache:
            self._routing_cache_put(name, routing)
        if routing is None:
            return self._no_application(f"application '{name}' was not found")
        return routing

    async def _resolve_via_tracker(self, application_name: str) -> ResolvedRouting | None:
        url: Final = f"{self._tracker_api_base}/tracking/beta/resolve_application"
        try:
            response: Final = await self._async_handler.post(
                url, headers=self._tracker_headers, json={"application_name": application_name}
            )
            response.raise_for_status()
            body: Final = response.json()
            return ResolvedRouting(
                application_id=str(body["application_id"]),
                checkpoint_id_pre=body.get("checkpoint_id_pre"),
                checkpoint_id_post=body.get("checkpoint_id_post"),
                checkpoint_id_pre_file=body.get("checkpoint_id_pre_file"),
                checkpoint_id_post_file=body.get("checkpoint_id_post_file"),
            )
        except httpx.HTTPStatusError as e:
            if e.response.status_code == _APPLICATION_NOT_FOUND_STATUS:
                return None
            raise self._routing_error(e) from e
        except Exception as e:
            raise self._routing_error(e) from e

    @staticmethod
    def get_config_model() -> type["GuardrailConfigModel"] | None:
        from litellm.types.proxy.guardrails.guardrail_hooks.ovalix import (
            OvalixGuardrailConfigModel,
        )

        return OvalixGuardrailConfigModel
