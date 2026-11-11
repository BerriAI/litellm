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
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal

import httpx
from typing_extensions import TypedDict, Unpack

from litellm._logging import verbose_proxy_logger
from litellm.exceptions import GuardrailRaisedException
from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    log_guardrail_information,
)
from litellm.litellm_core_utils.core_helpers import as_str_mapping
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.proxy.guardrails.guardrail_hooks.ovalix.ovalix_extraction import (
    FilePart,
    extract_file_parts_from_images,
    extract_file_parts_from_messages,
    extract_tool_calls_from_messages,
    extract_tool_results,
    make_tool_data,
    str_or_none,
    tool_call_to_tool_data,
    tool_data_key,
    tool_result_text_indices,
)
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.utils import GenericGuardrailAPIInputs

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.types.proxy.guardrails.guardrail_hooks.base import GuardrailConfigModel


BLOCKED_BY_OVALIX_FALLBACK_MESSAGE: Final = "This message was blocked by Ovalix"
BLOCKED_ACTION_TYPE: Final = "block"
_MODIFY_ACTION_TYPES: Final = ("anonymize", "sanitize")
_DEFAULT_FILE_SIZE_LIMIT: Final = 64 * 1024 * 1024
_NO_METADATA: Final[Mapping[str, object]] = MappingProxyType({})
_FILE_BLOCK_ESCALATION_REASON: Final = (
    "This message was blocked by Ovalix because file content anonymization isn't possible via LiteLLM"
)
_UNDECODABLE_FILE_BLOCK_REASON: Final = (
    "This message was blocked by Ovalix because an attached file couldn't be decoded for inspection"
)
_TOOL_BLOCK_ESCALATION_REASON: Final = (
    "This message was blocked by Ovalix because tool call anonymization isn't possible via LiteLLM"
)
_TOOL_RESULT_BLOCK_ESCALATION_REASON: Final = (
    "This message was blocked by Ovalix because tool result anonymization isn't possible via LiteLLM"
)


def _encode_file_wire_format(raw: bytes) -> str:
    return base64.b64encode(gzip.compress(raw)).decode()


def _mapping_or_empty(value: object) -> Mapping[str, object]:
    return as_str_mapping(value) or _NO_METADATA


class _CustomGuardrailOptions(TypedDict, total=False, extra_items=object):
    """Base-class constructor options this guardrail forwards untouched to CustomGuardrail."""


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
            blocked_content=True,
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
        supported_event_hooks: list[GuardrailEventHooks] | None = None,
        **kwargs: Unpack[_CustomGuardrailOptions],
    ):
        self._tracker_api_base = tracker_api_base or os.environ.get("OVALIX_TRACKER_API_BASE")
        self._tracker_api_key = tracker_api_key or os.environ.get("OVALIX_TRACKER_API_KEY")
        self._application_id = application_id or os.environ.get("OVALIX_APPLICATION_ID")
        self._pre_checkpoint_id = pre_checkpoint_id or os.environ.get("OVALIX_PRE_CHECKPOINT_ID")
        self._post_checkpoint_id = post_checkpoint_id or os.environ.get("OVALIX_POST_CHECKPOINT_ID")
        self._file_checkpoint_id = file_checkpoint_id or os.environ.get("OVALIX_FILE_CHECKPOINT_ID")

        event_hooks: Final = self._validated_event_hooks(supported_event_hooks or ())

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

        super().__init__(supported_event_hooks=event_hooks, **kwargs)
        verbose_proxy_logger.debug(
            "Ovalix Guardrail initialized: tracker=%s, application_id=%s, pre_checkpoint_id=%s, post_checkpoint_id=%s",
            self._tracker_api_base,
            self._application_id,
            self._pre_checkpoint_id,
            self._post_checkpoint_id,
        )

    def _validated_event_hooks(self, requested: Sequence[GuardrailEventHooks]) -> list[GuardrailEventHooks]:
        """Ensure required secrets and checkpoint IDs are set; return the requested hooks plus one per configured checkpoint."""
        errors: Final = tuple(
            message
            for missing, message in (
                (not self._tracker_api_base, "Tracker API base, set OVALIX_TRACKER_API_BASE or pass tracker_api_base"),
                (not self._tracker_api_key, "Tracker API key, set OVALIX_TRACKER_API_KEY or pass tracker_api_key"),
                (not self._application_id, "Application ID, set OVALIX_APPLICATION_ID or pass application_id"),
                (
                    not self._pre_checkpoint_id and GuardrailEventHooks.pre_call in requested,
                    "Pre-checkpoint ID, set OVALIX_PRE_CHECKPOINT_ID or pass pre_checkpoint_id",
                ),
                (
                    not self._post_checkpoint_id and GuardrailEventHooks.post_call in requested,
                    "Post-checkpoint ID, set OVALIX_POST_CHECKPOINT_ID or pass post_checkpoint_id",
                ),
                (
                    not self._pre_checkpoint_id and not self._post_checkpoint_id,
                    "Pre-checkpoint ID or Post-checkpoint ID, set OVALIX_PRE_CHECKPOINT_ID or OVALIX_POST_CHECKPOINT_ID or pass pre_checkpoint_id or post_checkpoint_id",
                ),
            )
            if missing
        )

        if errors:
            raise OvalixGuardrailMissingSecrets("Missing Ovalix guardrail configuration errors: " + ". ".join(errors))

        auto_added: Final = tuple(
            hook
            for configured, hook in (
                (self._pre_checkpoint_id, GuardrailEventHooks.pre_call),
                (self._post_checkpoint_id, GuardrailEventHooks.post_call),
            )
            if configured and hook not in requested
        )
        return [*requested, *auto_added]

    def _get_actor(self, data: Mapping[str, object]) -> str:
        """Return a stable actor identifier from request metadata (e.g. user email or id)."""
        metadata: Final = _mapping_or_empty(data.get("metadata") or data.get("litellm_metadata"))
        return (
            str_or_none(metadata.get("user_api_key_user_email"))
            or str_or_none(metadata.get("user_api_key_user_id"))
            or "unknown"
        )

    def _get_tracker_actor_id(self, data: Mapping[str, object]) -> str:
        """Normalize the actor string into a short, stable id for Tracker API payloads."""
        # NOTE: this hash is purely for normalization — it collapses an arbitrary actor
        # string (email, user id, or "unknown") into a compact, fixed-length, consistent
        # key. It is not a privacy/security measure and the actor value is not sensitive,
        # so a plain SHA-256 (truncated) is sufficient; no salting/KDF is needed here.
        actor_id: Final = self._get_actor(data).encode()
        normalized_actor_id: Final = hashlib.sha256(actor_id).hexdigest()[:8]
        return normalized_actor_id

    def _get_session_id(self, data: Mapping[str, object]) -> str:
        """Return a unique identifier for the chat/session (actor + date + application_id)."""
        actor_hash: Final = self._get_tracker_actor_id(data)
        today: Final = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
        return f"{actor_hash}_{today}_{self._application_id}"

    async def _call_checkpoint(
        self,
        data_type: str,
        data: Mapping[str, object],
        checkpoint_id: str,
        actor: str,
        session_id: str,
        oversize: bool = False,
    ) -> Mapping[str, object]:
        """Call the Ovalix Tracker checkpoint API and return the JSON response.

        Both routes live on the tracker's /beta litellm router, which accepts the api key this
        guardrail already sends.
        """
        application_id: Final = self._application_id
        if not application_id or not checkpoint_id:
            raise ValueError("Ovalix: application_id or checkpoint_id not resolved")

        route: Final = "file_checkpoint" if data_type == "FILE" else "checkpoint"
        payload: Final = {
            "application_id": application_id,
            "checkpoint_id": checkpoint_id,
            "actor": actor,
            "session_id": session_id,
            "data_type": data_type,
            "data": data,
            "tool": "LiteLLM",
            **(MappingProxyType({"oversize": True}) if oversize else _NO_METADATA),
        }
        response: Final = await self._async_handler.post(
            f"{self._tracker_api_base}/tracking/beta/{route}",
            headers=self._tracker_headers,
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def _verdict(self, resp: Mapping[str, object]) -> tuple[str, str | None]:
        return (str_or_none(resp.get("action_type")) or "").lower(), self._get_trackers_corrected_message(resp)

    async def _block_reason_for_item(
        self,
        data_type: str,
        data: Mapping[str, object],
        checkpoint_id: str,
        actor: str,
        session_id: str,
        escalation_reason: str,
        oversize: bool = False,
    ) -> str | None:
        try:
            resp: Final = await self._call_checkpoint(
                data_type, data, checkpoint_id, actor, session_id, oversize=oversize
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
            return corrected or BLOCKED_BY_OVALIX_FALLBACK_MESSAGE
        if action in _MODIFY_ACTION_TYPES:
            return escalation_reason
        return None

    async def _check_items_block_only(
        self,
        items: Sequence[tuple[str, Mapping[str, object]]],
        checkpoint_id: str,
        actor: str,
        session_id: str,
        escalation_reason: str,
    ) -> str | None:
        for data_type, data in items:
            reason = await self._block_reason_for_item(
                data_type, data, checkpoint_id, actor, session_id, escalation_reason
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
    ) -> str | None:
        if any(part.undecodable for part in file_parts):
            return _UNDECODABLE_FILE_BLOCK_REASON
        for part in sorted(file_parts, key=lambda p: p.message_index, reverse=True):
            data = await self._file_part_to_data(part)
            reason = await self._block_reason_for_item(
                "FILE",
                data,
                checkpoint_id,
                actor,
                session_id,
                _FILE_BLOCK_ESCALATION_REASON,
                oversize=part.oversize,
            )
            if reason is not None:
                return reason
        return None

    @log_guardrail_information
    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: Mapping[str, object],
        input_type: Literal["request", "response"],
        logging_obj: "LiteLLMLoggingObj | None" = None,
    ) -> GenericGuardrailAPIInputs:
        actor: Final = self._get_actor(request_data)
        session_id: Final = self._get_session_id(request_data)
        is_response: Final = input_type == "response"

        prompt_checkpoint: Final = self._post_checkpoint_id if is_response else self._pre_checkpoint_id
        file_checkpoint: Final = self._file_checkpoint_id or prompt_checkpoint
        if not file_checkpoint:
            verbose_proxy_logger.debug(
                "Ovalix guardrail: application %s has no %s checkpoint, leaving this direction uninspected",
                self._application_id,
                input_type,
            )
            return inputs

        structured_messages: Final = inputs.get("structured_messages") or ()
        file_parts: Final = (
            extract_file_parts_from_images(inputs.get("images"), size_limit=_DEFAULT_FILE_SIZE_LIMIT)
            if is_response
            else extract_file_parts_from_messages(structured_messages, size_limit=_DEFAULT_FILE_SIZE_LIMIT)
        )
        file_block: Final = await self._check_files_for_block(file_parts, file_checkpoint, actor, session_id)
        if file_block is not None:
            self._block_current_message(file_block)

        if not prompt_checkpoint:
            verbose_proxy_logger.debug(
                "Ovalix guardrail: application %s has only a %s file checkpoint, skipping text and tool inspection",
                self._application_id,
                input_type,
            )
            return inputs

        tool_calls: Final = (
            *(inputs.get("tool_calls") or ()),
            *extract_tool_calls_from_messages(structured_messages),
        )
        unique_tool_data: Final = MappingProxyType(
            {tool_data_key(data): data for data in (tool_call_to_tool_data(tc) for tc in tool_calls) if data}
        )
        tool_call_items: Final = tuple(("TOOL", data) for data in unique_tool_data.values())
        tool_block: Final = await self._check_items_block_only(
            tool_call_items,
            prompt_checkpoint,
            actor,
            session_id,
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
            tool_result_text_indices(structured_messages, texts),
        )
        if output_texts is None:
            return inputs
        return {**inputs, "texts": output_texts}

    async def _file_part_to_data(self, part: FilePart) -> Mapping[str, object]:
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
        skip_indices: frozenset[int],
    ) -> list[str] | None:
        newest_first: Final = tuple(index for index in reversed(range(len(texts))) if index not in skip_indices)
        checked: Final = MappingProxyType(
            {
                index: await self._check_text(
                    texts[index], checkpoint_id, actor, session_id, is_newest=index == len(texts) - 1
                )
                for index in newest_first
            }
        )
        output: Final = [checked.get(index, text) for index, text in enumerate(texts)]
        return output if output != list(texts) else None

    async def _check_text(
        self,
        content: str,
        checkpoint_id: str,
        actor: str,
        session_id: str,
        is_newest: bool,
    ) -> str:
        """Return the text that replaces `content` after Ovalix checks it; a blocked newest text raises."""
        try:
            resp: Final = await self._call_checkpoint("TEXT", {"content": content}, checkpoint_id, actor, session_id)
        except Exception as e:
            verbose_proxy_logger.exception("Ovalix checkpoint call failed: %s", e)
            raise GuardrailRaisedException(
                guardrail_name=self.guardrail_name,
                message=f"Ovalix guardrail error: {e}",
                should_wrap_with_default_message=False,
            ) from e
        action, corrected = self._verdict(resp)
        if action == BLOCKED_ACTION_TYPE:
            block_message: Final = corrected or BLOCKED_BY_OVALIX_FALLBACK_MESSAGE
            if is_newest:
                self._block_current_message(block_message)
            return block_message
        if action in _MODIFY_ACTION_TYPES and corrected is not None:
            return corrected
        return content

    def _block_current_message(self, blocking_message: str) -> None:
        """Raise OvalixGuardrailBlockedException with the given message (no default wrapper)."""
        raise OvalixGuardrailBlockedException(
            guardrail_name=self.guardrail_name,
            message=blocking_message,
            should_wrap_with_default_message=False,
        )

    def _get_trackers_corrected_message(self, resp: Mapping[str, object]) -> str | None:
        """Extract corrected/blocking message content from Tracker checkpoint response."""
        modified: Final = as_str_mapping(resp.get("modified_data"))
        if modified is None:
            return None
        content: Final = modified.get("content")
        return content if isinstance(content, str) else None

    @staticmethod
    def get_config_model() -> type["GuardrailConfigModel"] | None:
        from litellm.types.proxy.guardrails.guardrail_hooks.ovalix import (
            OvalixGuardrailConfigModel,
        )

        return OvalixGuardrailConfigModel
