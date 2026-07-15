# +-------------------------------------------------------------+
#
#           Use Bedrock Guardrails for your LLM calls
#
# +-------------------------------------------------------------+
#  Thank you users! We ❤️ you! - Krrish & Ishaan

import os
import sys

sys.path.insert(0, os.path.abspath("../.."))  # Adds the parent directory to the system path
import json
import sys
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    ClassVar,
    Dict,
    List,
    Literal,
    NamedTuple,
    Optional,
    Tuple,
    Union,
    cast,
)

import copy
from collections.abc import Mapping
from datetime import datetime, timezone

import httpx
from fastapi import HTTPException
from pydantic import TypeAdapter, ValidationError

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.core_helpers import redact_nested_match_and_regex_keys
from litellm.caching import DualCache
from litellm.exceptions import ModifyResponseException
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.proxy._types import UserAPIKeyAuth
from litellm.secret_managers.main import get_secret_str
from litellm.types.guardrails import BedrockChecksConfigModel, GuardrailEventHooks
from litellm.types.llms.openai import AllMessageValues, ChatCompletionUserMessage
from litellm.types.proxy.guardrails.guardrail_hooks.bedrock_guardrails import (
    BedrockChecksMessage,
    BedrockChecksViolation,
    BedrockContentItem,
    BedrockGuardrailChecksResponse,
    BedrockGuardrailOutput,
    BedrockGuardrailQualifier,
    BedrockGuardrailResponse,
    BedrockRequest,
    BedrockTextContent,
)
from litellm.types.utils import GenericGuardrailAPIInputs

if TYPE_CHECKING:
    from botocore.awsrequest import AWSPreparedRequest

    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj

from litellm.types.utils import (
    CallTypes,
    CallTypesLiteral,
    Choices,
    GuardrailStatus,
    GuardrailTracingDetail,
    Message,
    ModelResponse,
    ModelResponseStream,
    StreamingChoices,
    TextChoices,
)

GUARDRAIL_NAME = "bedrock"
_BEDROCK_DYNAMIC_BODY_DENYLIST = frozenset({"content", "source"})
# Resource-less, detect-only InvokeGuardrailChecks API (no guardrail resource required).
_BEDROCK_INVOKE_GUARDRAIL_CHECKS_PATH = "/guardrail-checks/invoke"
# InvokeGuardrailChecks accepts at most 10 content blocks per message. A message with
# more text blocks is split across multiple messages so ALL content is scanned --
# never truncated (truncation would let a user hide content past the limit).
_BEDROCK_CHECKS_MAX_CONTENT_BLOCKS = 10
_BEDROCK_CHECKS_KNOWN_KEYS = frozenset({"contentFilter", "promptAttack", "sensitiveInformation"})
# Keys in a sensitiveInformation result that pinpoint the PII location. They are
# stripped before the response is handed to standard logging / telemetry so the
# detected PII span cannot be reconstructed from logs.
_BEDROCK_CHECKS_PII_LOCATION_KEYS = (
    "beginOffset",
    "endOffset",
    "messageIndex",
    "contentIndex",
)

# Maps an OpenAI message content-block ``type`` to the Bedrock guardrail qualifier
# it represents, so callers can drive contextual grounding by tagging their content.
# The model response is qualified as ``guard_content`` directly by the OUTPUT builder;
# the existing ``guarded_text`` marker is intentionally left unmapped here so its
# guardrail-hook payload is unchanged by this feature.
_CONTENT_TYPE_TO_QUALIFIER: Dict[str, BedrockGuardrailQualifier] = {
    "grounding_source": "grounding_source",
    "query": "query",
}

# Roles whose ``grounding_source`` blocks are trusted as reference material for the
# contextual-grounding check. Only app-authored roles qualify: ``tool``/``function``
# results and ``user`` content can carry caller- or externally-influenced text, which
# must not be graded against as if it were the application's own source material.
_GROUNDING_SOURCE_TRUSTED_ROLES = frozenset({"system", "developer"})


class QualifiedTextBlock(NamedTuple):
    """A piece of message text paired with its Bedrock grounding qualifier (if any)."""

    text: str
    qualifier: Optional[BedrockGuardrailQualifier]


class GuardrailMessageFilterResult(NamedTuple):
    payload_messages: Optional[List[AllMessageValues]]
    original_messages: Optional[List[AllMessageValues]]
    target_indices: Optional[List[int]]


class ApplyGuardrailMessageSelection(NamedTuple):
    """Messages selected for an apply_guardrail scan + write-back metadata."""

    filtered_messages: Optional[list[AllMessageValues]]
    # Slice of the flat `texts` list actually scanned (offset, length),
    # used to write masked content back to the right positions. None = whole list.
    scanned_slice: Optional[tuple[int, int]]
    # True when messages were selected by their original role.
    scanned_role_subset: bool
    # True when there is nothing to scan (e.g. no user-role message).
    skip_scan: bool = False


def _redact_pii_matches(response_json: dict) -> dict:
    """
    Redact match-like fields from a Bedrock ApplyGuardrail JSON payload.

    Delegates to :func:`redact_nested_match_and_regex_keys` (same rules as spend
    logging). Kept as a Bedrock-module entry point for existing unit tests.
    """
    redacted = redact_nested_match_and_regex_keys(response_json)
    return redacted if isinstance(redacted, dict) else response_json


def _redact_assessment_match_fields(assessments: List[dict]) -> List[dict]:
    """
    Redact sensitive match-like fields from blocked assessment summaries.

    This is used for customer-visible error payloads (HTTPException.detail) where
    we want to preserve policy/type/action metadata without echoing raw matched
    content.
    """
    redacted = redact_nested_match_and_regex_keys(assessments)
    return redacted if isinstance(redacted, list) else assessments


class BedrockGuardrail(CustomGuardrail, BaseAWSLLM):
    # During-call must use async_moderation_hook (not unified apply_guardrail), otherwise
    # OpenAI translation always passes input_type="request" and spend/UI show PRE-CALL.
    use_native_during_call_hook: ClassVar[bool] = True

    def __init__(
        self,
        guardrailIdentifier: Optional[str] = None,
        guardrailVersion: Optional[str] = None,
        disable_exception_on_block: Optional[bool] = False,
        checks: BedrockChecksConfigModel | Mapping[str, object] | None = None,
        content_filter_threshold: float | None = 0.5,
        prompt_attack_threshold: float | None = 0.5,
        pii_confidence_threshold: float | None = 0.5,
        **kwargs,
    ):
        self.async_handler = get_async_httpx_client(llm_provider=httpxSpecialProvider.GuardrailCallback)
        self.guardrailIdentifier = guardrailIdentifier
        self.guardrailVersion = guardrailVersion
        self.guardrail_provider = "bedrock"
        self.experimental_use_latest_role_message_only = bool(kwargs.get("experimental_use_latest_role_message_only"))

        # Resource-less, detect-only InvokeGuardrailChecks mode. Present `checks`
        # routes the guardrail to InvokeGuardrailChecks; absent => ApplyGuardrail.
        self.checks: dict[str, Any] | None = self._normalize_checks(checks)
        # Per-check block thresholds; a score >= threshold blocks. None => the
        # check is detect-only (logged, never blocks).
        self.content_filter_threshold = content_filter_threshold
        self.prompt_attack_threshold = prompt_attack_threshold
        self.pii_confidence_threshold = pii_confidence_threshold

        # store kwargs as optional_params
        self.optional_params = kwargs

        self.disable_exception_on_block: bool = disable_exception_on_block or False
        """
        If True, will not raise an exception when the guardrail is blocked.
        """

        # `checks` (InvokeGuardrailChecks) and `guardrailIdentifier`/`guardrailVersion`
        # (ApplyGuardrail) are two different APIs; configuring both is ambiguous.
        if self.checks is not None and (self.guardrailIdentifier is not None or self.guardrailVersion is not None):
            raise ValueError(
                "Bedrock guardrail accepts either 'guardrailIdentifier'/'guardrailVersion' (ApplyGuardrail) "
                "or 'checks' (InvokeGuardrailChecks), not both."
            )

        # Set supported event hooks to include MCP hooks
        kwargs.setdefault("supported_event_hooks", list(self.get_supported_event_hooks()))

        super().__init__(**kwargs)
        BaseAWSLLM.__init__(self)

        # InvokeGuardrailChecks is detect-only: it never returns rewritten content,
        # so masking has no effect in checks mode.
        if self.checks is not None and (
            getattr(self, "mask_request_content", False) or getattr(self, "mask_response_content", False)
        ):
            verbose_proxy_logger.warning(
                "Bedrock Guardrail: mask_request_content/mask_response_content have no "
                "effect with 'checks' (InvokeGuardrailChecks is detect-only)."
            )

        verbose_proxy_logger.debug(
            "Bedrock Guardrail initialized with guardrailIdentifier: %s, guardrailVersion: %s, checks: %s",
            self.guardrailIdentifier,
            self.guardrailVersion,
            list(self.checks.keys()) if self.checks else None,
        )

    @classmethod
    def get_supported_event_hooks(cls) -> List[GuardrailEventHooks]:
        return [
            GuardrailEventHooks.pre_call,
            GuardrailEventHooks.post_call,
            GuardrailEventHooks.during_call,
            GuardrailEventHooks.pre_mcp_call,
            GuardrailEventHooks.during_mcp_call,
        ]

    @staticmethod
    def _normalize_checks(checks: BedrockChecksConfigModel | Mapping[str, object] | None) -> dict[str, Any] | None:
        """Normalize the configured `checks` into a plain dict for the API body.

        Accepts a pydantic ``BedrockChecksConfigModel`` or a raw dict; drops None /
        unknown keys. Returns None when no usable check is configured (=> ApplyGuardrail).
        """
        if checks is None:
            return None
        raw = checks.model_dump(exclude_none=True) if isinstance(checks, BedrockChecksConfigModel) else dict(checks)
        unknown_keys = set(raw.keys()) - _BEDROCK_CHECKS_KNOWN_KEYS
        if unknown_keys:
            verbose_proxy_logger.warning(
                "BedrockGuardrail: unrecognized check key(s) %s will be ignored; "
                "recognized keys will still be used for InvokeGuardrailChecks. "
                "Known keys: %s.",
                sorted(unknown_keys),
                sorted(_BEDROCK_CHECKS_KNOWN_KEYS),
            )
        cleaned = {key: value for key, value in raw.items() if key in _BEDROCK_CHECKS_KNOWN_KEYS and value is not None}
        if not cleaned and raw:
            raise ValueError(
                f"BedrockGuardrail: 'checks' block contained only unrecognized or empty keys {sorted(raw.keys())}. "
                f"Known keys: {sorted(_BEDROCK_CHECKS_KNOWN_KEYS)}. "
                "Fix the guardrail config or remove the 'checks' block to use ApplyGuardrail mode."
            )
        return cleaned or None

    def _create_bedrock_input_content_request(self, messages: Optional[List[AllMessageValues]]) -> BedrockRequest:
        """
        Create a bedrock request for the input content - the LLM request.
        """
        bedrock_request: BedrockRequest = BedrockRequest(source="INPUT")
        bedrock_request_content: List[BedrockContentItem] = []
        if messages is None:
            return bedrock_request
        for message in messages:
            blocks = self.get_content_items_for_message(message=message)
            if blocks is None:
                continue
            for block in blocks:
                # INPUT scans send plain text only. Grounding qualifiers are attached
                # exclusively when assembling the OUTPUT request, so a caller cannot use
                # a grounding_source/query tag to change how input-safety policies treat
                # their content (which would be an input-guardrail bypass).
                bedrock_request_content.append(BedrockContentItem(text=BedrockTextContent(text=block.text)))

        bedrock_request["content"] = bedrock_request_content
        return bedrock_request

    def _create_bedrock_output_content_request(
        self,
        response: Union[Any, ModelResponse],
        messages: Optional[List[AllMessageValues]] = None,
    ) -> BedrockRequest:
        """
        Create a bedrock request for the output content - the LLM response.

        Contextual grounding grades the response against the reference source and
        the user query from the request. When the request tagged any
        ``grounding_source``/``query`` blocks, they are emitted first and the
        response is qualified as ``guard_content`` so Bedrock can score grounding.
        Without such tags the payload is the legacy single response block.
        """
        bedrock_request: BedrockRequest = BedrockRequest(source="OUTPUT")
        grounding_blocks = self._collect_grounding_blocks(messages)
        bedrock_request_content: List[BedrockContentItem] = [
            self._build_content_item(block) for block in grounding_blocks
        ]
        has_grounding = len(bedrock_request_content) > 0
        # Append the response (the content to guard) after any grounding blocks; assign
        # unconditionally so harvested grounding blocks survive a non-ModelResponse input.
        bedrock_request_content.extend(self._build_response_content_items(response, has_grounding=has_grounding))
        bedrock_request["content"] = bedrock_request_content
        return bedrock_request

    def _build_response_content_items(
        self, response: Union[Any, ModelResponse], has_grounding: bool
    ) -> List[BedrockContentItem]:
        """Build content item(s) from the model response. When the request supplied
        grounding, the response is qualified ``guard_content`` so Bedrock can score it.
        """
        items: List[BedrockContentItem] = []
        if not isinstance(response, litellm.ModelResponse):
            return items
        for choice in response.choices:
            if (
                isinstance(choice, litellm.Choices)
                and isinstance(choice.message.content, str)
                and choice.message.content
            ):
                block = QualifiedTextBlock(
                    text=choice.message.content,
                    qualifier="guard_content" if has_grounding else None,
                )
                items.append(self._build_content_item(block))
        return items

    def convert_to_bedrock_format(
        self,
        source: Literal["INPUT", "OUTPUT"],
        messages: Optional[List[AllMessageValues]] = None,
        response: Optional[Union[Any, ModelResponse]] = None,
    ) -> BedrockRequest:
        """
        Convert the litellm messages/response to the bedrock request format.

        If source is "INPUT", then messages is required.
        If source is "OUTPUT", then response is required.

        Returns:
            BedrockRequest: The bedrock request object.
        """
        bedrock_request: BedrockRequest = BedrockRequest(source=source)
        if source == "INPUT":
            bedrock_request = self._create_bedrock_input_content_request(messages=messages)
        elif source == "OUTPUT":
            bedrock_request = self._create_bedrock_output_content_request(response=response, messages=messages)
        return bedrock_request

    def get_content_items_for_message(self, message: AllMessageValues) -> Optional[List[QualifiedTextBlock]]:
        """
        Flatten a message into text blocks, preserving any contextual-grounding
        qualifier carried by the content-block ``type`` (grounding_source / query).
        Untagged text keeps ``qualifier=None`` so the payload is unchanged for
        callers that do not use grounding.
        """
        content = message.get("content")
        if content is None:
            return None
        blocks: List[QualifiedTextBlock] = []
        if isinstance(content, str):
            blocks.append(QualifiedTextBlock(text=content, qualifier=None))
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and "text" in item:
                    qualifier = _CONTENT_TYPE_TO_QUALIFIER.get(item.get("type", ""))
                    blocks.append(QualifiedTextBlock(text=item["text"], qualifier=qualifier))
                elif isinstance(item, str):
                    blocks.append(QualifiedTextBlock(text=item, qualifier=None))
        return blocks

    def _build_content_item(self, block: QualifiedTextBlock) -> BedrockContentItem:
        """Build a Bedrock content item, attaching qualifiers only when present."""
        text_content = BedrockTextContent(text=block.text)
        if block.qualifier is not None:
            text_content["qualifiers"] = [block.qualifier]
        return BedrockContentItem(text=text_content)

    def _collect_grounding_blocks(self, messages: Optional[List[AllMessageValues]]) -> List[QualifiedTextBlock]:
        """Harvest grounding_source/query blocks from the request for an OUTPUT scan.

        ``grounding_source`` is honored only from app-authored roles (system /
        developer). A grounding_source tag on a ``user``, ``tool`` or ``function``
        message is ignored, so neither a forwarded end-user message nor a tool/function
        result carrying externally-influenced content can supply fake evidence for the
        contextual-grounding check to grade the response against. ``query`` is accepted
        from any role (it is the user's question).
        """
        grounding: List[QualifiedTextBlock] = []
        for message in messages or []:
            role = message.get("role")
            for block in self.get_content_items_for_message(message=message) or []:
                if block.qualifier == "query":
                    grounding.append(block)
                elif block.qualifier == "grounding_source" and role in _GROUNDING_SOURCE_TRUSTED_ROLES:
                    grounding.append(block)
        return grounding

    def _prepare_guardrail_messages_for_role(
        self,
        messages: Optional[List[AllMessageValues]],
    ) -> GuardrailMessageFilterResult:
        """Return payload + merge metadata for the latest user message."""
        # NOTE: This logic probably belongs in CustomGuardrail once other guardrails adopt the feature.

        if messages is None:
            return GuardrailMessageFilterResult(None, None, None)

        if self.experimental_use_latest_role_message_only is not True:
            return GuardrailMessageFilterResult(messages, None, None)

        latest_index = self._find_latest_message_index(messages, target_role="user")
        if latest_index is None:
            return GuardrailMessageFilterResult(None, None, None)

        original_messages = list(messages)
        payload_messages = [messages[latest_index]]
        return GuardrailMessageFilterResult(
            payload_messages=payload_messages,
            original_messages=original_messages,
            target_indices=[latest_index],
        )

    def _find_latest_message_index(self, messages: List[AllMessageValues], target_role: str) -> Optional[int]:
        for index in range(len(messages) - 1, -1, -1):
            if messages[index].get("role", None) == target_role:
                return index
        return None

    @staticmethod
    def _count_message_texts(message: AllMessageValues) -> int:
        """Count the text segments the guardrail translation layer extracts from a message."""
        content = message.get("content")
        if isinstance(content, str):
            return 1
        if isinstance(content, list):
            return sum(1 for item in content if isinstance(item, dict) and item.get("text") is not None)
        return 0

    def _locate_message_texts_slice(
        self,
        structured_messages: list[AllMessageValues],
        target_index: int,
        texts: list[str],
    ) -> Optional[tuple[int, int]]:
        """
        Map one message's text segments to their (offset, length) slice in the
        flat `texts` list built by the guardrail translation handler.

        Returns None when the reconstruction does not line up with `texts`
        (the caller must then avoid positional write-back).
        """
        offset = 0
        total = 0
        target_count = 0
        for index, message in enumerate(structured_messages):
            count = self._count_message_texts(message)
            if index < target_index:
                offset += count
            elif index == target_index:
                target_count = count
            total += count
        if total != len(texts) or target_count == 0:
            return None
        return offset, target_count

    def _select_messages_for_apply_guardrail(
        self,
        texts: list[str],
        inputs: "GenericGuardrailAPIInputs",
        request_data: dict,
        input_type: Literal["request", "response"],
    ) -> ApplyGuardrailMessageSelection:
        """
        Decide which messages an apply_guardrail scan should cover.

        With ``experimental_use_latest_role_message_only`` enabled, request
        scans must select by the ORIGINAL message roles. The flat `texts` list
        has no role information, and wrapping it in role="user" mock messages
        makes the latest-user filter degenerate to "latest text of any role",
        leaking tool/assistant content to the INPUT scan
        (https://github.com/BerriAI/litellm/issues/23476).
        """
        mock_messages: list[AllMessageValues] = [ChatCompletionUserMessage(role="user", content=text) for text in texts]

        if self.experimental_use_latest_role_message_only is not True:
            return ApplyGuardrailMessageSelection(
                filtered_messages=mock_messages,
                scanned_slice=None,
                scanned_role_subset=False,
            )

        # Prefer inputs["structured_messages"]: it is built alongside `texts` by
        # the translation handler and stays aligned with it even when
        # skip_system_message_in_guardrail / skip_tool_message_in_guardrail drop
        # messages. The fallback to request_data["messages"] is the *unfiltered*
        # list, so it only lines up with `texts` when no skip flags are active.
        # When a skip flag is set and we land on this fallback (direct
        # apply_guardrail callers with no structured_messages),
        # _locate_message_texts_slice will detect the length mismatch and return
        # None, and the write-back guard below safely skips masking rather than
        # corrupting positions.
        structured_messages = cast(
            Optional[list[AllMessageValues]],
            inputs.get("structured_messages") or request_data.get("messages"),
        )
        if input_type != "request" or not structured_messages:
            # No role information available (e.g. raw-text callers like
            # /guardrails/apply_guardrail) — keep the legacy behavior of
            # scanning the latest text only.
            filter_result = self._prepare_guardrail_messages_for_role(messages=mock_messages)
            return ApplyGuardrailMessageSelection(
                filtered_messages=filter_result.payload_messages or mock_messages,
                scanned_slice=None,
                scanned_role_subset=False,
            )

        latest_user_index = self._find_latest_message_index(structured_messages, target_role="user")
        if latest_user_index is None:
            verbose_proxy_logger.debug("Bedrock Guardrail: no user-role message in request, skipping INPUT scan")
            return ApplyGuardrailMessageSelection(None, None, True, skip_scan=True)

        selected_message = structured_messages[latest_user_index]
        if self._count_message_texts(selected_message) == 0:
            verbose_proxy_logger.debug(
                "Bedrock Guardrail: latest user message has no text content, skipping INPUT scan"
            )
            return ApplyGuardrailMessageSelection(None, None, True, skip_scan=True)

        return ApplyGuardrailMessageSelection(
            filtered_messages=[selected_message],
            scanned_slice=self._locate_message_texts_slice(
                structured_messages=structured_messages,
                target_index=latest_user_index,
                texts=texts,
            ),
            scanned_role_subset=True,
        )

    def _merge_masked_texts(
        self,
        masked_texts: list,
        texts: list,
        scanned_slice: Optional[tuple[int, int]],
        scanned_role_subset: bool,
    ) -> list:
        """
        Reconcile the guardrail's masked output with the flat `texts` list.

        - No masked output: keep the originals (guardrail allowed content as-is).
        - A slice was scanned: write masked content back to those positions only,
          keeping the list aligned with the caller's message↔text mappings.
        - A role-selected subset was scanned but could not be mapped back to
          flat-text positions (scanned_slice is None): keep the originals rather
          than misapply masked content to the wrong message. Guarding on
          scanned_slice rather than a length comparison also covers the case
          where the masked subset happens to match len(texts) (e.g. both length
          1).
        - Otherwise (whole list scanned): use the masked output as-is.
        """
        if not masked_texts:
            return texts
        if scanned_slice is not None:
            offset, length = scanned_slice
            merged_texts = list(texts)
            for masked_index, masked_text in enumerate(masked_texts[:length]):
                merged_texts[offset + masked_index] = masked_text
            return merged_texts
        if scanned_role_subset:
            verbose_proxy_logger.warning(
                "Bedrock Guardrail: could not align masked texts with request texts, skipping masking write-back"
            )
            return texts
        return masked_texts

    def _merge_filtered_messages(
        self,
        original_messages: Optional[List[AllMessageValues]],
        updated_target_messages: List[AllMessageValues],
        target_indices: Optional[List[int]],
    ) -> List[AllMessageValues]:
        if not target_indices:
            return updated_target_messages

        if not original_messages:
            original_messages = []

        merged_messages = list(original_messages)
        if not merged_messages:
            merged_messages = list(updated_target_messages)
        for replacement_index, updated_message in zip(target_indices, updated_target_messages):
            if replacement_index < len(merged_messages):
                merged_messages[replacement_index] = updated_message

        return merged_messages

    # NOTE: Consider moving these helpers to CustomGuardrail when the filtering
    # logic becomes shared across providers.

    #### CALL HOOKS - proxy only ####
    def _load_credentials(
        self,
    ):
        try:
            from botocore.credentials import Credentials
        except ImportError:
            raise ImportError("Missing boto3 to call bedrock. Run 'pip install boto3'.")
        ## CREDENTIALS ##
        aws_secret_access_key = self.optional_params.get("aws_secret_access_key", None)
        aws_access_key_id = self.optional_params.get("aws_access_key_id", None)
        aws_session_token = self.optional_params.get("aws_session_token", None)
        aws_region_name = self.optional_params.get("aws_region_name", None)
        aws_role_name = self.optional_params.get("aws_role_name", None)
        aws_session_name = self.optional_params.get("aws_session_name", None)
        aws_profile_name = self.optional_params.get("aws_profile_name", None)
        aws_web_identity_token = self.optional_params.get("aws_web_identity_token", None)
        aws_sts_endpoint = self.optional_params.get("aws_sts_endpoint", None)

        ### SET REGION NAME ###
        aws_region_name = self.get_aws_region_name_for_non_llm_api_calls(
            aws_region_name=aws_region_name,
        )

        credentials: Credentials = self.get_credentials(
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token,
            aws_region_name=aws_region_name,
            aws_session_name=aws_session_name,
            aws_profile_name=aws_profile_name,
            aws_role_name=aws_role_name,
            aws_web_identity_token=aws_web_identity_token,
            aws_sts_endpoint=aws_sts_endpoint,
        )
        return credentials, aws_region_name

    def _prepare_request(
        self,
        credentials,
        data: dict,
        optional_params: dict,
        aws_region_name: str,
        api_key: Optional[str] = None,
        extra_headers: Optional[dict] = None,
        request_path: str | None = None,
    ):
        headers = {"Content-Type": "application/json"}
        if extra_headers is not None:
            headers = {"Content-Type": "application/json", **extra_headers}

        aws_bedrock_runtime_endpoint = self.optional_params.get("aws_bedrock_runtime_endpoint", None)
        _, proxy_endpoint_url = self.get_runtime_endpoint(
            api_base=None,
            aws_bedrock_runtime_endpoint=aws_bedrock_runtime_endpoint,
            aws_region_name=aws_region_name,
        )
        # Default to the ApplyGuardrail resource path. Callers pass an explicit
        # request_path for the resource-less InvokeGuardrailChecks endpoint (where
        # guardrailIdentifier/guardrailVersion are None and must not be interpolated).
        if request_path is None:
            request_path = f"/guardrail/{self.guardrailIdentifier}/version/{self.guardrailVersion}/apply"
        proxy_endpoint_url = f"{proxy_endpoint_url}{request_path}"
        encoded_data = json.dumps(data).encode("utf-8")

        # first check api-key, if none, fall back to sigV4
        if api_key is not None:
            aws_bearer_token: Optional[str] = api_key
        else:
            aws_bearer_token = get_secret_str("AWS_BEARER_TOKEN_BEDROCK")

        if aws_bearer_token:
            try:
                from botocore.awsrequest import AWSRequest
            except ImportError:
                raise ImportError("Missing boto3 to call bedrock. Run 'pip install boto3'.")
            headers["Authorization"] = f"Bearer {aws_bearer_token}"
            request = AWSRequest(
                method="POST",
                url=proxy_endpoint_url,
                data=encoded_data,
                headers=headers,
            )
        else:
            try:
                from botocore.auth import SigV4Auth
                from botocore.awsrequest import AWSRequest
            except ImportError:
                raise ImportError("Missing boto3 to call bedrock. Run 'pip install boto3'.")

            sigv4 = SigV4Auth(credentials, "bedrock", aws_region_name)
            request = AWSRequest(
                method="POST",
                url=proxy_endpoint_url,
                data=encoded_data,
                headers=headers,
            )
            sigv4.add_auth(request)
            if (
                extra_headers is not None and "Authorization" in extra_headers
            ):  # prevent sigv4 from overwriting the auth header
                request.headers["Authorization"] = extra_headers["Authorization"]
        prepped_request = request.prepare()

        return prepped_request

    async def make_bedrock_api_request(
        self,
        source: Literal["INPUT", "OUTPUT"],
        messages: list[AllMessageValues] | None = None,
        response: litellm.ModelResponse | None = None,
        request_data: dict | None = None,
        logging_event_type: GuardrailEventHooks | None = None,
    ) -> BedrockGuardrailResponse:
        """Dispatch to the configured Bedrock guardrail API.

        ``checks`` selects the resource-less, detect-only InvokeGuardrailChecks API;
        otherwise the ApplyGuardrail API is used. Both return a ``BedrockGuardrailResponse``
        (the checks path returns an empty one on a pass, which downstream masking treats
        as a no-op) and raise on a blocked request.
        """
        if self.checks is not None:
            return await self._make_invoke_guardrail_checks_request(
                source=source,
                messages=messages,
                response=response,
                request_data=request_data,
                logging_event_type=logging_event_type,
            )
        return await self._make_apply_guardrail_request(
            source=source,
            messages=messages,
            response=response,
            request_data=request_data,
            logging_event_type=logging_event_type,
        )

    async def _make_apply_guardrail_request(
        self,
        source: Literal["INPUT", "OUTPUT"],
        messages: list[AllMessageValues] | None = None,
        response: litellm.ModelResponse | None = None,
        request_data: dict | None = None,
        logging_event_type: GuardrailEventHooks | None = None,
    ) -> BedrockGuardrailResponse:
        start_time = datetime.now(timezone.utc)
        credentials, aws_region_name = self._load_credentials()
        bedrock_request_data: dict = dict(
            self.convert_to_bedrock_format(source=source, messages=messages, response=response)
        )
        bedrock_guardrail_response: BedrockGuardrailResponse = BedrockGuardrailResponse()
        api_key: Optional[str] = None
        if request_data:
            dynamic_request_body_params = self.get_guardrail_dynamic_request_body_params(request_data=request_data)
            bedrock_request_data.update(
                {
                    key: value
                    for key, value in dynamic_request_body_params.items()
                    if key not in _BEDROCK_DYNAMIC_BODY_DENYLIST
                }
            )
            if request_data.get("api_key") is not None:
                api_key = request_data["api_key"]

        prepared_request = self._prepare_request(
            credentials=credentials,
            data=bedrock_request_data,
            optional_params=self.optional_params,
            aws_region_name=aws_region_name,
            api_key=api_key,
        )
        verbose_proxy_logger.debug(
            "Bedrock AI request body: %s, url %s, headers: %s",
            bedrock_request_data,
            prepared_request.url,
            prepared_request.headers,
        )

        # UI / spend logs use event_type. Bedrock's `source` is INPUT vs OUTPUT for the API
        # body, which must not be confused with the proxy hook (pre_call / during_call /
        # post_call). When omitted, keep legacy mapping for backward compatibility.
        if logging_event_type is not None:
            event_type = logging_event_type
        else:
            event_type = GuardrailEventHooks.pre_call if source == "INPUT" else GuardrailEventHooks.post_call

        httpx_response = await self._sign_and_post(
            prepared_request=prepared_request,
            request_data=request_data,
            event_type=event_type,
            start_time=start_time,
        )

        #########################################################
        # Add guardrail information to request trace
        #########################################################
        _json_response = httpx_response.json()
        tracing_detail = self._build_tracing_detail(_json_response)

        # Raw Bedrock JSON is passed here; match/regex redaction runs once inside
        # CustomGuardrail.add_standard_logging_guardrail_information_to_request_data.
        self.add_standard_logging_guardrail_information_to_request_data(
            guardrail_provider=self.guardrail_provider,
            guardrail_json_response=_json_response,
            request_data=request_data or {},
            guardrail_status=self._get_bedrock_guardrail_response_status(response=httpx_response),
            start_time=start_time.timestamp(),
            end_time=datetime.now(timezone.utc).timestamp(),
            duration=(datetime.now(timezone.utc) - start_time).total_seconds(),
            event_type=event_type,
            tracing_detail=tracing_detail or None,
        )
        #########################################################
        if httpx_response.status_code == 200:
            # check if the response was flagged
            verbose_proxy_logger.debug(
                "Bedrock AI response : %s",
                redact_nested_match_and_regex_keys(_json_response),
            )
            bedrock_guardrail_response = BedrockGuardrailResponse(**_json_response)
            if self._should_raise_guardrail_blocked_exception(bedrock_guardrail_response):
                raise self._get_http_exception_for_blocked_guardrail(
                    bedrock_guardrail_response, request_data=request_data
                )
        else:
            status_code, detail_message = self._parse_bedrock_guardrail_error_response(httpx_response)
            verbose_proxy_logger.error(
                "Bedrock AI: error in response. Status code: %s, response: %s",
                httpx_response.status_code,
                httpx_response.text,
            )
            raise HTTPException(status_code=status_code, detail=detail_message)

        return bedrock_guardrail_response

    async def _sign_and_post(
        self,
        prepared_request: "AWSPreparedRequest",
        request_data: dict | None,
        event_type: GuardrailEventHooks,
        start_time: "datetime",
    ) -> httpx.Response:
        """POST a signed Bedrock request, logging+raising on network/HTTP errors.

        Shared by both the ApplyGuardrail and InvokeGuardrailChecks paths so their
        transport-error handling cannot drift. Returns the raw ``httpx.Response`` on
        success (including non-2xx that httpx did not raise on); the 200-path logging,
        status and tracing stay with each caller because the two APIs report differently.
        """
        try:
            return await self.async_handler.post(
                url=prepared_request.url,
                data=prepared_request.body,
                headers=dict(prepared_request.headers),
            )
        except HTTPException:
            # Propagate HTTPException (e.g. from non-200 path) as-is
            raise
        except Exception as e:
            # If this is an HTTP error with a response body (e.g. httpx.HTTPStatusError),
            # extract the AWS error message and propagate it
            err_response = getattr(e, "response", None)
            if isinstance(err_response, httpx.Response):
                try:
                    (
                        status_code,
                        detail_message,
                    ) = self._parse_bedrock_guardrail_error_response(err_response)
                    self.add_standard_logging_guardrail_information_to_request_data(
                        guardrail_provider=self.guardrail_provider,
                        guardrail_json_response={"error": detail_message},
                        request_data=request_data or {},
                        guardrail_status="guardrail_failed_to_respond",
                        start_time=start_time.timestamp(),
                        end_time=datetime.now(timezone.utc).timestamp(),
                        duration=(datetime.now(timezone.utc) - start_time).total_seconds(),
                        event_type=event_type,
                    )
                    raise HTTPException(status_code=status_code, detail=detail_message) from e
                except HTTPException:
                    raise
            # Endpoint down, timeout, or other HTTP/network errors
            verbose_proxy_logger.error("Bedrock AI: failed to make guardrail request: %s", str(e))
            self.add_standard_logging_guardrail_information_to_request_data(
                guardrail_provider=self.guardrail_provider,
                guardrail_json_response={"error": str(e)},
                request_data=request_data or {},
                guardrail_status="guardrail_failed_to_respond",
                start_time=start_time.timestamp(),
                end_time=datetime.now(timezone.utc).timestamp(),
                duration=(datetime.now(timezone.utc) - start_time).total_seconds(),
                event_type=event_type,
            )
            raise

    ###########  InvokeGuardrailChecks (resource-less, detect-only) ############

    @staticmethod
    def _chunk_texts_into_checks_messages(
        role: Literal["user", "assistant", "system"], texts: list[str]
    ) -> list[BedrockChecksMessage]:
        """Group ``texts`` into role-tagged messages of <= the API content-block cap.

        A source message with more text blocks than the per-message limit is split
        across multiple messages so EVERY block is scanned. Truncating instead would
        let a user hide prohibited content past the limit (guardrail bypass).
        """
        cap = _BEDROCK_CHECKS_MAX_CONTENT_BLOCKS
        return [
            BedrockChecksMessage(
                role=role,
                content=[{"text": text} for text in texts[start : start + cap]],
            )
            for start in range(0, len(texts), cap)
        ]

    def _build_invoke_guardrail_checks_messages(
        self,
        source: Literal["INPUT", "OUTPUT"],
        messages: list[AllMessageValues] | None = None,
        response: litellm.ModelResponse | None = None,
    ) -> list[BedrockChecksMessage]:
        """Build the role-tagged `messages` array for InvokeGuardrailChecks.

        INPUT scans the request messages, OUTPUT scans the model response as an
        ``assistant`` turn. Every non-empty text block of every message is scanned;
        messages exceeding the per-message content-block cap are split into multiple
        messages rather than truncated.

        INPUT content is tagged ``user`` regardless of the caller-supplied role.
        Bedrock excludes ``system`` content from prompt-attack evaluation, so
        trusting a caller's ``system``/``developer`` label would let an injection
        avoid the promptAttack check. At the proxy every INPUT message is
        caller-controlled, so all of it is treated as untrusted user input, matching
        AWS guidance to tag untrusted content as user input.
        """
        if source == "OUTPUT":
            # Reuse the ApplyGuardrail output extractor (single source of truth for
            # pulling assistant text out of a ModelResponse), then re-tag as an
            # assistant turn for the role-based InvokeGuardrailChecks payload.
            output_request = self._create_bedrock_output_content_request(response=response)
            output_texts = [
                text for item in output_request.get("content") or [] if (text := (item.get("text") or {}).get("text"))
            ]
            return self._chunk_texts_into_checks_messages("assistant", output_texts)

        return [
            checks_message
            for message in messages or []
            for checks_message in self._chunk_texts_into_checks_messages(
                "user",
                [block.text for block in self.get_content_items_for_message(message) or [] if block.text],
            )
        ]

    async def _make_invoke_guardrail_checks_request(
        self,
        source: Literal["INPUT", "OUTPUT"],
        messages: list[AllMessageValues] | None = None,
        response: litellm.ModelResponse | None = None,
        request_data: dict | None = None,
        logging_event_type: GuardrailEventHooks | None = None,
    ) -> BedrockGuardrailResponse:
        """Run the resource-less InvokeGuardrailChecks API and enforce thresholds.

        Detect-only: the API returns scores, never rewritten content. We map scores
        to a block decision via the configured thresholds. On a pass we return an
        empty ``BedrockGuardrailResponse`` (downstream masking treats it as a no-op).
        """
        start_time = datetime.now(timezone.utc)

        checks_messages = self._build_invoke_guardrail_checks_messages(
            source=source, messages=messages, response=response
        )
        if not checks_messages:
            # Nothing to scan (e.g. tool-only turn) -> allow, like ApplyGuardrail does.
            return BedrockGuardrailResponse()

        credentials, aws_region_name = self._load_credentials()
        body: dict[str, Any] = {"messages": checks_messages, "checks": self.checks}
        api_key: str | None = request_data.get("api_key") if request_data else None

        prepared_request = self._prepare_request(
            credentials=credentials,
            data=body,
            optional_params=self.optional_params,
            aws_region_name=aws_region_name,
            api_key=api_key,
            request_path=_BEDROCK_INVOKE_GUARDRAIL_CHECKS_PATH,
        )
        verbose_proxy_logger.debug("Bedrock InvokeGuardrailChecks request url: %s", prepared_request.url)

        event_type = logging_event_type or (
            GuardrailEventHooks.pre_call if source == "INPUT" else GuardrailEventHooks.post_call
        )

        httpx_response = await self._sign_and_post(
            prepared_request=prepared_request,
            request_data=request_data,
            event_type=event_type,
            start_time=start_time,
        )

        if httpx_response.status_code != 200:
            status_code, detail_message = self._parse_bedrock_guardrail_error_response(httpx_response)
            verbose_proxy_logger.error(
                "Bedrock InvokeGuardrailChecks: error response. Status %s: %s",
                httpx_response.status_code,
                detail_message,
            )
            self.add_standard_logging_guardrail_information_to_request_data(
                guardrail_provider=self.guardrail_provider,
                guardrail_json_response={"error": detail_message},
                request_data=request_data or {},
                guardrail_status="guardrail_failed_to_respond",
                start_time=start_time.timestamp(),
                end_time=datetime.now(timezone.utc).timestamp(),
                duration=(datetime.now(timezone.utc) - start_time).total_seconds(),
                event_type=event_type,
            )
            raise HTTPException(status_code=status_code, detail=detail_message)

        try:
            json_response = TypeAdapter(BedrockGuardrailChecksResponse).validate_python(httpx_response.json())
        except (ValidationError, ValueError) as e:
            verbose_proxy_logger.error("Bedrock InvokeGuardrailChecks: unparseable 200 response: %s", str(e))
            self.add_standard_logging_guardrail_information_to_request_data(
                guardrail_provider=self.guardrail_provider,
                guardrail_json_response={"error": str(e)},
                request_data=request_data or {},
                guardrail_status="guardrail_failed_to_respond",
                start_time=start_time.timestamp(),
                end_time=datetime.now(timezone.utc).timestamp(),
                duration=(datetime.now(timezone.utc) - start_time).total_seconds(),
                event_type=event_type,
            )
            raise HTTPException(
                status_code=500,
                detail={"error": "Bedrock InvokeGuardrailChecks returned an unexpected response shape"},
            ) from e
        violations = self._collect_invoke_checks_violations(json_response)

        # Log a copy with PII location offsets stripped: offsets + the (separately
        # logged) request messages would otherwise reconstruct the detected PII span.
        self.add_standard_logging_guardrail_information_to_request_data(
            guardrail_provider=self.guardrail_provider,
            guardrail_json_response=self._sanitize_invoke_checks_response_for_logging(json_response),
            request_data=request_data or {},
            guardrail_status=self._get_invoke_checks_status(bool(violations)),
            start_time=start_time.timestamp(),
            end_time=datetime.now(timezone.utc).timestamp(),
            duration=(datetime.now(timezone.utc) - start_time).total_seconds(),
            event_type=event_type,
            tracing_detail=self._build_invoke_checks_tracing_detail(violations) if violations else None,
        )

        if violations:
            raise self._get_block_exception_for_checks(violations, request_data=request_data)

        return BedrockGuardrailResponse()

    def _collect_invoke_checks_violations(
        self, response: BedrockGuardrailChecksResponse | None
    ) -> list[BedrockChecksViolation]:
        """Return the check results whose score meets/exceeds the configured threshold.

        Only checks present in the configured ``checks`` block are evaluated; a
        threshold of ``None`` makes that check detect-only (never contributes a
        violation). A truncated sensitiveInformation result counts as a violation
        (fail closed: omitted detections were never scored). Only the non-sensitive
        label (category/type) and the numeric score are kept -- never offsets or
        matched text.
        """
        results: dict[str, Any] = dict((response or {}).get("results") or {})
        # (results key, score field, label field, threshold). PII uses
        # confidenceScore/type; the other two use severityScore/category.
        check_specs = [
            (
                "contentFilter",
                "severityScore",
                "category",
                self.content_filter_threshold,
            ),
            ("promptAttack", "severityScore", "category", self.prompt_attack_threshold),
            (
                "sensitiveInformation",
                "confidenceScore",
                "type",
                self.pii_confidence_threshold,
            ),
        ]

        configured_checks = self.checks or {}
        violations: list[BedrockChecksViolation] = []
        for check_key, score_field, label_field, threshold in check_specs:
            if threshold is None or check_key not in configured_checks:
                continue
            check_result = results.get(check_key) or {}
            if check_key == "sensitiveInformation" and check_result.get("truncated"):
                violations.append({"check": check_key, "truncated": True})
            for entry in check_result.get("results") or []:
                score = entry.get(score_field)
                if isinstance(score, (int, float)) and float(score) >= threshold:
                    violation: BedrockChecksViolation = (
                        {"check": check_key, "category": entry.get("category"), "severityScore": float(score)}
                        if score_field == "severityScore"
                        else {"check": check_key, "type": entry.get("type"), "confidenceScore": float(score)}
                    )
                    violations.append(violation)
        return violations

    @staticmethod
    def _sanitize_invoke_checks_response_for_logging(
        response: BedrockGuardrailChecksResponse,
    ) -> dict[str, Any]:
        """Strip PII location offsets from a checks response before it is logged."""
        sanitized: dict[str, Any] = copy.deepcopy(dict(response))
        sensitive = (sanitized.get("results") or {}).get("sensitiveInformation") or {}
        for entry in sensitive.get("results") or []:
            if isinstance(entry, dict):
                for key in _BEDROCK_CHECKS_PII_LOCATION_KEYS:
                    entry.pop(key, None)
        return sanitized

    @staticmethod
    def _get_invoke_checks_status(over_threshold: bool) -> GuardrailStatus:
        return "guardrail_intervened" if over_threshold else "success"

    @staticmethod
    def _build_invoke_checks_tracing_detail(
        violations: list[BedrockChecksViolation],
    ) -> GuardrailTracingDetail:
        tracing_detail: GuardrailTracingDetail = {}
        categories = [
            label
            for label in (v.get("category") or v.get("type") for v in violations)
            if isinstance(label, str) and label
        ]
        if categories:
            tracing_detail["violation_categories"] = categories
        tracing_detail["guardrail_action"] = "GUARDRAIL_INTERVENED" if violations else "NONE"
        return tracing_detail

    def _get_block_exception_for_checks(
        self, violations: list[BedrockChecksViolation], request_data: dict | None = None
    ) -> Union[HTTPException, ModifyResponseException]:
        """Build the block exception for an over-threshold InvokeGuardrailChecks result.

        Mirrors ``_get_http_exception_for_blocked_guardrail``'s return-type branching.
        The detail carries only non-sensitive labels + scores (no offsets / raw input).
        """
        if self.disable_exception_on_block is True:
            _request_data = request_data or {}
            return ModifyResponseException(
                message="Violated guardrail policy",
                model=_request_data.get("model", "bedrock-guardrail"),
                request_data=_request_data,
                guardrail_name=self.guardrail_name,
            )
        return HTTPException(
            status_code=400,
            detail={
                "error": "Violated guardrail policy",
                "bedrock_guardrail_checks": violations,
            },
        )

    def _check_bedrock_response_for_exception(self, response) -> bool:
        """
        Return True if the Bedrock ApplyGuardrail response indicates an exception.

        Works with real httpx.Response objects and MagicMock responses used in tests.
        """
        payload = None

        try:
            json_method = getattr(response, "json", None)
            if callable(json_method):
                payload = json_method()
        except Exception:
            payload = None

        if payload is None:
            try:
                raw = getattr(response, "content", None)
                if isinstance(raw, (bytes, bytearray)):
                    payload = json.loads(raw.decode("utf-8"))
                else:
                    text = getattr(response, "text", None)
                    if isinstance(text, str):
                        payload = json.loads(text)
            except Exception:
                # Can't parse -> assume no explicit Exception marker
                return False

        if not isinstance(payload, dict):
            return False

        return "Exception" in payload.get("Output", {}).get("__type", "")

    def _get_bedrock_guardrail_response_status(self, response: httpx.Response) -> GuardrailStatus:
        """
        Get the status of the bedrock guardrail response.

        Returns:
            "success": Content allowed through with no violations
            "guardrail_intervened": Content blocked due to policy violations
            "guardrail_failed_to_respond": Technical error or API failure
        """
        if response.status_code == 200:
            if self._check_bedrock_response_for_exception(response):
                return "guardrail_failed_to_respond"

            # Check if the guardrail would block content
            try:
                _json_response = response.json()
                bedrock_guardrail_response = BedrockGuardrailResponse(**_json_response)
                if self._should_raise_guardrail_blocked_exception(bedrock_guardrail_response):
                    return "guardrail_intervened"
            except Exception:
                pass

            return "success"
        return "guardrail_failed_to_respond"

    def _parse_bedrock_guardrail_error_response(self, response: httpx.Response) -> Tuple[int, str]:
        """
        Parse AWS Bedrock guardrail error response body to extract status code and message.

        AWS may return shapes like {"message": "..."} or {"error": {"message": "..."}}.
        Returns (status_code, message) for use in HTTPException.
        """
        status_code = response.status_code
        message = "Bedrock guardrail request failed"
        try:
            body = response.json()
        except Exception:
            text = getattr(response, "text", None) or ""
            if isinstance(text, str) and text.strip():
                return (status_code, text.strip())
            return (status_code, message)
        if isinstance(body, dict):
            if isinstance(body.get("message"), str):
                return (status_code, body["message"])
            err = body.get("error")
            if isinstance(err, dict) and isinstance(err.get("message"), str):
                return (status_code, err["message"])
            if isinstance(err, str):
                return (status_code, err)
        return (status_code, message)

    def _build_tracing_detail(self, response: BedrockGuardrailResponse) -> GuardrailTracingDetail:
        """
        Build the tracing detail from the raw Bedrock response, before
        redaction, so downstream loggers (OTEL, Langfuse, ...) get the
        actual category names rather than the "[REDACTED]" sentinel that
        replaces customWords.match later. Bedrock's top-level ``action``
        field ("GUARDRAIL_INTERVENED" or "NONE") is also surfaced so the
        OTEL integration can expose it as a queryable span attribute
        without re-parsing the redacted guardrail_response blob.
        """
        tracing_detail: GuardrailTracingDetail = {}
        violation_categories = self._extract_violation_category_names(response)
        if violation_categories:
            tracing_detail["violation_categories"] = violation_categories
        bedrock_action = response.get("action")
        if isinstance(bedrock_action, str):
            tracing_detail["guardrail_action"] = bedrock_action
        return tracing_detail

    def _extract_violation_category_names(self, response: BedrockGuardrailResponse) -> List[str]:
        """
        Flatten the BLOCKED assessments into a list of human-readable category
        names suitable for queryable OTEL / standard-logging attributes.

        SECURITY: only emits the non-sensitive policy *label* (topic name,
        content-filter type, PII entity type, named-regex name). The raw
        ``match`` field is intentionally NOT used — it carries the user's
        original input that triggered the rule (e.g. a credit-card number
        that hit a regex, or the literal custom word). Surfacing it to
        telemetry would re-introduce the sensitive content the guardrail
        was supposed to keep out. Entries that only have a ``match`` (bare
        customWords, unnamed regexes) are therefore skipped — operators
        can still see the count in ``_extract_blocked_assessments`` which
        feeds the HTTP error detail.
        """
        names: List[str] = []
        for block in self._extract_blocked_assessments(response):
            for match in block.get("matches", []) or []:
                # Allow-list non-sensitive labels only. Never fall back to
                # `match.get("match")` — that's user-submitted content.
                label = match.get("name") or match.get("type")
                if isinstance(label, str) and label:
                    names.append(label)
        return names

    def _extract_blocked_assessments(self, response: BedrockGuardrailResponse) -> List[dict]:
        """
        Walk the Bedrock guardrail response and emit a structured list of
        BLOCKED assessment entries describing exactly which policies fired.

        Mirrors the iteration in `_should_raise_guardrail_blocked_exception()`
        but produces a list of `{policy, matches}` dicts instead of a bool.
        Each `match` carries the originating subcategory, type, action, and
        matched term where available, so the client can render a precise
        explanation of the violation.
        """
        blocked: List[dict] = []
        assessments = response.get("assessments", []) or []

        for assessment in assessments:
            # Topic policy
            topic_policy = assessment.get("topicPolicy")
            if topic_policy:
                topic_matches = [
                    {
                        "category": "topics",
                        "name": t.get("name"),
                        "type": t.get("type"),
                        "action": t.get("action"),
                    }
                    for t in (topic_policy.get("topics") or [])
                    if t.get("action") == "BLOCKED"
                ]
                if topic_matches:
                    blocked.append({"policy": "topicPolicy", "matches": topic_matches})

            # Content policy
            content_policy = assessment.get("contentPolicy")
            if content_policy:
                content_matches = [
                    {
                        "category": "filters",
                        "type": f.get("type"),
                        "confidence": f.get("confidence"),
                        "filterStrength": f.get("filterStrength"),
                        "action": f.get("action"),
                    }
                    for f in (content_policy.get("filters") or [])
                    if f.get("action") == "BLOCKED"
                ]
                if content_matches:
                    blocked.append({"policy": "contentPolicy", "matches": content_matches})

            # Word policy
            word_policy = assessment.get("wordPolicy")
            if word_policy:
                word_matches: List[dict] = []
                for w in word_policy.get("customWords") or []:
                    if w.get("action") == "BLOCKED":
                        word_matches.append(
                            {
                                "category": "customWords",
                                "match": w.get("match"),
                                "action": w.get("action"),
                            }
                        )
                for mw in word_policy.get("managedWordLists") or []:
                    if mw.get("action") == "BLOCKED":
                        word_matches.append(
                            {
                                "category": "managedWordLists",
                                "type": mw.get("type"),
                                "match": mw.get("match"),
                                "action": mw.get("action"),
                            }
                        )
                if word_matches:
                    blocked.append({"policy": "wordPolicy", "matches": word_matches})

            # Sensitive information policy (PII)
            sensitive_info = assessment.get("sensitiveInformationPolicy")
            if sensitive_info:
                pii_matches: List[dict] = []
                for p in sensitive_info.get("piiEntities") or []:
                    if p.get("action") == "BLOCKED":
                        pii_matches.append(
                            {
                                "category": "piiEntities",
                                "type": p.get("type"),
                                "match": p.get("match"),
                                "action": p.get("action"),
                            }
                        )
                for r in sensitive_info.get("regexes") or []:
                    if r.get("action") == "BLOCKED":
                        pii_matches.append(
                            {
                                "category": "regexes",
                                "name": r.get("name"),
                                "regex": r.get("regex"),
                                "match": r.get("match"),
                                "action": r.get("action"),
                            }
                        )
                if pii_matches:
                    blocked.append(
                        {
                            "policy": "sensitiveInformationPolicy",
                            "matches": pii_matches,
                        }
                    )

            # Contextual grounding policy
            contextual = assessment.get("contextualGroundingPolicy")
            if contextual:
                grounding_matches = [
                    {
                        "category": "filters",
                        "type": f.get("type"),
                        "threshold": f.get("threshold"),
                        "score": f.get("score"),
                        "action": f.get("action"),
                    }
                    for f in (contextual.get("filters") or [])
                    if f.get("action") == "BLOCKED"
                ]
                if grounding_matches:
                    blocked.append(
                        {
                            "policy": "contextualGroundingPolicy",
                            "matches": grounding_matches,
                        }
                    )

        return blocked

    def _get_http_exception_for_blocked_guardrail(
        self, response: BedrockGuardrailResponse, request_data: Optional[dict] = None
    ) -> Union[HTTPException, ModifyResponseException]:
        """
        Get the HTTP exception for a blocked guardrail.
        """
        bedrock_guardrail_output_text: str = ""
        outputs: Optional[List[BedrockGuardrailOutput]] = response.get("outputs", []) or []
        if outputs:
            for output in outputs:
                if output.get("text"):
                    bedrock_guardrail_output_text += output.get("text") or ""

        if self.disable_exception_on_block is True:
            _request_data = request_data or {}
            return ModifyResponseException(
                message=bedrock_guardrail_output_text,
                model=_request_data.get("model", "bedrock-guardrail"),
                request_data=_request_data,
                guardrail_name=self.guardrail_name,
            )

        detail: Dict[str, Any] = {
            "error": "Violated guardrail policy",
            "bedrock_guardrail_response": bedrock_guardrail_output_text,
        }
        if self.guardrailIdentifier:
            detail["guardrailIdentifier"] = self.guardrailIdentifier
        if self.guardrailVersion:
            detail["guardrailVersion"] = self.guardrailVersion

        assessments = self._extract_blocked_assessments(response)
        if assessments:
            detail["assessments"] = _redact_assessment_match_fields(assessments)

        return HTTPException(status_code=400, detail=detail)

    def _should_raise_guardrail_blocked_exception(self, response: BedrockGuardrailResponse) -> bool:
        """
        Only raise exception for "BLOCKED" actions, not for "ANONYMIZED" actions.

        If `self.mask_request_content` or `self.mask_response_content` is set to `True`,
        then use the output from the guardrail to mask the request or response content.

        However, even with masking enabled, content with action="BLOCKED" should still
        raise an exception, only content with action="ANONYMIZED" should be masked.
        """

        # if no intervention, return False
        if response.get("action") != "GUARDRAIL_INTERVENED":
            return False

        # Check assessments to determine if any actions were BLOCKED (vs ANONYMIZED)
        # NOTE: Use `.get("k") or []` not `.get("k", [])` — Bedrock can return explicit
        # JSON null; dict.get("k", []) then yields None, and `for x in None` raises.
        assessments = response.get("assessments") or []
        if not assessments:
            return False

        for assessment in assessments:
            # Check topic policy
            topic_policy = assessment.get("topicPolicy")
            if topic_policy:
                topics = topic_policy.get("topics") or []
                for topic in topics:
                    if topic.get("action") == "BLOCKED":
                        return True

            # Check content policy
            content_policy = assessment.get("contentPolicy")
            if content_policy:
                filters = content_policy.get("filters") or []
                for filter_item in filters:
                    if filter_item.get("action") == "BLOCKED":
                        return True

            # Check word policy
            word_policy = assessment.get("wordPolicy")
            if word_policy:
                custom_words = word_policy.get("customWords") or []
                for custom_word in custom_words:
                    if custom_word.get("action") == "BLOCKED":
                        return True
                managed_words = word_policy.get("managedWordLists") or []
                for managed_word in managed_words:
                    if managed_word.get("action") == "BLOCKED":
                        return True

            # Check sensitive information policy
            sensitive_info_policy = assessment.get("sensitiveInformationPolicy")
            if sensitive_info_policy:
                pii_entities = sensitive_info_policy.get("piiEntities") or []
                if pii_entities:
                    for pii_entity in pii_entities:
                        if pii_entity.get("action") == "BLOCKED":
                            return True
                regexes = sensitive_info_policy.get("regexes") or []
                if regexes:
                    for regex in regexes:
                        if regex.get("action") == "BLOCKED":
                            return True

            # Check contextual grounding policy
            contextual_grounding_policy = assessment.get("contextualGroundingPolicy")
            if contextual_grounding_policy:
                grounding_filters = contextual_grounding_policy.get("filters") or []
                for grounding_filter in grounding_filters:
                    if grounding_filter.get("action") == "BLOCKED":
                        return True

        # If we got here, intervention occurred but no BLOCKED actions found
        # This means all actions were ANONYMIZED or NONE, so don't raise exception
        return False

    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict,
        call_type: CallTypesLiteral,
    ) -> Union[Exception, str, dict, None]:
        verbose_proxy_logger.debug("Inside Bedrock Pre-Call Hook for call_type: %s", call_type)

        from litellm.proxy.common_utils.callback_utils import (
            add_guardrail_to_applied_guardrails_header,
        )

        event_type: GuardrailEventHooks = GuardrailEventHooks.pre_call
        if self.should_run_guardrail(data=data, event_type=event_type) is not True:
            return data

        new_messages = self.get_guardrails_messages_for_call_type(
            call_type=cast(CallTypes, call_type),
            data=data,
        )

        # Handle None case
        if new_messages is None:
            verbose_proxy_logger.debug("No messages found for call_type, skipping guardrail")
            return data

        filter_result = self._prepare_guardrail_messages_for_role(messages=new_messages)

        filtered_messages = filter_result.payload_messages
        if not filtered_messages:
            verbose_proxy_logger.debug("No user-role messages available for guardrail payload")
            return data

        #########################################################
        ########## 1. Make the Bedrock API request ##########
        #########################################################
        # A block with disable_exception_on_block=True raises ModifyResponseException
        # from make_bedrock_api_request; that propagates to the endpoint handler,
        # which returns a 200 whose message is the guardrail's blockedInputMessaging.
        bedrock_guardrail_response = await self.make_bedrock_api_request(
            source="INPUT",
            messages=filtered_messages,
            request_data=data,
            logging_event_type=GuardrailEventHooks.pre_call,
        )
        #########################################################

        #########################################################
        ########## 2. Update the messages with the guardrail response ##########
        #########################################################
        updated_subset = self._update_messages_with_updated_bedrock_guardrail_response(
            messages=filtered_messages,
            bedrock_guardrail_response=bedrock_guardrail_response,
        )
        data["messages"] = self._merge_filtered_messages(
            original_messages=filter_result.original_messages or new_messages,
            updated_target_messages=updated_subset,
            target_indices=filter_result.target_indices,
        )

        #########################################################
        ########## 3. Add the guardrail to the applied guardrails header ##########
        #########################################################
        add_guardrail_to_applied_guardrails_header(request_data=data, guardrail_name=self.guardrail_name)
        return data

    async def async_moderation_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        call_type: CallTypesLiteral,
    ):
        from litellm.proxy.common_utils.callback_utils import (
            add_guardrail_to_applied_guardrails_header,
        )

        event_type: GuardrailEventHooks = GuardrailEventHooks.during_call
        if self.should_run_guardrail(data=data, event_type=event_type) is not True:
            return

        new_messages = self.get_guardrails_messages_for_call_type(
            call_type=cast(CallTypes, call_type),
            data=data,
        )

        if new_messages is None:
            verbose_proxy_logger.warning("Bedrock AI: not running guardrail. No messages in data")
            return

        filter_result = self._prepare_guardrail_messages_for_role(messages=new_messages)
        filtered_messages = filter_result.payload_messages
        if not filtered_messages:
            verbose_proxy_logger.debug("Bedrock AI: not running guardrail. No user-role messages")
            return

        #########################################################
        ########## 1. Make the Bedrock API request ##########
        #########################################################
        # A block with disable_exception_on_block=True raises ModifyResponseException
        # from make_bedrock_api_request. Because during_call runs in an asyncio.gather
        # alongside the LLM call (common_request_processing.py), swallowing the
        # exception here to set data["mock_response"] was ineffective: route_request
        # unpacked kwargs before this hook ran, and the LLM task's response was taken
        # unconditionally. Letting the exception propagate cancels the LLM task and
        # the endpoint handler returns the block response.
        bedrock_guardrail_response = await self.make_bedrock_api_request(
            source="INPUT",
            messages=filtered_messages,
            request_data=data,
            logging_event_type=GuardrailEventHooks.during_call,
        )
        #########################################################

        #########################################################
        ########## 2. Update the messages with the guardrail response ##########
        #########################################################
        updated_subset = self._update_messages_with_updated_bedrock_guardrail_response(
            messages=filtered_messages,
            bedrock_guardrail_response=bedrock_guardrail_response,
        )
        data["messages"] = self._merge_filtered_messages(
            original_messages=filter_result.original_messages or new_messages,
            updated_target_messages=updated_subset,
            target_indices=filter_result.target_indices,
        )

        #########################################################
        ########## 3. Add the guardrail to the applied guardrails header ##########
        #########################################################
        add_guardrail_to_applied_guardrails_header(request_data=data, guardrail_name=self.guardrail_name)

        return data

    async def async_post_call_success_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        response,
    ):
        from litellm.proxy.common_utils.callback_utils import (
            add_guardrail_to_applied_guardrails_header,
        )
        from litellm.types.guardrails import GuardrailEventHooks

        if self.should_run_guardrail(data=data, event_type=GuardrailEventHooks.post_call) is not True:
            return

        new_messages: Optional[List[AllMessageValues]] = data.get("messages")
        if new_messages is None:
            verbose_proxy_logger.warning("Bedrock AI: not running guardrail. No messages in data")
            return

        # Check if the ModelResponse has text content in its choices
        # to avoid sending empty content to Bedrock (e.g., during tool calls)
        if isinstance(response, litellm.ModelResponse):
            has_text_content = False
            for choice in response.choices:
                if isinstance(choice, litellm.Choices):
                    if choice.message.content and isinstance(choice.message.content, str):
                        has_text_content = True
                        break

            if not has_text_content:
                verbose_proxy_logger.warning("Bedrock AI: not running guardrail. No output text in response")
                return

        #########################################################
        ########## 1. Make Bedrock API requests ##########
        #########################################################
        # post_call is the response-validation hook by definition — only scan
        # OUTPUT. Input scanning belongs to pre_call / during_call hooks, which
        # users should configure if they want input validation. Running an
        # extra INPUT scan here produced a duplicate post-call entry in the
        # trace and made no semantic sense for a "post-call" event.
        # A block with disable_exception_on_block=True raises ModifyResponseException
        # from make_bedrock_api_request; that propagates to the endpoint handler,
        # which returns a 200 whose message is the guardrail's blockedInputMessaging.
        # Attach the LLM response to original_response so the synthetic block reply
        # reports the real token usage the upstream call consumed instead of zero.
        try:
            output_content_bedrock = await self.make_bedrock_api_request(
                source="OUTPUT",
                response=response,
                messages=new_messages,
                request_data=data,
                logging_event_type=GuardrailEventHooks.post_call,
            )
        except ModifyResponseException as e:
            if e.original_response is None:
                e.original_response = response
            raise

        #########################################################
        ########## 2. Apply masking to response with output guardrail response ##########
        #########################################################
        if output_content_bedrock is not None:
            self._apply_masking_to_response(
                response=response,
                bedrock_guardrail_response=output_content_bedrock,
            )

        #########################################################
        ########## 3. Add the guardrail to the applied guardrails header ##########
        #########################################################
        add_guardrail_to_applied_guardrails_header(request_data=data, guardrail_name=self.guardrail_name)

    ###########  HELPER FUNCTIONS for bedrock guardrails ############################
    ##############################################################################
    ##############################################################################
    def _update_messages_with_updated_bedrock_guardrail_response(
        self,
        messages: List[AllMessageValues],
        bedrock_guardrail_response: BedrockGuardrailResponse,
    ) -> List[AllMessageValues]:
        """
        Use the output from the bedrock guardrail to mask sensitive content in messages.

        Args:
            messages: Original list of messages
            bedrock_guardrail_response: Response from Bedrock guardrail containing masked content

        Returns:
            List of messages with content masked according to guardrail response
        """
        # Get masked texts from guardrail response
        masked_texts = self._extract_masked_texts_from_response(bedrock_guardrail_response)

        # If guardrail provided masked output, use it regardless of masking flags
        # because the guardrail has already determined this content needs anonymization
        if masked_texts:
            verbose_proxy_logger.debug("Bedrock guardrail provided masked output, applying to messages")
            return self._apply_masking_to_messages(messages=messages, masked_texts=masked_texts)

        # If masking is enabled but no masked texts available, still try to apply
        # (this maintains backward compatibility for edge cases)
        if self.mask_request_content or self.mask_response_content:
            verbose_proxy_logger.debug(
                "Masking enabled but no masked output from guardrail, returning original messages"
            )

        return messages

    async def async_post_call_streaming_iterator_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        response: Any,
        request_data: dict,
    ) -> AsyncGenerator[ModelResponseStream, None]:
        """
        Process streaming response chunks.

        Collect content from the stream and run the bedrock OUTPUT scan
        (post_call only validates the response).
        """
        # Import here to avoid circular imports
        from litellm.llms.base_llm.base_model_iterator import MockResponseIterator
        from litellm.main import stream_chunk_builder
        from litellm.types.utils import TextCompletionResponse

        # Collect all chunks to process them together
        all_chunks: List[ModelResponseStream] = []
        async for chunk in response:
            all_chunks.append(chunk)

        assembled_model_response: Optional[Union[ModelResponse, TextCompletionResponse]] = stream_chunk_builder(
            chunks=all_chunks,
        )
        if isinstance(assembled_model_response, ModelResponse):
            ####################################################################
            ########## 1. Make Bedrock Apply Guardrail API request ##########
            #
            # post_call only scans OUTPUT — input scanning belongs to
            # pre_call / during_call. Bedrock will raise if the response
            # violates the guardrail policy.
            ###################################################################
            # A block with disable_exception_on_block=True raises ModifyResponseException
            # from make_bedrock_api_request. Non-streaming paths let it propagate so
            # the endpoint handler turns it into a 200. Streaming can't do that: the
            # SSE response headers are already flushed, so a raise would be serialized
            # as an error frame by async_streaming_data_generator. Instead, replace
            # the assembled response with the synthetic block content in-place and
            # yield it as a normal stream, matching the shape a non-streaming block
            # produces.
            try:
                output_guardrail_response = await self.make_bedrock_api_request(
                    source="OUTPUT",
                    response=assembled_model_response,
                    messages=request_data.get("messages"),
                    request_data=request_data,
                    logging_event_type=GuardrailEventHooks.post_call,
                )
            except ModifyResponseException as e:
                # Preserve upstream usage from the LLM call we already
                # consumed. Non-streaming blocks carry it via
                # ModifyResponseException.original_response +
                # _blocked_response_usage; streaming has to do the copy
                # itself since the exception can't escape this generator.
                _original_usage = getattr(assembled_model_response, "usage", None)
                assembled_model_response = ModelResponse(
                    choices=[
                        Choices(
                            index=0,
                            message=Message(role="assistant", content=e.message),
                            finish_reason="content_filter",
                        )
                    ],
                    model=e.model,
                )
                if _original_usage is not None:
                    assembled_model_response.usage = _original_usage
                output_guardrail_response = None

            #########################################################################
            ########## 2. Apply masking to response with output guardrail response ##########
            #########################################################################
            if output_guardrail_response is not None:
                self._apply_masking_to_response(
                    response=assembled_model_response,
                    bedrock_guardrail_response=output_guardrail_response,
                )

            #########################################################################
            ########## 3. Return the (potentially masked) chunks ##########
            #########################################################################
            mock_response = MockResponseIterator(model_response=assembled_model_response)

            # Return the reconstructed stream
            async for chunk in mock_response:
                yield chunk
        else:
            for chunk in all_chunks:
                yield chunk

    def _extract_masked_texts_from_response(self, bedrock_guardrail_response: BedrockGuardrailResponse) -> List[str]:
        """
        Extract all masked text outputs from the guardrail response.

        Args:
            bedrock_guardrail_response: Response from Bedrock guardrail

        Returns:
            List of masked text strings
        """
        masked_output_text: List[str] = []
        masked_outputs: Optional[List[BedrockGuardrailOutput]] = bedrock_guardrail_response.get("outputs", []) or []
        if not masked_outputs:
            verbose_proxy_logger.debug("No masked outputs found in guardrail response")
            return []

        for output in masked_outputs:
            text_content: Optional[str] = output.get("text")
            if text_content is not None:
                masked_output_text.append(text_content)

        return masked_output_text

    def _apply_masking_to_messages(
        self, messages: List[AllMessageValues], masked_texts: List[str]
    ) -> List[AllMessageValues]:
        """
        Apply masked texts to message content using index tracking.

        Args:
            messages: Original messages
            masked_texts: List of masked text strings from guardrail

        Returns:
            Updated messages with masked content
        """
        updated_messages = []
        masking_index = 0

        for message in messages:
            new_message = message.copy()
            content = new_message.get("content")

            # Skip messages with no content
            if content is None:
                updated_messages.append(new_message)
                continue

            # Handle string content
            if isinstance(content, str):
                if masking_index < len(masked_texts):
                    new_message["content"] = masked_texts[masking_index]
                    masking_index += 1
            # Handle list content
            elif isinstance(content, list):
                new_message["content"], masking_index = self._mask_content_list(
                    content_list=content,
                    masked_texts=masked_texts,
                    masking_index=masking_index,
                )

            updated_messages.append(new_message)

        return updated_messages

    def _mask_content_list(
        self, content_list: List[Any], masked_texts: List[str], masking_index: int
    ) -> Tuple[List[Any], int]:
        """
        Apply masking to a list of content items.

        Args:
            content_list: List of content items
            masked_texts: List of masked text strings
            starting_index: Starting index in the masked_texts list

        Returns:
            Updated content list with masked items
        """
        new_content: List[Union[dict, str]] = []
        for item in content_list:
            if isinstance(item, dict) and "text" in item:
                new_item = item.copy()
                if masking_index < len(masked_texts):
                    new_item["text"] = masked_texts[masking_index]
                    masking_index += 1
                new_content.append(new_item)
            elif isinstance(item, str):
                if masking_index < len(masked_texts):
                    item = masked_texts[masking_index]
                    masking_index += 1
                if item is not None:
                    new_content.append(item)

        return new_content, masking_index

    def _apply_masking_to_response(
        self,
        response: Union[ModelResponse, Any],
        bedrock_guardrail_response: BedrockGuardrailResponse,
    ) -> None:
        """
        Apply masked content from bedrock guardrail to the response object.

        Args:
            response: The response object to modify
            bedrock_guardrail_response: Response from Bedrock guardrail containing masked content
        """
        # Get masked texts from guardrail response
        masked_texts = self._extract_masked_texts_from_response(bedrock_guardrail_response)

        if not masked_texts:
            verbose_proxy_logger.debug("No masked outputs found, skipping response masking")
            return

        verbose_proxy_logger.debug("Applying masking to response with %d masked texts", len(masked_texts))

        # Apply masking to ModelResponse
        if isinstance(response, litellm.ModelResponse):
            self._apply_masking_to_model_response(response, masked_texts)
        else:
            verbose_proxy_logger.warning("Unsupported response type for masking: %s", type(response))

    def _apply_masking_to_model_response(self, response: litellm.ModelResponse, masked_texts: List[str]) -> None:
        """
        Apply masked texts to a ModelResponse object.

        Args:
            response: The ModelResponse object to modify in-place
            masked_texts: List of masked text strings from guardrail
        """
        masking_index = 0

        for choice in response.choices:
            if isinstance(choice, Choices):
                # For chat completions
                if choice.message.content and isinstance(choice.message.content, str):
                    if masking_index < len(masked_texts):
                        choice.message.content = masked_texts[masking_index]
                        masking_index += 1
                        verbose_proxy_logger.debug("Applied masking to choice message content")
            elif isinstance(choice, StreamingChoices):
                # For streaming responses, modify delta content
                if choice.delta.content and isinstance(choice.delta.content, str):
                    if masking_index < len(masked_texts):
                        choice.delta.content = masked_texts[masking_index]
                        masking_index += 1
                        verbose_proxy_logger.debug("Applied masking to choice delta content")
            elif isinstance(choice, TextChoices):
                # For text completions
                if choice.text and isinstance(choice.text, str):
                    if masking_index < len(masked_texts):
                        choice.text = masked_texts[masking_index]
                        masking_index += 1
                        verbose_proxy_logger.debug("Applied masking to choice text content")

    async def apply_guardrail(
        self,
        inputs: "GenericGuardrailAPIInputs",
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: Optional["LiteLLMLoggingObj"] = None,
    ) -> "GenericGuardrailAPIInputs":
        """
        Apply Bedrock guardrail to a batch of texts for testing purposes.

        This method allows users to test Bedrock guardrails without making actual LLM calls.
        It creates mock messages to test the guardrail functionality.

        Args:
            inputs: Dictionary containing texts and optional images
            request_data: Request data dictionary for logging metadata
            input_type: Whether this is a "request" or "response"
            logging_obj: Optional logging object

        Returns:
            GenericGuardrailAPIInputs - processed_texts may be masked, images unchanged

        Raises:
            Exception: If content is blocked by Bedrock guardrail
        """
        # NOTE: Use `or []` to handle case where inputs["texts"] is explicitly None.
        # dict.get("texts", []) would return None if the key exists with a None value.
        texts = inputs.get("texts") or []
        try:
            verbose_proxy_logger.debug(f"Bedrock Guardrail: Applying guardrail to {len(texts)} text(s)")

            masked_texts = []

            selection = self._select_messages_for_apply_guardrail(
                texts=texts,
                inputs=inputs,
                request_data=request_data,
                input_type=input_type,
            )
            if selection.skip_scan:
                return inputs
            filtered_messages = selection.filtered_messages
            scanned_slice = selection.scanned_slice
            scanned_role_subset = selection.scanned_role_subset

            # Bedrock will throw an error if there is no text to process
            if filtered_messages:
                _log_hook = GuardrailEventHooks.pre_call if input_type == "request" else GuardrailEventHooks.post_call
                # Map the abstract input_type to the Bedrock source parameter.
                # "request"  -> INPUT  (scan user-supplied content)
                # "response" -> OUTPUT (scan model-generated content)
                # Bedrock guardrail policies are often configured differently
                # for Input vs Output (e.g. PII blocking only on Output), so
                # the source MUST match where the text originated.
                bedrock_source: Literal["INPUT", "OUTPUT"] = "OUTPUT" if input_type == "response" else "INPUT"
                if bedrock_source == "OUTPUT":
                    # Build a synthetic ModelResponse whose choices carry the
                    # text(s) to scan, so _create_bedrock_output_content_request
                    # can produce the correct Bedrock OUTPUT payload.
                    synthetic_response = ModelResponse(
                        choices=[
                            Choices(
                                index=_idx,
                                message=Message(
                                    role="assistant",
                                    content=str(_msg.get("content") or ""),
                                ),
                                finish_reason="stop",
                            )
                            for _idx, _msg in enumerate(filtered_messages)
                        ]
                    )
                    bedrock_response = await self.make_bedrock_api_request(
                        source="OUTPUT",
                        response=synthetic_response,
                        request_data=request_data,
                        logging_event_type=_log_hook,
                    )
                else:
                    bedrock_response = await self.make_bedrock_api_request(
                        source="INPUT",
                        messages=filtered_messages,
                        request_data=request_data,
                        logging_event_type=_log_hook,
                    )

                # Apply any masking that was applied by the guardrail
                output_list = bedrock_response.get("output")
                if output_list:
                    # If the guardrail returned modified content, use that
                    for output_item in output_list:
                        text_content = output_item.get("text")
                        if text_content:
                            masked_text = str(text_content)
                            masked_texts.append(masked_text)
                else:
                    outputs_list = bedrock_response.get("outputs")
                    if outputs_list:
                        # Fallback to outputs field if output is not available
                        for output_item in outputs_list:
                            text_content = output_item.get("text")
                            if text_content:
                                masked_text = str(text_content)
                                masked_texts.append(masked_text)

            # Reconcile masked output with the flat `texts` list (write back to
            # the scanned slice only, or skip if it can't be aligned).
            masked_texts = self._merge_masked_texts(
                masked_texts=masked_texts,
                texts=texts,
                scanned_slice=scanned_slice,
                scanned_role_subset=scanned_role_subset,
            )

            verbose_proxy_logger.debug("Bedrock Guardrail: Successfully applied guardrail")

            inputs["texts"] = masked_texts
            return inputs

        except (HTTPException, ModifyResponseException):
            # Let guardrail blocking exceptions propagate as-is so the proxy can
            # return the correct HTTP status (400 for HTTPException, 200 with the
            # block message for ModifyResponseException in disable_exception_on_block
            # mode). Without this, the generic except below wraps them into a plain
            # Exception, losing the semantics and preventing the proxy from
            # properly blocking the call.
            raise
        except Exception as e:
            verbose_proxy_logger.error("Bedrock Guardrail: Failed to apply guardrail: %s", str(e))
            raise Exception(f"Bedrock guardrail failed: {str(e)}")
