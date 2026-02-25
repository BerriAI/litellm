"""
Block Code Execution guardrail.

Detects markdown fenced code blocks in request/response content and blocks or masks them
when the language is in the blocked list (or all blocks when list is empty). Supports
confidence scoring and a tunable threshold (only block when confidence >= threshold).
"""

import re
from datetime import datetime
from typing import (TYPE_CHECKING, Any, AsyncGenerator, Dict, List, Literal,
                    Optional, Tuple, Union, cast)

from fastapi import HTTPException

from litellm.integrations.custom_guardrail import (CustomGuardrail,
                                                   log_guardrail_information)
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.proxy.guardrails.guardrail_hooks.base import \
    GuardrailConfigModel
from litellm.types.proxy.guardrails.guardrail_hooks.block_code_execution import (
    CodeBlockActionTaken, CodeBlockDetection)
from litellm.types.utils import (GenericGuardrailAPIInputs, GuardrailStatus,
                                 GuardrailTracingDetail, ModelResponseStream)

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import \
        Logging as LiteLLMLoggingObj

# Default executable languages when blocked_languages is not set (block-all mode uses this for "block all")
DEFAULT_BLOCKED_LANGUAGES: List[str] = [
    "python",
    "javascript",
    "js",
    "bash",
    "sh",
    "ruby",
    "go",
    "java",
    "csharp",
    "php",
    "c",
    "cpp",
    "rust",
    "sql",
]

# Language tag aliases (normalize to canonical for comparison)
LANGUAGE_ALIASES: Dict[str, str] = {
    "js": "javascript",
    "py": "python",
}

# Tags that indicate non-executable / plain text (lower confidence when block-all)
NON_EXECUTABLE_TAGS: frozenset = frozenset(
    {"text", "plaintext", "plain", "markdown", "md", "output", "result"}
)

# Regex: fenced code block with optional language tag. Handles ```lang\n...\n```
# Content between fences; does not handle nested ``` inside body (documented edge case).
FENCED_BLOCK_RE = re.compile(r"```(\w*)\n(.*?)```", re.DOTALL)


def _normalize_escaped_newlines(text: str) -> str:
    """
    Replace literal escaped newlines (backslash + n or backslash + r) with real newlines.
    API/JSON payloads sometimes deliver newlines as the two-character sequence \\n.
    """
    if not text:
        return text
    # Order matters: replace \r\n first so we don't produce extra \n from \r then \n
    text = text.replace("\\r\\n", "\n")
    text = text.replace("\\n", "\n")
    text = text.replace("\\r", "\n")
    return text


def _normalize_language(tag: str) -> str:
    """Normalize language tag (lowercase, resolve aliases)."""
    tag = (tag or "").strip().lower()
    return LANGUAGE_ALIASES.get(tag, tag)


def _is_blocked_language(
    tag: str,
    blocked_languages: Optional[List[str]],
    block_all: bool,
) -> bool:
    """True if this language tag should be considered blocked."""
    normalized = _normalize_language(tag)
    if block_all:
        # Block all: only allow through if it's explicitly non-executable (we still block but with lower confidence)
        return True
    if not blocked_languages:
        return True
    normalized_list = [_normalize_language(t) for t in blocked_languages]
    return normalized in normalized_list


def _confidence_for_block(
    tag: str,
    block_all: bool,
    tag_in_blocked_list: bool,
) -> float:
    """Return confidence in [0, 1] for this code block detection."""
    normalized = _normalize_language(tag)
    if tag_in_blocked_list:
        return 1.0
    if block_all:
        # Explicit non-executable tags (e.g. text, plaintext) get lower confidence
        if normalized in NON_EXECUTABLE_TAGS:
            return 0.5
        # Untagged or other tags in block-all mode: treat as executable, high confidence
        return 1.0
    return 0.0


class BlockCodeExecutionGuardrail(CustomGuardrail):
    """
    Guardrail that detects fenced code blocks (markdown ```) and blocks or masks them
    when the language is in the blocked list (or all when list is empty/None).
    Supports confidence threshold: only block when confidence >= confidence_threshold.
    """

    MASK_PLACEHOLDER = "[CODE_BLOCK_REDACTED]"

    def __init__(
        self,
        guardrail_name: Optional[str] = None,
        blocked_languages: Optional[List[str]] = None,
        action: Literal["block", "mask"] = "block",
        confidence_threshold: float = 0.5,
        event_hook: Optional[
            Union[Literal["pre_call", "post_call", "during_call"], List[str]]
        ] = None,
        default_on: bool = False,
        **kwargs: Any,
    ) -> None:
        # Normalize to type expected by CustomGuardrail
        _event_hook: Optional[Union[GuardrailEventHooks, List[GuardrailEventHooks]]] = (
            None
        )
        if event_hook is not None:
            if isinstance(event_hook, list):
                _event_hook = [
                    GuardrailEventHooks(h) if isinstance(h, str) else h
                    for h in event_hook
                ]
            else:
                _event_hook = GuardrailEventHooks(event_hook)
        super().__init__(
            guardrail_name=guardrail_name or "block_code_execution",
            supported_event_hooks=[
                GuardrailEventHooks.pre_call,
                GuardrailEventHooks.post_call,
                GuardrailEventHooks.during_call,
            ],
            event_hook=_event_hook
            or [
                GuardrailEventHooks.pre_call,
                GuardrailEventHooks.post_call,
            ],
            default_on=default_on,
            **kwargs,
        )
        self.blocked_languages = blocked_languages
        self.block_all = blocked_languages is None or len(blocked_languages) == 0
        self.action = action
        self.confidence_threshold = max(0.0, min(1.0, confidence_threshold))

    @staticmethod
    def get_config_model() -> Optional[type[GuardrailConfigModel]]:
        from litellm.types.proxy.guardrails.guardrail_hooks.block_code_execution import \
            BlockCodeExecutionGuardrailConfigModel

        return BlockCodeExecutionGuardrailConfigModel

    def _find_blocks(
        self, text: str
    ) -> List[Tuple[str, str, float, CodeBlockActionTaken]]:
        """
        Find all fenced code blocks in text. Returns list of
        (language_tag, block_content, confidence, action_taken).
        """
        results: List[Tuple[str, str, float, CodeBlockActionTaken]] = []
        for m in FENCED_BLOCK_RE.finditer(text):
            tag = (m.group(1) or "").strip()
            body = m.group(2)
            tag_in_list = not self.block_all and _normalize_language(tag) in [
                _normalize_language(t) for t in (self.blocked_languages or [])
            ]
            is_blocked = _is_blocked_language(
                tag, self.blocked_languages, self.block_all
            )
            confidence = _confidence_for_block(tag, self.block_all, tag_in_list)
            if not is_blocked:
                action_taken: CodeBlockActionTaken = "allow"
            elif confidence >= self.confidence_threshold:
                action_taken = "block"
            else:
                action_taken = "log_only"
            results.append((tag or "(none)", body, confidence, action_taken))
        return results

    def _scan_text(
        self,
        text: str,
        detections: Optional[List[CodeBlockDetection]] = None,
    ) -> Tuple[str, bool]:
        """
        Scan one text: find blocks, apply block/mask/allow by confidence.
        Returns (modified_text, should_raise).
        """
        if not text:
            return text, False
        text = _normalize_escaped_newlines(text)
        blocks = self._find_blocks(text)
        if not blocks:
            return text, False

        should_raise = False
        last_end = 0
        parts: List[str] = []
        for m in FENCED_BLOCK_RE.finditer(text):
            tag = (m.group(1) or "").strip()
            tag_in_list = not self.block_all and _normalize_language(tag) in [
                _normalize_language(t) for t in (self.blocked_languages or [])
            ]
            is_blocked = _is_blocked_language(
                tag, self.blocked_languages, self.block_all
            )
            confidence = _confidence_for_block(tag, self.block_all, tag_in_list)
            if not is_blocked:
                action_taken: CodeBlockActionTaken = "allow"
            elif confidence >= self.confidence_threshold:
                action_taken = "block"
            else:
                action_taken = "log_only"

            if detections is not None:
                detections.append(
                    cast(
                        CodeBlockDetection,
                        {
                            "type": "code_block",
                            "language": tag or "(none)",
                            "confidence": round(confidence, 2),
                            "action_taken": action_taken,
                        },
                    )
                )

            if action_taken == "block" and self.action == "block":
                should_raise = True
            parts.append(text[last_end : m.start()])
            if action_taken == "block":
                parts.append(self.MASK_PLACEHOLDER)
            else:
                parts.append(text[m.start() : m.end()])
            last_end = m.end()

        parts.append(text[last_end:])
        new_text = "".join(parts)
        return new_text, should_raise

    def _raise_block_error(
        self, language: str, is_output: bool, request_data: dict
    ) -> None:
        msg = f"Content blocked: executable code block detected (language: {language})"
        if is_output:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": msg,
                    "guardrail": self.guardrail_name,
                    "language": language,
                },
            )
        self.raise_passthrough_exception(
            violation_message=msg,
            request_data=request_data,
            detection_info={"language": language},
        )

    @log_guardrail_information
    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: Optional["LiteLLMLoggingObj"] = None,
    ) -> GenericGuardrailAPIInputs:
        start_time = datetime.now()
        detections: List[CodeBlockDetection] = []
        status: GuardrailStatus = "success"
        exception_str = ""

        try:
            texts = inputs.get("texts", [])
            if not texts:
                return inputs

            is_output = input_type == "response"
            processed: List[str] = []
            for text in texts:
                new_text, should_raise = self._scan_text(text, detections)
                processed.append(new_text)
                if should_raise:
                    # Determine language from first blocking detection
                    lang = "unknown"
                    for d in detections:
                        if d.get("action_taken") == "block":
                            lang = d.get("language", "unknown")
                            break
                    self._raise_block_error(lang, is_output, request_data)

            inputs["texts"] = processed
            return inputs
        except HTTPException:
            status = "guardrail_intervened"
            raise
        except Exception as e:
            status = "guardrail_failed_to_respond"
            exception_str = str(e)
            raise
        finally:
            guardrail_response: Union[List[dict], str] = [dict(d) for d in detections]
            if status != "success" and not detections:
                guardrail_response = exception_str
            max_confidence: Optional[float] = None
            for d in detections:
                c = d.get("confidence")
                if c is not None and (max_confidence is None or c > max_confidence):
                    max_confidence = c
            tracing_kw: Dict[str, Any] = {
                "guardrail_id": self.guardrail_name,
                "detection_method": "fenced_code_block",
                "match_details": guardrail_response,
            }
            if max_confidence is not None:
                tracing_kw["confidence_score"] = max_confidence
            self.add_standard_logging_guardrail_information_to_request_data(
                guardrail_provider="block_code_execution",
                guardrail_json_response=guardrail_response,
                request_data=request_data,
                guardrail_status=status,
                start_time=start_time.timestamp(),
                end_time=datetime.now().timestamp(),
                duration=(datetime.now() - start_time).total_seconds(),
                tracing_detail=GuardrailTracingDetail(**tracing_kw),  # type: ignore[typeddict-item]
            )

    async def async_post_call_streaming_iterator_hook(
        self,
        user_api_key_dict: Any,
        response: Any,
        request_data: dict,
    ) -> AsyncGenerator[ModelResponseStream, None]:
        """Accumulate streamed content and block if a complete fenced code block is detected."""
        accumulated = ""
        async for item in response:
            if isinstance(item, ModelResponseStream) and item.choices:
                delta_content = ""
                is_final = False
                for choice in item.choices:
                    if hasattr(choice, "delta") and choice.delta:
                        content = getattr(choice.delta, "content", None)
                        if content and isinstance(content, str):
                            delta_content += content
                    if getattr(choice, "finish_reason", None):
                        is_final = True
                accumulated += delta_content
                if is_final:
                    # Run detection on full accumulated text (streaming: block only, no mask)
                    blocks = self._find_blocks(accumulated)
                    for _tag, _body, confidence, action_taken in blocks:
                        if (
                            action_taken == "block"
                            and confidence >= self.confidence_threshold
                        ):
                            lang = _tag or "unknown"
                            self._raise_block_error(lang, True, request_data)
            yield item
