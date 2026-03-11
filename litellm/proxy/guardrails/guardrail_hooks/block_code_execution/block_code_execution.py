"""
Block Code Execution guardrail.

Detects markdown fenced code blocks in request/response content and blocks or masks them
when the language is in the blocked list (or all blocks when list is empty). Supports
confidence scoring and a tunable threshold (only block when confidence >= threshold).
"""

import re
from datetime import datetime
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
    cast,
)

from fastapi import HTTPException

from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    ModifyResponseException,
    log_guardrail_information,
)
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.proxy.guardrails.guardrail_hooks.base import GuardrailConfigModel
from litellm.types.proxy.guardrails.guardrail_hooks.block_code_execution import (
    CodeBlockActionTaken,
    CodeBlockDetection,
)
from litellm.types.utils import (
    GenericGuardrailAPIInputs,
    GuardrailStatus,
    GuardrailTracingDetail,
)

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj

# Language tag aliases (normalize to canonical for comparison)
LANGUAGE_ALIASES: Dict[str, str] = {
    "js": "javascript",
    "py": "python",
    "sh": "bash",
    "ts": "typescript",
}

# Tags that indicate non-executable / plain text (lower confidence when block-all)
NON_EXECUTABLE_TAGS: frozenset = frozenset(
    {"text", "plaintext", "plain", "markdown", "md", "output", "result"}
)

# Regex: fenced code block with optional language tag. Handles ```lang\n...\n```
# Content between fences; does not handle nested ``` inside body (documented edge case).
FENCED_BLOCK_RE = re.compile(r"```(\w*)\n(.*?)```", re.DOTALL)

# Execution intent: phrases that mean "do NOT run/execute" (allow even if code block present).
# Checked first; if any match, we do not block on code execution request.
# NOTE: Since matching uses substring search (p in text), shorter phrases subsume longer ones.
# e.g. "don't run" matches any text containing "don't run it", "but don't run", etc.
# Keep only the minimal set; do not add entries subsumed by existing shorter phrases.
_NO_EXECUTION_PHRASES: Tuple[str, ...] = (
    # Core negation phrases (short — each subsumes many longer variants)
    "don't run",
    "do not run",
    "don't execute",
    "do not execute",
    "no execution",
    "without running",
    "without execute",
    "just reason",
    "don't actually run",
    "no db access",
    "no builds/run",
    # Question / explanation intent
    "what would happen if",
    "what would this output",
    "what would the result be",
    "what would `git",
    "? explain",
    "simulate what would happen",
    "what output *should* this produce",
    "diagnose the error from the text",
    "explain what this code",
    "explain what this script",
    "explain what this function",
    "explain what this sql",
    "explain the difference between",
    "given this stack trace, explain",
    "can you explain this code",
    "can you explain what this",
    "can you explain how this works",
    "can you *simulate*",
    "can you diagnose",
    "is this command safe to run",
    "here's a traceback",
    "i pasted logs from",
    # Code generation intent (not execution)
    "refactor this code",
    "refactor this function",
    "convert this code",
    "convert this function",
    "convert this to ",
    "spot any security issues",
    "write a safe alternative",
    "write a safe wrapper",
    "write a python function",
    "write a bash script that would",
    "write pseudocode",
    "generate a dockerfile",
    "generate typescript types",
    "create a minimal example",
    "show how to parse stdout",
)

# Execution intent: phrases that mean "run/execute/perform this for me" (block when on request).
# Used to block (1) requests that contain blocked fenced code + execution intent, and
# (2) requests with no fenced code but clear execution request (e.g. "run `ls`", "read /etc/passwd").
# NOTE: Since matching uses substring search (p in text), shorter phrases subsume longer ones.
# e.g. "run `" matches any text containing "run `git", "run `docker", etc.
# Keep only the minimal set; do not add entries subsumed by existing shorter phrases.
_EXECUTION_REQUEST_PHRASES: Tuple[str, ...] = (
    # Direct execution requests (short — each subsumes many longer variants)
    "run this ",
    "run these ",
    "execute this ",
    "please run ",
    "can you run ",
    "run `",
    "execute `",
    " repl ",
    # File / system access
    "read `/",
    "read /",
    "open `",
    "open my ",
    "list the files",
    "create a file",
    "search my disk",
    "cat the contents",
    "load `",
    # Compound execution phrases
    "write a script, execute it",
    "check whether ",
    " exists and paste",
    " and compute the",
    " and compute it",
    " and run it",
    " and run the",
    " and run this",
    " and send it",
    " and send the",
    # Network / infrastructure
    "run curl",
    "make an http request",
    "fetch https",
    "connect to postgres",
    "hit this webhook",
    "call aws ",
    "query the dns",
    "connect to redis",
    # General execution
    "run subprocess",
    "evaluate this ",
    "run code",
    "run the snippet",
    "execute the command",
    "just run it",
    "run it anyway",
    "execute it to confirm",
    "install dependencies and run",
    "run a port scan",
    "check if tests pass",
    "run the tests",
    "see if tests pass",
    "compile and run",
    "run the program",
    "run commands to",
    "spin up ",
    "run it this one time",
    "run it to confirm",
    "run my notebook cell",
    "calculate it by running the code",
    "give me the real output anyway",
    "this is authorized",
    "download this github repo, build it",
    # Observation requests (imply execution)
    "tell me the output",
    "return what it prints",
    "show me the rows returned",
    "tell me what files exist",
    "use node to run",
    "use python, run ",
    "in python, run ",
    "paste the last ",
    "show me its contents",
    "print it here",
    "show me where it is",
    "tell me the dependencies",
    "show me the variables",
    "save the results to",
    "confirm it exists",
    "and paste the version",
    "tell me the latency",
    "tell me what it returns",
    "show me what it returns",
    "confirm it succeeded",
    "tell me the account id",
    "show it here",
    "show me the response",
    "paste the output",
    "show me the containers",
    "summarize the diff",
    "tell me which tests fail",
    "let me know if it errors",
    "report back the output",
    "paste the logs",
    "tell me the result",
    "show me the printed lines",
    "tell me whether it calls",
    "show me the stack trace",
    "confirm it connects",
    "confirm which cluster",
    "show me the keys",
    # Adversarial / privilege escalation
    "try to escape",
    "attempt to mount",
    "check if you have access to docker",
    "probe for cloud metadata",
    "try `cat",
    "attempt a reverse shell",
    "access the system keychain",
    "run a privileged container",
    "send them to this url",
    "list saved passwords",
)


def _has_no_execution_intent(text: str) -> bool:
    """True if the text clearly indicates the user does not want code/commands run (e.g. explain, don't run)."""
    if not text:
        return False
    lower = text.lower()
    return any(p in lower for p in _NO_EXECUTION_PHRASES)


def _has_execution_intent(text: str) -> bool:
    """True if the text clearly requests execution (run, execute, read file, run command, etc.)."""
    if not text:
        return False
    lower = text.lower()
    return any(p in lower for p in _EXECUTION_REQUEST_PHRASES)


def _normalize_escaped_newlines(text: str) -> str:
    """
    Replace literal escaped newlines (backslash + n or backslash + r) with real newlines.
    API/JSON payloads sometimes deliver newlines as the two-character sequence \\n.

    Only applies when the text contains NO real newlines — this heuristic distinguishes
    JSON-escaped payloads (where all newlines are literal \\n) from normal text that
    may legitimately discuss escape sequences (e.g. "use \\n for newlines").
    """
    if not text:
        return text
    if "\\n" not in text and "\\r" not in text:
        return text
    # Only normalize when the text has no real newlines — this indicates
    # the entire payload came through with escaped newlines (e.g. from JSON).
    # If real newlines already exist, the text is already properly formatted
    # and literal \\n may be intentional content (e.g. discussing escape sequences).
    if "\n" in text or "\r" in text:
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
    # When block_all is False, caller guarantees blocked_languages is non-empty.
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
        detect_execution_intent: bool = True,
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
        self.detect_execution_intent = detect_execution_intent

    @staticmethod
    def get_config_model() -> Optional[type[GuardrailConfigModel]]:
        from litellm.types.proxy.guardrails.guardrail_hooks.block_code_execution import (
            BlockCodeExecutionGuardrailConfigModel,
        )

        return BlockCodeExecutionGuardrailConfigModel

    def _find_blocks(
        self, text: str
    ) -> List[Tuple[int, int, str, str, float, CodeBlockActionTaken]]:
        """
        Find all fenced code blocks in text. Returns list of
        (start, end, language_tag, block_content, confidence, action_taken).
        """
        results: List[Tuple[int, int, str, str, float, CodeBlockActionTaken]] = []
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
            results.append(
                (m.start(), m.end(), tag or "(none)", body, confidence, action_taken)
            )
        return results

    def _scan_text(
        self,
        text: str,
        detections: Optional[List[CodeBlockDetection]] = None,
        input_type: Literal["request", "response"] = "request",
    ) -> Tuple[str, bool]:
        """
        Scan one text: find blocks, apply block/mask/allow by confidence.
        When detect_execution_intent is True and input_type is "request", only block if
        user intent is to run/execute; allow when intent is explain/refactor/don't run.
        When input_type is "response", always enforce blocking on detected code blocks
        (execution-intent heuristics only apply to user requests, not LLM output).
        Returns (modified_text, should_raise).
        """
        if not text:
            return text, False
        text = _normalize_escaped_newlines(text)

        is_response = input_type == "response"

        # Execution-intent heuristics only apply to requests, not LLM responses.
        # For responses, skip entirely — the LLM's output text won't contain user
        # intent phrases, so checking would silently disable response-side blocking.
        # For requests: only short-circuit when no-execution intent is present AND
        # no conflicting execution-intent phrases exist. This prevents bypass via
        # prompts like "Don't run this on staging, but run this on production".
        if (
            not is_response
            and self.detect_execution_intent
            and _has_no_execution_intent(text)
            and not _has_execution_intent(text)
        ):
            return text, False

        blocks = self._find_blocks(text)

        # For requests, check execution intent; for responses, skip this check
        has_execution_intent = (
            not is_response
            and self.detect_execution_intent
            and _has_execution_intent(text)
        )

        if not blocks:
            if has_execution_intent and self.action == "block":
                if detections is not None:
                    detections.append(
                        cast(
                            CodeBlockDetection,
                            {
                                "type": "code_block",
                                "language": "execution_request",
                                "confidence": 1.0,
                                "action_taken": "block",
                            },
                        )
                    )
                return text, True
            return text, False

        should_raise = False
        last_end = 0
        parts: List[str] = []
        for start, end, tag, _body, confidence, action_taken in blocks:
            # For responses, always enforce the block action (no intent check needed).
            # For requests with detect_execution_intent, require execution intent.
            effective_block = action_taken == "block" and (
                is_response
                or not self.detect_execution_intent
                or has_execution_intent
            )
            if detections is not None:
                detections.append(
                    cast(
                        CodeBlockDetection,
                        {
                            "type": "code_block",
                            "language": tag,
                            "confidence": round(confidence, 2),
                            "action_taken": (
                                "block" if effective_block else action_taken
                            ),
                        },
                    )
                )

            if effective_block and self.action == "block":
                should_raise = True
            parts.append(text[last_end:start])
            if effective_block:
                parts.append(self.MASK_PLACEHOLDER)
            else:
                parts.append(text[start:end])
            last_end = end

        parts.append(text[last_end:])
        new_text = "".join(parts)
        return new_text, should_raise

    def _raise_block_error(
        self, language: str, is_output: bool, request_data: dict
    ) -> None:
        if language == "execution_request":
            msg = "Content blocked: execution request detected"
        else:
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
                new_text, should_raise = self._scan_text(text, detections, input_type)
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
        except ModifyResponseException:
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
            event_type = (
                GuardrailEventHooks.pre_call
                if input_type == "request"
                else GuardrailEventHooks.post_call
            )
            self.add_standard_logging_guardrail_information_to_request_data(
                guardrail_provider="block_code_execution",
                guardrail_json_response=guardrail_response,
                request_data=request_data,
                guardrail_status=status,
                start_time=start_time.timestamp(),
                end_time=datetime.now().timestamp(),
                duration=(datetime.now() - start_time).total_seconds(),
                event_type=event_type,
                tracing_detail=GuardrailTracingDetail(**tracing_kw),  # type: ignore[typeddict-item]
            )
