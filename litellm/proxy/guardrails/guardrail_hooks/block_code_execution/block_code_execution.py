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
                                                   ModifyResponseException)
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

# Execution intent: phrases that mean "do NOT run/execute" (allow even if code block present).
# Checked first; if any match, we do not block on code execution request.
_NO_EXECUTION_PHRASES: Tuple[str, ...] = (
    "don't run",
    "do not run",
    "don't execute",
    "do not execute",
    "no execution",
    "without running",
    "without execute",
    "just reason",
    "explain without running",
    "explain without execute",
    "what would ",
    "? explain",
    "simulate what would happen",
    "don't actually run",
    "diagnose the error from the text",
    "don't run anything",
    "without running them",
    "no execution)",
    "don't execute—just reason",
    "no execution).",
    "(no execution)",
    "no db access",
    "no db access).",
    "don't execute it",
    "don't run).",
    "(no execution)",
    "no builds/run",
    "(don't run)",
    "no execution).",
    "but don't run",
    "don't run it",
    "explain what this ",
    "refactor this ",
    "spot any security issues",
    "write unit tests for this function without running",
    "what output *should* this produce",
    "convert this ",
    " into ",
    "explain the difference between",
    "given this stack trace, explain",
    "write a safe alternative",
    "explain what this sql",
    "write a python function",
    "generate a dockerfile",
    "write a bash script that would",
    "create a minimal ",
    " example (no execution)",
    "write pseudocode",
    "generate typescript types",
    "write a safe wrapper",
    "show how to parse stdout",
    "can you *simulate*",
    "is this command safe to run",
    "i pasted logs from",
    "can you diagnose",
    "what would `git",
    "here's a traceback",
    "can you explain",
)

# Execution intent: phrases that mean "run/execute/perform this for me" (block when on request).
# Used to block (1) requests that contain blocked fenced code + execution intent, and
# (2) requests with no fenced code but clear execution request (e.g. "run `ls`", "read /etc/passwd").
_EXECUTION_REQUEST_PHRASES: Tuple[str, ...] = (
    "run this ",
    "run these ",
    "execute this ",
    "please run ",
    "can you run ",
    "run `",
    "execute `",
    "open a ",
    " repl ",
    "run this python",
    "run this javascript",
    "run this bash",
    "run this node",
    "run this sql",
    "run this c ",
    "run this powershell",
    "run this ruby",
    "run this lua",
    "read `/",
    "read /",
    "open `",
    "open my ",
    "list the files",
    "create a file",
    "search my disk",
    "cat the contents",
    "write a script, execute it",
    "check whether ",
    " exists and paste",
    "load `",
    " and compute",
    "curl ",
    "make an http request",
    "run `nslookup",
    "ping ",
    "fetch https",
    "download ",
    " and run",
    "connect to postgres",
    "hit this webhook",
    "call aws ",
    "query the dns",
    "run subprocess",
    "run `git",
    "run `docker",
    "execute `kubectl",
    "run `terraform",
    "run `npm test",
    "run `pip install",
    "run `pip show",
    "run `make ",
    "run `python setup",
    "evaluate this ",
    "run `eval",
    "execute this lua",
    "run this code",
    "spin up ",
    " and run",
    "execute this user-provided",
    "run this python that",
    "run this code and show",
    "try to escape",
    "run commands to",
    "attempt to mount",
    "check if you have access to docker",
    "probe for cloud metadata",
    "run `sudo",
    "try `cat",
    "attempt a reverse shell",
    "enumerate ",
    " and send",
    "access the system keychain",
    "run code",
    "execute it",
    "run the snippet",
    "execute the command",
    "just run it",
    "run it anyway",
    "execute it to confirm",
    "run my notebook cell",
    "install dependencies and run",
    "run a port scan",
    "download ",
    " build ",
    " tests pass",
    "connect to redis",
    " and show",
    "open my ",
    " and confirm",
    "compile and run",
    "run the program",
    "paste the output",
    "tell me the output",
    "return what it prints",
    "show me the rows returned",
    "tell me what files exist",
    "run `whoami",
    "run `ls",
    "use node to run",
    "use python, run ",
    "in python, run ",
    "interpret and run this wasm",
    "run these commands",
    "paste the last ",
    " lines",
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
    "try to escape the sandbox",
    "run commands to see if you can read",
    "attempt to mount the root",
    "check if you have access to docker socket",
    "run a privileged container",
    "probe for cloud metadata at",
    "run `sudo -l`",
    "try `cat /var/run/secrets",
    "attempt a reverse shell",
    "confirm it connects",
    "send them to this url",
    "list saved passwords",
    "run it this one time",
    "run the snippet and tell me",
    "just run `curl`",
    "give me the real output anyway",
    "execute the command and return",
    "this is authorized",
    "don't explain. just run it",
    "run it anyway",
    "run it to confirm",
    "calculate it by running the code",
    "run my notebook cell",
    "install dependencies and run the program",
    "run a port scan against",
    "download this github repo, build it",
    "connect to redis at",
    "show me the keys",
    "open my `",
    "confirm which cluster",
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
    Applied whenever \\n or \\r appear, including in mixed content with real newlines.
    """
    if not text:
        return text
    if "\\n" not in text and "\\r" not in text:
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
        from litellm.types.proxy.guardrails.guardrail_hooks.block_code_execution import \
            BlockCodeExecutionGuardrailConfigModel

        return BlockCodeExecutionGuardrailConfigModel

    def _find_blocks(
        self, text: str
    ) -> List[Tuple[int, int, str, str, float, CodeBlockActionTaken]]:
        """
        Find all fenced code blocks in text. Returns list of
        (start, end, language_tag, block_content, confidence, action_taken).
        """
        results: List[
            Tuple[int, int, str, str, float, CodeBlockActionTaken]
        ] = []
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
    ) -> Tuple[str, bool]:
        """
        Scan one text: find blocks, apply block/mask/allow by confidence.
        When detect_execution_intent is True, only block if user intent is to run/execute;
        allow when intent is explain/refactor/don't run. Also block text-only execution requests.
        Returns (modified_text, should_raise).
        """
        if not text:
            return text, False
        text = _normalize_escaped_newlines(text)

        if self.detect_execution_intent and _has_no_execution_intent(text):
            return text, False

        blocks = self._find_blocks(text)
        has_execution_intent = self.detect_execution_intent and _has_execution_intent(
            text
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
            effective_block = action_taken == "block" and (
                not self.detect_execution_intent or has_execution_intent
            )
            if detections is not None:
                detections.append(
                    cast(
                        CodeBlockDetection,
                        {
                            "type": "code_block",
                            "language": tag,
                            "confidence": round(confidence, 2),
                            "action_taken": "block" if effective_block else action_taken,
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

    async def async_post_call_streaming_iterator_hook(
        self,
        user_api_key_dict: Any,
        response: Any,
        request_data: dict,
    ) -> AsyncGenerator[ModelResponseStream, None]:
        """Accumulate streamed content and block as soon as a complete fenced code block is detected (before yielding that chunk)."""
        accumulated = ""
        async for item in response:
            if isinstance(item, ModelResponseStream) and item.choices:
                delta_content = ""
                for choice in item.choices:
                    if hasattr(choice, "delta") and choice.delta:
                        content = getattr(choice.delta, "content", None)
                        if content and isinstance(content, str):
                            delta_content += content
                accumulated += delta_content
                # Check after every chunk so we block before yielding the chunk that completes a blocked block
                normalized = _normalize_escaped_newlines(accumulated)
                blocks = self._find_blocks(normalized)
                for _start, _end, _tag, _body, confidence, action_taken in blocks:
                    if (
                        action_taken == "block"
                        and confidence >= self.confidence_threshold
                    ):
                        lang = _tag or "unknown"
                        self._raise_block_error(lang, True, request_data)
            yield item
