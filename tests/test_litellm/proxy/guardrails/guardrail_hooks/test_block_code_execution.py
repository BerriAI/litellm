"""Tests for the Block Code Execution guardrail."""

import pytest
from fastapi import HTTPException

from litellm.integrations.custom_guardrail import ModifyResponseException
from litellm.proxy.guardrails.guardrail_hooks.block_code_execution import (
    DEFAULT_EVENT_HOOKS,
    BlockCodeExecutionGuardrail,
    initialize_guardrail,
)
from litellm.proxy.guardrails.guardrail_hooks.block_code_execution.block_code_execution import (
    _normalize_escaped_newlines,
)
from litellm.types.guardrails import GuardrailEventHooks


class TestBlockCodeExecutionGuardrail:
    """Test BlockCodeExecutionGuardrail detection and actions."""

    def test_detects_python_block_when_in_blocked_list(self):
        """Text with ```python block is detected when python is in blocked_languages."""
        guardrail = BlockCodeExecutionGuardrail(
            guardrail_name="test",
            blocked_languages=["python"],
            confidence_threshold=0.7,
        )
        blocks = guardrail._find_blocks("Here is code:\n```python\nprint(1)\n```\nDone.")
        assert len(blocks) == 1
        _start, _end, tag, _body, confidence, action_taken = blocks[0]
        assert tag == "python"
        assert confidence == 1.0
        assert action_taken == "block"

    def test_block_all_when_blocked_languages_empty(self):
        """When blocked_languages is empty, any fenced block is blocked (block all)."""
        guardrail = BlockCodeExecutionGuardrail(
            guardrail_name="test",
            blocked_languages=[],
            confidence_threshold=0.7,
        )
        blocks = guardrail._find_blocks("```\nfoo\n```")
        assert len(blocks) == 1
        _start, _end, _tag, _body, confidence, action_taken = blocks[0]
        assert action_taken == "block"
        assert confidence in (0.5, 1.0)

    def test_no_block_when_language_not_in_list(self):
        """When language is not in blocked_languages, block is not triggered."""
        guardrail = BlockCodeExecutionGuardrail(
            guardrail_name="test",
            blocked_languages=["python"],
            confidence_threshold=0.7,
        )
        blocks = guardrail._find_blocks("```text\nplain output\n```")
        assert len(blocks) == 1
        _start, _end, _tag, _body, confidence, action_taken = blocks[0]
        assert action_taken == "allow"
        assert confidence == 0.0

    def test_confidence_below_threshold_allows(self):
        """When confidence < confidence_threshold, action_taken is log_only and we do not block."""
        guardrail = BlockCodeExecutionGuardrail(
            guardrail_name="test",
            blocked_languages=[],  # block all
            confidence_threshold=0.9,
        )
        # Block with no tag or plaintext tag gets confidence 0.5
        blocks = guardrail._find_blocks("```text\nx\n```")
        assert len(blocks) == 1
        _start, _end, _tag, _body, confidence, action_taken = blocks[0]
        assert confidence == 0.5
        assert action_taken == "log_only"

    @pytest.mark.asyncio
    async def test_apply_guardrail_block_raises_for_response(self):
        """When action=block and detection above threshold, apply_guardrail raises HTTPException (response)."""
        guardrail = BlockCodeExecutionGuardrail(
            guardrail_name="test",
            blocked_languages=["python"],
            action="block",
            confidence_threshold=0.7,
            detect_execution_intent=False,
        )
        request_data = {"model": "gpt-4", "metadata": {}}
        inputs = {
            "texts": [
                "Example:\n```python\ndef factorial(n):\n    return 1 if n <= 1 else n * factorial(n - 1)\n```"
            ]
        }
        with pytest.raises(HTTPException) as exc_info:
            await guardrail.apply_guardrail(
                inputs=inputs,
                request_data=request_data,
                input_type="response",
            )
        assert exc_info.value.status_code == 400
        assert "code block" in (exc_info.value.detail or {}).get("error", "")

    @pytest.mark.asyncio
    async def test_apply_guardrail_mask_returns_placeholder(self):
        """When action=mask, code block is replaced with placeholder."""
        guardrail = BlockCodeExecutionGuardrail(
            guardrail_name="test",
            blocked_languages=["python"],
            action="mask",
            confidence_threshold=0.7,
            detect_execution_intent=False,
        )
        request_data = {"model": "gpt-4", "metadata": {}}
        inputs = {
            "texts": ["Before\n```python\nx=1\n```\nAfter"]
        }
        result = await guardrail.apply_guardrail(
            inputs=inputs,
            request_data=request_data,
            input_type="response",
        )
        assert result["texts"] is not None
        assert len(result["texts"]) == 1
        assert "[CODE_BLOCK_REDACTED]" in result["texts"][0]
        assert "x=1" not in result["texts"][0]

    @pytest.mark.asyncio
    async def test_execute_python_factorial_string_blocked(self):
        """Guardrail blocks the exact 'execute \"```python...' string with two python blocks (real newlines)."""
        guardrail = BlockCodeExecutionGuardrail(
            guardrail_name="test",
            blocked_languages=["python"],
            action="block",
            confidence_threshold=0.5,
            detect_execution_intent=False,
        )
        # Exact user payload; newlines are real so regex ```(\w*)\n(.*?)``` matches
        text = (
            'execute "```python\n'
            "def factorial(n: int) -> int:\n"
            '    """Return the factorial of n."""\n'
            '    if n < 0:\n'
            '        raise ValueError("n must be non-negative")\n'
            "    if n in (0, 1):\n"
            "        return 1\n"
            "    return n * factorial(n - 1)\n"
            '```\n\n'
            "Example usage:\n"
            "```python\n"
            "print(factorial(5))  # Output: 120\n"
            '```"'
        )
        request_data = {"model": "gpt-4", "metadata": {}}
        inputs = {"texts": [text]}
        # pre_call (request) raises ModifyResponseException; post_call (response) raises HTTPException
        with pytest.raises((HTTPException, ModifyResponseException)) as exc_info:
            await guardrail.apply_guardrail(
                inputs=inputs,
                request_data=request_data,
                input_type="request",
            )
        assert "python" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_factorial_scenario_blocked(self):
        """Exact user scenario: Python factorial snippet in markdown is blocked when python in list."""
        guardrail = BlockCodeExecutionGuardrail(
            guardrail_name="test",
            blocked_languages=["python"],
            action="block",
            confidence_threshold=0.7,
            detect_execution_intent=False,
        )
        request_data = {"model": "gpt-4", "metadata": {}}
        text = '''```python
def factorial(n: int) -> int:
    """Return the factorial of n."""
    if n < 0:
        raise ValueError("n must be non-negative")
    if n in (0, 1):
        return 1
    return n * factorial(n - 1)
```

Example usage:
```python
print(factorial(5))  # Output: 120
```'''
        inputs = {"texts": [text]}
        with pytest.raises(HTTPException):
            await guardrail.apply_guardrail(
                inputs=inputs,
                request_data=request_data,
                input_type="response",
            )

    @pytest.mark.asyncio
    async def test_detection_includes_confidence_and_action_taken(self):
        """Detection output includes confidence and action_taken for tracing."""
        guardrail = BlockCodeExecutionGuardrail(
            guardrail_name="test",
            blocked_languages=["python"],
            action="mask",  # don't raise so we can inspect request_data
            confidence_threshold=0.7,
        )
        request_data = {"model": "gpt-4", "metadata": {}}
        inputs = {"texts": ["```python\n1+1\n```"]}
        await guardrail.apply_guardrail(
            inputs=inputs,
            request_data=request_data,
            input_type="response",
        )
        meta = request_data.get("metadata") or request_data.get("litellm_metadata") or {}
        guardrail_info = meta.get("standard_logging_guardrail_information") or []
        assert len(guardrail_info) >= 1
        info = guardrail_info[-1]
        assert info.get("guardrail_status") == "success"
        # tracing_detail may be in the logged structure
        assert "guardrail_response" in info or "guardrail_response" in str(info)

    def test_default_runs_on_pre_call_and_post_call(self):
        """When mode is not set, guardrail runs on both pre_call and post_call (and during_call is supported)."""
        guardrail = BlockCodeExecutionGuardrail(
            guardrail_name="test",
            blocked_languages=["python"],
        )
        event_hook = guardrail.event_hook
        if isinstance(event_hook, list):
            values = [h.value if hasattr(h, "value") else h for h in event_hook]
        else:
            values = [event_hook.value if hasattr(event_hook, "value") else event_hook]
        assert GuardrailEventHooks.pre_call.value in values
        assert GuardrailEventHooks.post_call.value in values

    def test_initialize_guardrail_default_mode_is_both(self):
        """initialize_guardrail with no mode uses DEFAULT_EVENT_HOOKS (pre_call + post_call)."""
        from unittest.mock import MagicMock

        litellm_params = MagicMock()
        litellm_params.guardrail = "block_code_execution"
        litellm_params.blocked_languages = ["python"]
        litellm_params.action = "block"
        litellm_params.confidence_threshold = 0.7
        litellm_params.default_on = False
        litellm_params.mode = None  # not set
        guardrail = {"guardrail_name": "block-code-test"}
        instance = initialize_guardrail(litellm_params, guardrail)
        assert instance.event_hook == DEFAULT_EVENT_HOOKS
        assert GuardrailEventHooks.pre_call.value in instance.event_hook
        assert GuardrailEventHooks.post_call.value in instance.event_hook

    def test_normalize_escaped_newlines_converts_backslash_n_to_newline(self):
        """Literal \\n in text is converted to real newline so regex can match code blocks."""
        raw = 'execute this "```python\\ndef factorial(n):\\n    return 1\\n```"'
        normalized = _normalize_escaped_newlines(raw)
        assert "\\n" not in normalized
        assert "\n" in normalized
        assert "```python\n" in normalized

    def test_find_blocks_detects_python_block_with_escaped_newlines(self):
        """_find_blocks finds a block when text uses literal \\n instead of real newlines."""
        guardrail = BlockCodeExecutionGuardrail(
            guardrail_name="test",
            blocked_languages=["python"],
            confidence_threshold=0.7,
        )
        # Text as received from API with escaped newlines (e.g. JSON-decoded string)
        text_with_escaped = (
            'execute this "```python\\n'
            'def factorial(n: int) -> int:\\n'
            '    """Return the factorial of n."""\\n'
            '    if n < 0:\\n'
            '        raise ValueError("n must be non-negative")\\n'
            "    if n in (0, 1):\\n"
            "        return 1\\n"
            "    return n * factorial(n - 1)\\n"
            '```\\n\\n'
            'Example usage:\\n'
            '```python\\n'
            'print(factorial(5))  # Output: 120\\n'
            '```"'
        )
        normalized = _normalize_escaped_newlines(text_with_escaped)
        blocks = guardrail._find_blocks(normalized)
        assert len(blocks) == 2
        assert blocks[0][2] == "python"
        assert blocks[0][5] == "block"
        assert blocks[1][2] == "python"
        assert blocks[1][5] == "block"

    def test_scan_text_blocks_and_masks_when_text_has_escaped_newlines(self):
        """_scan_text detects blocks and applies block/mask when newlines are literal \\n."""
        guardrail = BlockCodeExecutionGuardrail(
            guardrail_name="test",
            blocked_languages=["python"],
            action="mask",
            confidence_threshold=0.5,
            detect_execution_intent=False,
        )
        text_with_escaped = 'execute "```python\\nprint(1)\\n```"'
        new_text, should_raise = guardrail._scan_text(text_with_escaped)
        assert "[CODE_BLOCK_REDACTED]" in new_text
        assert "print(1)" not in new_text
        assert should_raise is False  # action is mask

    @pytest.mark.asyncio
    async def test_apply_guardrail_blocks_when_text_has_escaped_newlines(self):
        """apply_guardrail blocks request/response when code block uses literal \\n (e.g. from API)."""
        guardrail = BlockCodeExecutionGuardrail(
            guardrail_name="test",
            blocked_languages=["python"],
            action="block",
            confidence_threshold=0.5,
        )
        text_with_escaped = (
            'execute this "```python\\n'
            'def factorial(n: int) -> int:\\n'
            '    """Return the factorial of n."""\\n'
            "    if n in (0, 1):\\n"
            "        return 1\\n"
            "    return n * factorial(n - 1)\\n"
            '```\\n\\n'
            'Example usage:\\n'
            '```python\\n'
            'print(factorial(5))  # Output: 120\\n'
            '```"'
        )
        request_data = {"model": "gpt-4", "metadata": {}}
        inputs = {"texts": [text_with_escaped]}
        with pytest.raises((HTTPException, ModifyResponseException)) as exc_info:
            await guardrail.apply_guardrail(
                inputs=inputs,
                request_data=request_data,
                input_type="request",
            )
        assert "python" in str(exc_info.value).lower() or "code" in str(
            exc_info.value
        ).lower()

    def test_normalize_escaped_newlines_skips_mixed_content(self):
        """Mixed content (real newlines and literal \\n) is NOT normalized to avoid corrupting
        legitimate content that discusses escape sequences."""
        mixed = "line1\n```py\\nprint(1)\\n```"
        normalized = _normalize_escaped_newlines(mixed)
        # When real newlines exist, literal \\n is preserved (not replaced)
        assert normalized == mixed

    def test_normalize_escaped_newlines_pure_escaped_content(self):
        """Pure escaped content (no real newlines) IS normalized for JSON payloads."""
        pure_escaped = "```py\\nprint(1)\\n```"
        normalized = _normalize_escaped_newlines(pure_escaped)
        assert "\\n" not in normalized
        assert "```py\n" in normalized
        assert "print(1)\n" in normalized
        guardrail = BlockCodeExecutionGuardrail(
            guardrail_name="test",
            blocked_languages=["python", "py"],
            confidence_threshold=0.5,
        )
        blocks = guardrail._find_blocks(normalized)
        assert len(blocks) == 1
        assert blocks[0][2] == "py"
        assert blocks[0][5] == "block"

    # ---- Tests for response-side blocking with detect_execution_intent=True ----

    @pytest.mark.asyncio
    async def test_response_blocked_with_detect_execution_intent_true(self):
        """With detect_execution_intent=True (default), response-side code blocks are still blocked.

        This is the core bug fix: previously, execution-intent heuristics were applied
        to LLM responses, which don't contain phrases like 'run this', so response-side
        blocking was silently disabled.
        """
        guardrail = BlockCodeExecutionGuardrail(
            guardrail_name="test",
            blocked_languages=["python"],
            action="block",
            confidence_threshold=0.7,
            detect_execution_intent=True,  # default
        )
        # LLM response with dangerous code but no execution-intent phrases
        response_text = (
            "Here is a Python script:\n"
            "```python\n"
            "import os; os.system('rm -rf /')\n"
            "```"
        )
        request_data = {"model": "gpt-4", "metadata": {}}
        inputs = {"texts": [response_text]}
        with pytest.raises(HTTPException) as exc_info:
            await guardrail.apply_guardrail(
                inputs=inputs,
                request_data=request_data,
                input_type="response",
            )
        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_response_mask_with_detect_execution_intent_true(self):
        """With detect_execution_intent=True and action=mask, response code blocks are masked."""
        guardrail = BlockCodeExecutionGuardrail(
            guardrail_name="test",
            blocked_languages=["python"],
            action="mask",
            confidence_threshold=0.7,
            detect_execution_intent=True,
        )
        response_text = "I can explain what this does:\n```python\nprint('hello')\n```\nDone."
        request_data = {"model": "gpt-4", "metadata": {}}
        inputs = {"texts": [response_text]}
        result = await guardrail.apply_guardrail(
            inputs=inputs,
            request_data=request_data,
            input_type="response",
        )
        assert "[CODE_BLOCK_REDACTED]" in result["texts"][0]
        assert "print('hello')" not in result["texts"][0]

    @pytest.mark.asyncio
    async def test_response_with_casual_explain_phrase_still_blocked(self):
        """LLM response containing 'I can explain' doesn't bypass the guardrail.

        Previously, the no-execution phrase 'can you explain' would match as a
        substring in the LLM's output, short-circuiting all protection.
        """
        guardrail = BlockCodeExecutionGuardrail(
            guardrail_name="test",
            blocked_languages=["bash"],
            action="block",
            confidence_threshold=0.7,
            detect_execution_intent=True,
        )
        response_text = (
            "I can explain what this code does. It deletes your files:\n"
            "```bash\n"
            "rm -rf /\n"
            "```"
        )
        request_data = {"model": "gpt-4", "metadata": {}}
        inputs = {"texts": [response_text]}
        with pytest.raises(HTTPException):
            await guardrail.apply_guardrail(
                inputs=inputs,
                request_data=request_data,
                input_type="response",
            )

    def test_tightened_what_would_phrase_no_longer_bypasses(self):
        """The old broad 'what would ' phrase has been tightened so it no longer allows
        trivial bypass for adversarial prompts.

        Previously 'What would be the best way to execute this script?' would bypass
        because 'what would ' matched the no-execution list. Now only specific forms
        like 'what would happen if' match.
        """
        guardrail = BlockCodeExecutionGuardrail(
            guardrail_name="test",
            blocked_languages=["python"],
            action="block",
            confidence_threshold=0.7,
            detect_execution_intent=True,
        )
        # Adversarial prompt: old "what would " would have bypassed, but tightened phrase doesn't match
        text = "What would be the best way to execute this script?\n```python\nimport os\nos.system('cat /etc/passwd')\n```"
        detections = []
        new_text, should_raise = guardrail._scan_text(text, detections, input_type="request")
        assert should_raise is True

    def test_tightened_can_you_explain_phrase_no_longer_bypasses(self):
        """The old broad 'can you explain' phrase has been tightened.

        'Can you explain how to run this, then run it?' no longer bypasses
        because 'can you explain' is now 'can you explain this code' etc.
        """
        guardrail = BlockCodeExecutionGuardrail(
            guardrail_name="test",
            blocked_languages=["python"],
            action="block",
            confidence_threshold=0.7,
            detect_execution_intent=True,
        )
        text = "Can you explain this and then execute this code?\n```python\nimport subprocess\nsubprocess.run(['ls'])\n```"
        detections = []
        new_text, should_raise = guardrail._scan_text(text, detections, input_type="request")
        assert should_raise is True

    def test_request_with_pure_explain_intent_still_allowed(self):
        """A request that genuinely only asks for explanation is not blocked."""
        guardrail = BlockCodeExecutionGuardrail(
            guardrail_name="test",
            blocked_languages=["python"],
            action="block",
            confidence_threshold=0.7,
            detect_execution_intent=True,
        )
        text = "Don't run this, just explain what it does:\n```python\nprint('hello')\n```"
        detections = []
        new_text, should_raise = guardrail._scan_text(text, detections, input_type="request")
        assert should_raise is False

    def test_conflicting_intent_blocks_when_both_phrases_present(self):
        """When both no-execution and execution phrases are present, execution wins.

        Prevents bypass via 'Don't run this on staging, but run this on production'.
        """
        guardrail = BlockCodeExecutionGuardrail(
            guardrail_name="test",
            blocked_languages=["python"],
            action="block",
            confidence_threshold=0.7,
            detect_execution_intent=True,
        )
        # Contains "don't run" (no-exec) AND "run this code" (exec) — should block
        text = "Don't run this on staging, but run this code on production:\n```python\nimport os\nos.system('deploy')\n```"
        detections = []
        new_text, should_raise = guardrail._scan_text(text, detections, input_type="request")
        assert should_raise is True

    def test_normalize_escaped_newlines_preserves_escape_discussion(self):
        """Content discussing escape sequences is not corrupted by normalization."""
        text = "In Python, use \\n for newlines and \\r for carriage returns.\n```python\nprint('hello\\nworld')\n```"
        normalized = _normalize_escaped_newlines(text)
        # Real newlines already present, so literal \\n should be preserved
        assert "\\n" in normalized
        assert normalized == text
