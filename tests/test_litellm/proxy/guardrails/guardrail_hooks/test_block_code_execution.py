"""Tests for the Block Code Execution guardrail."""

import pytest
from fastapi import HTTPException

from litellm.integrations.custom_guardrail import ModifyResponseException
from litellm.proxy.guardrails.guardrail_hooks.block_code_execution import (
    DEFAULT_EVENT_HOOKS, BlockCodeExecutionGuardrail, initialize_guardrail)
from litellm.proxy.guardrails.guardrail_hooks.block_code_execution.block_code_execution import \
    _normalize_escaped_newlines
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
        tag, _body, confidence, action_taken = blocks[0]
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
        _tag, _body, confidence, action_taken = blocks[0]
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
        _tag, _body, confidence, action_taken = blocks[0]
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
        _tag, _body, confidence, action_taken = blocks[0]
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
        assert blocks[0][0] == "python"
        assert blocks[0][3] == "block"
        assert blocks[1][0] == "python"
        assert blocks[1][3] == "block"

    def test_scan_text_blocks_and_masks_when_text_has_escaped_newlines(self):
        """_scan_text detects blocks and applies block/mask when newlines are literal \\n."""
        guardrail = BlockCodeExecutionGuardrail(
            guardrail_name="test",
            blocked_languages=["python"],
            action="mask",
            confidence_threshold=0.5,
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

    @pytest.mark.asyncio
    async def test_streaming_hook_blocks_before_yielding_chunk_that_completes_block(
        self,
    ):
        """Streaming hook runs block check after every chunk and raises before yielding the chunk that completes a blocked fenced block."""
        from litellm.types.utils import (Delta, ModelResponseStream,
                                         StreamingChoices)

        guardrail = BlockCodeExecutionGuardrail(
            guardrail_name="test",
            blocked_languages=["python"],
            action="block",
            confidence_threshold=0.5,
        )

        # Chunks that form "```python\nprint(1)\n```" when concatenated
        async def mock_stream():
            yield ModelResponseStream(
                choices=[StreamingChoices(delta=Delta(content="```python\n"))],
            )
            yield ModelResponseStream(
                choices=[StreamingChoices(delta=Delta(content="print(1)\n"))],
            )
            yield ModelResponseStream(
                choices=[StreamingChoices(delta=Delta(content="```"))],
            )

        request_data = {"model": "gpt-4", "metadata": {}}
        yielded_chunks = []

        with pytest.raises(HTTPException) as exc_info:
            async for chunk in guardrail.async_post_call_streaming_iterator_hook(
                user_api_key_dict=None,
                response=mock_stream(),
                request_data=request_data,
            ):
                yielded_chunks.append(chunk)

        assert exc_info.value.status_code == 400
        assert "code block" in (exc_info.value.detail or {}).get("error", "")
        # The chunk that completes the block (third chunk) must not have been yielded
        assert len(yielded_chunks) == 2
