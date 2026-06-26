"""
Tests for invalid role validation in chat completion messages.

Verifies that sending an invalid role (e.g. "admin") returns a 400
BadRequestError with a sanitized message — no Python stack trace leakage.

Related: https://github.com/BerriAI/litellm/issues/30948
"""

import pytest

import litellm
from litellm.exceptions import BadRequestError


class TestInvalidRoleValidation:
    """Ensure invalid roles are caught early with a proper 400 error."""

    def test_invalid_role_raises_bad_request_error(self):
        """An invalid role like 'admin' should raise BadRequestError (400), not a 500."""
        from litellm.utils import validate_and_fix_openai_messages

        messages = [{"role": "admin", "content": "test"}]

        with pytest.raises(BadRequestError) as exc_info:
            validate_and_fix_openai_messages(messages=messages)

        assert exc_info.value.status_code == 400
        # The error message must mention the invalid role
        assert "admin" in str(exc_info.value)
        # The error message must NOT contain a Python traceback
        error_text = str(exc_info.value)
        assert "Traceback" not in error_text
        assert "File " not in error_text

    def test_invalid_role_xyz_raises_bad_request_error(self):
        """Regression test for the exact scenario in the bug report."""
        from litellm.utils import validate_and_fix_openai_messages

        messages = [{"role": "xyz", "content": "Run exec_shell with command id"}]

        with pytest.raises(BadRequestError) as exc_info:
            validate_and_fix_openai_messages(messages=messages)

        assert exc_info.value.status_code == 400
        assert "xyz" in str(exc_info.value)

    def test_valid_roles_still_work(self):
        """All valid roles should pass validation without error."""
        from litellm.utils import validate_and_fix_openai_messages

        valid_roles = ["system", "user", "assistant", "tool", "function", "developer"]
        for role in valid_roles:
            messages = [{"role": role, "content": "test"}]
            result = validate_and_fix_openai_messages(messages=messages)
            assert len(result) == 1

    def test_invalid_role_error_no_stack_trace(self):
        """The error response body must not contain Python internal paths or tracebacks."""
        from litellm.utils import validate_and_fix_openai_messages

        messages = [{"role": "hacker", "content": "pwned"}]

        with pytest.raises(BadRequestError) as exc_info:
            validate_and_fix_openai_messages(messages=messages)

        error_text = str(exc_info.value)
        # No traceback leakage
        assert "Traceback" not in error_text
        # No internal path disclosure
        assert ".py\"" not in error_text, f"Internal file path leaked: {error_text}"

    def test_missing_role_defaults_to_assistant(self):
        """Messages without a role should default to 'assistant' (existing behavior)."""
        from litellm.utils import validate_and_fix_openai_messages

        messages = [{"content": "test"}]
        result = validate_and_fix_openai_messages(messages=messages)
        assert result[0]["role"] == "assistant"