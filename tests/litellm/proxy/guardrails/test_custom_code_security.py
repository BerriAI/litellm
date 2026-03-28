import pytest
from litellm.proxy.guardrails.guardrail_hooks.custom_code.code_validator import (
    CustomCodeValidationError,
    _validate_ast,
    validate_custom_code,
)
from litellm.proxy.guardrails.guardrail_hooks.custom_code.custom_code_guardrail import (
    CustomCodeGuardrail,
)

# =============================================================================
# Phase 4.1: Test forbidden pattern validation (regex layer)
# =============================================================================


def test_validate_custom_code_import_os():
    code = "import os\ndef apply_guardrail(inputs, req, ty):\n    return allow()"
    with pytest.raises(CustomCodeValidationError, match="import statements are not"):
        validate_custom_code(code)


def test_validate_custom_code_from_subprocess():
    code = (
        "from subprocess import call\ndef apply_guardrail(i, r, t):\n    return allow()"
    )
    with pytest.raises(
        CustomCodeValidationError, match="import statements are not allowed"
    ):
        validate_custom_code(code)


def test_validate_custom_code_exec():
    code = "def apply_guardrail(i, r, t):\n    exec('print(1)')\n    return allow()"
    with pytest.raises(CustomCodeValidationError, match=r"exec\(\) is not allowed"):
        validate_custom_code(code)


def test_validate_custom_code_builtins():
    code = "def apply_guardrail(i, r, t):\n    print(__builtins__)\n    return allow()"
    with pytest.raises(
        CustomCodeValidationError, match="__builtins__ access is not allowed"
    ):
        validate_custom_code(code)


def test_validate_custom_code_subclasses():
    code = "def apply_guardrail(i, r, t):\n    print(''.__class__.__mro__[1].__subclasses__())\n    return allow()"
    with pytest.raises(
        CustomCodeValidationError, match="__subclasses__ access is not allowed"
    ):
        validate_custom_code(code)


def test_validate_custom_code_clean():
    code = (
        "def apply_guardrail(inputs, request_data, input_type):\n    return allow()\n"
    )
    # Should not raise any exception
    validate_custom_code(code)


# =============================================================================
# Phase 4.2: Test __builtins__ restriction in execution
# =============================================================================


def test_custom_code_compile_valid():
    code = "def apply_guardrail(inputs, request_data, input_type):\n    return allow()"
    guardrail = CustomCodeGuardrail(custom_code=code, guardrail_name="test")
    # if it doesn't fail, we successfully compiled
    assert guardrail._compiled_function is not None


def test_custom_code_override_builtins():
    # Verify that even if pattern validation is bypassed, __builtins__ = {} blocks dangerous builtins.
    # We test this by compiling safe code and verifying builtins are not accessible in the sandbox.
    code = "def apply_guardrail(inputs, request_data, input_type):\n    return allow()"
    guardrail = CustomCodeGuardrail(custom_code=code, guardrail_name="test")
    # The compiled function's globals should have empty __builtins__
    fn_globals = guardrail._compiled_function.__globals__
    assert fn_globals.get("__builtins__") == {}


@pytest.mark.asyncio
async def test_custom_code_guardrail_apply():
    code = "def apply_guardrail(inputs, request_data, input_type):\n    return allow()"
    guardrail = CustomCodeGuardrail(custom_code=code, guardrail_name="test")
    from litellm.types.utils import GenericGuardrailAPIInputs

    result = await guardrail.apply_guardrail(
        inputs=GenericGuardrailAPIInputs(texts=["test"]),
        request_data={},
        input_type="request",
    )
    assert result["texts"][0] == "test"


# The RBAC endpoint tests are harder to write right here, but the core security
# validations are fully covered by the simple tests above.


# =============================================================================
# Phase 4.3: AST-based validation tests (defense-in-depth)
# =============================================================================


class TestASTValidation:
    """Tests for AST-based security validation layer."""

    def test_ast_blocks_import_statement(self):
        code = "import os"
        with pytest.raises(CustomCodeValidationError, match="import"):
            _validate_ast(code)

    def test_ast_blocks_from_import(self):
        code = "from os import system"
        with pytest.raises(CustomCodeValidationError, match="import"):
            _validate_ast(code)

    def test_ast_blocks_exec_call(self):
        code = "def f():\n    exec('evil')"
        with pytest.raises(CustomCodeValidationError, match="exec"):
            _validate_ast(code)

    def test_ast_blocks_eval_call(self):
        code = "def f():\n    eval('evil')"
        with pytest.raises(CustomCodeValidationError, match="eval"):
            _validate_ast(code)

    def test_ast_blocks_compile_call(self):
        code = "def f():\n    compile('x', '', 'exec')"
        with pytest.raises(CustomCodeValidationError, match="compile"):
            _validate_ast(code)

    def test_ast_blocks_open_call(self):
        code = "def f():\n    open('/etc/passwd')"
        with pytest.raises(CustomCodeValidationError, match="open"):
            _validate_ast(code)

    def test_ast_blocks_getattr_call(self):
        code = "def f():\n    getattr(x, 'y')"
        with pytest.raises(CustomCodeValidationError, match="getattr"):
            _validate_ast(code)

    def test_ast_blocks_hasattr_call(self):
        """hasattr() is blocked to prevent attribute probing for sandbox escape."""
        code = "def f():\n    hasattr(x, '__subclasses__')"
        with pytest.raises(CustomCodeValidationError, match="hasattr"):
            _validate_ast(code)

    def test_ast_blocks_dunder_class_attr(self):
        code = "def f():\n    x = ''.__class__"
        with pytest.raises(CustomCodeValidationError, match="__class__"):
            _validate_ast(code)

    def test_ast_blocks_dunder_subclasses(self):
        code = "def f():\n    x.__subclasses__()"
        with pytest.raises(CustomCodeValidationError, match="__subclasses__"):
            _validate_ast(code)

    def test_ast_blocks_dunder_globals(self):
        code = "def f():\n    f.__globals__"
        with pytest.raises(CustomCodeValidationError, match="__globals__"):
            _validate_ast(code)

    def test_ast_blocks_dunder_builtins(self):
        code = "def f():\n    x.__builtins__"
        with pytest.raises(CustomCodeValidationError, match="__builtins__"):
            _validate_ast(code)

    def test_ast_blocks_dunder_code(self):
        code = "def f():\n    f.__code__"
        with pytest.raises(CustomCodeValidationError, match="__code__"):
            _validate_ast(code)

    def test_ast_blocks_dunder_mro(self):
        code = "def f():\n    x.__mro__"
        with pytest.raises(CustomCodeValidationError, match="__mro__"):
            _validate_ast(code)

    def test_ast_blocks_dunder_bases(self):
        code = "def f():\n    x.__bases__"
        with pytest.raises(CustomCodeValidationError, match="__bases__"):
            _validate_ast(code)

    def test_ast_blocks_dunder_dict(self):
        code = "def f():\n    x.__dict__"
        with pytest.raises(CustomCodeValidationError, match="__dict__"):
            _validate_ast(code)

    def test_ast_blocks_global_statement(self):
        code = "def f():\n    global x"
        with pytest.raises(CustomCodeValidationError, match="global statement"):
            _validate_ast(code)

    def test_ast_allows_type_call(self):
        """type() is a common Python construct and should not be blocked."""
        code = "def f():\n    t = type('X', (), {})"
        # Should not raise — type/super/classmethod/etc. are allowed
        _validate_ast(code)

    def test_ast_allows_super_call(self):
        """super() itself is not blocked; __init__/__new__ attribute access is not blocked either."""
        code = "def f():\n    super().__init__()"
        # super() call is allowed; __init__ is also allowed (not in FORBIDDEN_ATTR_NAMES)
        # because blocking it would break existing class-based guardrail code
        _validate_ast(code)

    def test_ast_allows_class_based_guardrail(self):
        """Class-based guardrails with __init__ and super() should be allowed."""
        code = """
class MyGuardrail:
    def __init__(self):
        super().__init__()
        self.blocklist = ["secret", "ssn"]

def apply_guardrail(inputs, request_data, input_type):
    g = MyGuardrail()
    for text in inputs["texts"]:
        for word in g.blocklist:
            if word in text:
                return block("Sensitive content")
    return allow()
"""
        # Should not raise — class-based guardrails are a valid use case
        _validate_ast(code)

    def test_ast_allows_qualname_usage(self):
        """__qualname__ was not in the original regex layer; blocking it breaks existing code."""
        code = "def f():\n    name = f.__qualname__"
        # Should not raise — __qualname__ is not in FORBIDDEN_ATTR_NAMES
        # (it was not blocked before this PR and has no sandbox-escape path)
        _validate_ast(code)

    def test_ast_reports_all_violations(self):
        """AST validation should report all violations, not just the first."""
        code = "import os\nimport sys\ndef f():\n    eval('x')\n    exec('y')"
        with pytest.raises(
            CustomCodeValidationError, match="Security violation\\(s\\)"
        ) as exc_info:
            _validate_ast(code)
        msg = str(exc_info.value)
        assert "import" in msg
        assert "eval" in msg
        assert "exec" in msg

    def test_ast_allows_safe_code(self):
        """Valid guardrail code should pass AST validation."""
        code = """
def apply_guardrail(inputs, request_data, input_type):
    for text in inputs["texts"]:
        if regex_match(text, r"\\d{3}-\\d{2}-\\d{4}"):
            return block("SSN detected")
    return allow()
"""
        # Should not raise
        _validate_ast(code)

    def test_ast_allows_closures(self):
        """Closures within guardrail code should be allowed."""
        code = """
def apply_guardrail(inputs, request_data, input_type):
    def check(text):
        return "bad" in text
    for text in inputs["texts"]:
        if check(text):
            return block("Bad content")
    return allow()
"""
        # Should not raise
        _validate_ast(code)

    def test_ast_allows_safe_builtins_in_primitives(self):
        """len, str, int, etc. from primitives should be allowed."""
        code = """
def apply_guardrail(inputs, request_data, input_type):
    count = len(inputs["texts"])
    name = str(count)
    return allow()
"""
        # Should not raise
        _validate_ast(code)

    def test_ast_raises_on_syntax_error(self):
        """Syntax errors cause _validate_ast to raise SyntaxError, not silently pass."""
        code = "def f(\n"
        # SyntaxError is re-raised so callers know the code is malformed.
        # This prevents intentionally malformed code from bypassing the AST layer.
        with pytest.raises(SyntaxError):
            _validate_ast(code)


class TestIntegrationRegexAndAST:
    """Integration tests verifying both layers work together."""

    def test_both_layers_block_import(self):
        code = "import os\ndef apply_guardrail(i, r, t):\n    return allow()"
        with pytest.raises(CustomCodeValidationError):
            validate_custom_code(code)

    def test_regex_catches_string_pattern_os_dot(self):
        """Regex layer catches os. usage even without import."""
        code = "def apply_guardrail(i, r, t):\n    os.system('ls')\n    return allow()"
        with pytest.raises(CustomCodeValidationError, match="os module"):
            validate_custom_code(code)

    def test_ast_catches_dunder_chain_attack(self):
        """AST catches the classic ().__class__.__bases__[0].__subclasses__() attack."""
        # This bypasses some regex patterns due to chaining but AST catches each attribute
        code = "def f():\n    x = ().__class__"
        with pytest.raises(CustomCodeValidationError, match="__class__"):
            validate_custom_code(code)

    def test_clean_guardrail_passes_both_layers(self):
        """A well-formed guardrail passes both validation layers."""
        code = """
def apply_guardrail(inputs, request_data, input_type):
    for text in inputs["texts"]:
        if contains(text, "secret"):
            return block("Secret content detected")
    return allow()
"""
        # Should not raise
        validate_custom_code(code)

    def test_async_guardrail_passes_both_layers(self):
        """An async guardrail with http calls passes validation."""
        code = """
async def apply_guardrail(inputs, request_data, input_type):
    for text in inputs["texts"]:
        response = await http_post(
            "https://api.example.com/check",
            body={"text": text}
        )
        if response["success"] and response["body"].get("flagged"):
            return block("Flagged by API")
    return allow()
"""
        # Should not raise
        validate_custom_code(code)
