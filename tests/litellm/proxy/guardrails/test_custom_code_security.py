import pytest
from litellm.proxy.guardrails.guardrail_hooks.custom_code.code_validator import (
    validate_custom_code,
    CustomCodeValidationError,
)
from litellm.proxy.guardrails.guardrail_hooks.custom_code.custom_code_guardrail import (
    CustomCodeGuardrail,
)

# Phase 4.1: Test forbidden pattern validation


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


# Phase 4.2: Test __builtins__ restriction in execution


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


# Phase 4.3: Test frame traversal patterns are blocked


def test_validate_custom_code_cr_frame():
    code = "async def apply_guardrail(i, r, t):\n    x = some_coro.cr_frame\n    return allow()"
    with pytest.raises(CustomCodeValidationError, match="coroutine frame access"):
        validate_custom_code(code)


def test_validate_custom_code_gi_frame():
    code = "def apply_guardrail(i, r, t):\n    x = gen.gi_frame\n    return allow()"
    with pytest.raises(CustomCodeValidationError, match="generator frame access"):
        validate_custom_code(code)


def test_validate_custom_code_f_back():
    code = "def apply_guardrail(i, r, t):\n    frame.f_back.f_globals\n    return allow()"
    with pytest.raises(CustomCodeValidationError, match="frame traversal via f_back"):
        validate_custom_code(code)


def test_validate_custom_code_f_globals():
    code = "def apply_guardrail(i, r, t):\n    x = frame.f_globals\n    return allow()"
    with pytest.raises(CustomCodeValidationError, match="frame globals access"):
        validate_custom_code(code)


def test_validate_custom_code_currentframe():
    code = "def apply_guardrail(i, r, t):\n    f = currentframe()\n    return allow()"
    with pytest.raises(CustomCodeValidationError, match="currentframe"):
        validate_custom_code(code)


def test_validate_custom_code_getframe():
    code = "def apply_guardrail(i, r, t):\n    f = _getframe(0)\n    return allow()"
    with pytest.raises(CustomCodeValidationError, match="_getframe"):
        validate_custom_code(code)


def test_validate_custom_code_co_consts():
    code = "def apply_guardrail(i, r, t):\n    x = func.co_consts\n    return allow()"
    with pytest.raises(CustomCodeValidationError, match="code object constants"):
        validate_custom_code(code)
