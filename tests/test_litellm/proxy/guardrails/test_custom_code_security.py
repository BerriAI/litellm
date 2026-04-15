import pytest

from litellm.proxy.guardrails.guardrail_hooks.custom_code.custom_code_guardrail import (
    CustomCodeCompilationError,
    CustomCodeGuardrail,
)


# str.mro() + generator gi_code + code.replace(co_names=...) + __setattr__
# to swap a function's bytecode and read http_get's real builtins dict.
BYTECODE_REWRITE_PAYLOAD = (
    "def apply_guardrail(inputs, request_data, input_type):\n"
    "    obj = str.mro()[1]\n"
    "    def g(fn):\n"
    "        yield fn.placeholder\n"
    "    c = g(None).gi_code\n"
    '    gn = "_"+"_gl"+"ob"+"als"+"_"+"_"\n'
    '    cn = "_"+"_co"+"de_"+"_"\n'
    "    obj.__setattr__(g, cn, c.replace(co_names=(gn,)))\n"
    "    for v in g(http_get):\n"
    "        gd = v\n"
    "        break\n"
    '    bn = "_"+"_bu"+"ilt"+"ins"+"_"+"_"\n'
    '    imp = gd[bn]["_"+"_im"+"po"+"rt_"+"_"]\n'
    '    return {"rce": imp("os").popen("id").read()}\n'
)


def _compile(code: str) -> CustomCodeGuardrail:
    return CustomCodeGuardrail(custom_code=code, guardrail_name="t")


def test_bytecode_rewrite_rejected_at_compile():
    with pytest.raises(CustomCodeCompilationError):
        _compile(BYTECODE_REWRITE_PAYLOAD)


# Call the async http_get primitive without awaiting, then pull f_builtins off
# the returned coroutine's cr_frame. INSPECT_ATTRIBUTES covers cr_frame and
# f_builtins so this is rejected at compile time.
CR_FRAME_PAYLOAD = (
    "def apply_guardrail(inputs, request_data, input_type):\n"
    '    co = http_get("http://x")\n'
    "    b = co.cr_frame.f_builtins\n"
    "    co.close()\n"
    '    imp = b["_" + "_imp" + "ort_" + "_"]\n'
    '    return block(imp("os").popen("id").read())\n'
)


def test_cr_frame_rejected_at_compile():
    with pytest.raises(CustomCodeCompilationError):
        _compile(CR_FRAME_PAYLOAD)


# NFKC homoglyph: U+FF47 'ｇ' normalizes to 'g' at parse time, so "__ｇlobals__"
# arrives at the AST as "__globals__" and trips the underscore-prefix rule.
NFKC_PAYLOAD = (
    "def apply_guardrail(inputs, request_data, input_type):\n"
    '    b_key = "buil" + "tins"\n'
    '    i_key = "im" + "port"\n'
    "    b = allow.__\uff47lobals__[b_key]\n"
    "    import_fn = b[i_key]\n"
    '    o = import_fn("o" + "s")\n'
    '    return block(o.popen("id").read())\n'
)


def test_nfkc_homoglyph_rejected_at_compile():
    with pytest.raises(CustomCodeCompilationError):
        _compile(NFKC_PAYLOAD)


@pytest.mark.parametrize(
    "snippet",
    [
        # Literal dunder attribute access.
        "def apply_guardrail(i, r, t):\n    return str.__class__\n",
        "def apply_guardrail(i, r, t):\n"
        "    return ().__class__.__bases__[0].__subclasses__()\n",
        # gi_code — on the transformer's restricted-names list.
        "def apply_guardrail(i, r, t):\n"
        "    def g():\n        yield 1\n"
        "    return g().gi_code\n",
        # Import forms.
        "import os\ndef apply_guardrail(i, r, t):\n    return allow()\n",
        "from subprocess import call\n"
        "def apply_guardrail(i, r, t):\n    return allow()\n",
        # __import__ is rejected as an underscore-prefixed name.
        "def apply_guardrail(i, r, t):\n" '    return __import__("os")\n',
    ],
)
def test_compile_time_rejections(snippet: str):
    with pytest.raises(CustomCodeCompilationError):
        _compile(snippet)


@pytest.mark.parametrize(
    "snippet",
    [
        # getattr is not in the sandbox builtins — NameError at call time.
        "def apply_guardrail(i, r, t):\n"
        '    return getattr(str, "_"+"_class_"+"_")\n',
        # setattr is guarded_setattr + full_write_guard — setting any attribute
        # on a user-defined object raises TypeError, whether the name is a
        # dunder or not.
        "def apply_guardrail(i, r, t):\n"
        "    def f():\n        pass\n"
        '    name = "_" + "_bad_" + "_"\n'
        "    setattr(f, name, None)\n"
        "    return allow()\n",
    ],
)
def test_runtime_rejections(snippet: str):
    guardrail = _compile(snippet)
    fn = guardrail._compiled_function
    assert fn is not None
    with pytest.raises((NameError, TypeError, AttributeError, SyntaxError)):
        fn({"texts": []}, {}, "request")


def test_documented_ssn_example_compiles_and_runs():
    code = (
        "def apply_guardrail(inputs, request_data, input_type):\n"
        '    for text in inputs["texts"]:\n'
        '        if regex_match(text, r"\\d{3}-\\d{2}-\\d{4}"):\n'
        '            return block("SSN detected")\n'
        "    return allow()\n"
    )
    guardrail = _compile(code)
    fn = guardrail._compiled_function
    assert fn is not None
    assert fn({"texts": ["hello"]}, {}, "request") == {"action": "allow"}
    blocked = fn({"texts": ["my ssn 123-45-6789"]}, {}, "request")
    assert blocked["action"] == "block"
    assert blocked["reason"] == "SSN detected"


@pytest.mark.asyncio
async def test_async_guardrail_compiles_and_runs():
    code = (
        "async def apply_guardrail(inputs, request_data, input_type):\n"
        "    return allow()\n"
    )
    guardrail = _compile(code)
    from litellm.types.utils import GenericGuardrailAPIInputs

    result = await guardrail.apply_guardrail(
        inputs=GenericGuardrailAPIInputs(texts=["test"]),
        request_data={},
        input_type="request",
    )
    assert result["texts"][0] == "test"


def test_typical_sync_guardrail_still_works():
    code = (
        "def apply_guardrail(inputs, request_data, input_type):\n"
        "    return allow()\n"
    )
    guardrail = _compile(code)
    assert guardrail._compiled_function is not None


def test_augmented_assignment_works():
    # The transformer rewrites `n += 1` into `n = _inplacevar_("+=", n, 1)`,
    # so the sandbox must bind `_inplacevar_`.
    code = (
        "def apply_guardrail(inputs, request_data, input_type):\n"
        "    count = 0\n"
        '    for _ in inputs["texts"]:\n'
        "        count += 1\n"
        '    return {"action": "allow", "n": count}\n'
    )
    guardrail = _compile(code)
    fn = guardrail._compiled_function
    assert fn is not None
    assert fn({"texts": ["a", "b", "c"]}, {}, "request") == {
        "action": "allow",
        "n": 3,
    }


def test_missing_apply_guardrail_raises():
    with pytest.raises(CustomCodeCompilationError, match="apply_guardrail"):
        _compile("x = 1\n")
