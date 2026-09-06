import pytest
from fastapi import HTTPException

from litellm.exceptions import ModifyResponseException
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


@pytest.mark.asyncio
async def test_custom_code_pre_call_block_uses_passthrough():
    code = (
        "def apply_guardrail(inputs, request_data, input_type):\n"
        '    return block("blocked by test")\n'
    )
    guardrail = _compile(code)

    with pytest.raises(ModifyResponseException) as exc_info:
        await guardrail.apply_guardrail(
            inputs={"texts": ["test"]},
            request_data={"model": "test-model"},
            input_type="request",
        )

    assert exc_info.value.message == "blocked by test"
    assert exc_info.value.model == "test-model"
    assert exc_info.value.guardrail_name == "t"


@pytest.mark.asyncio
async def test_custom_code_post_call_block_raises_http_400():
    code = (
        "def apply_guardrail(inputs, request_data, input_type):\n"
        '    return block("blocked by test")\n'
    )
    guardrail = _compile(code)

    with pytest.raises(HTTPException) as exc_info:
        await guardrail.apply_guardrail(
            inputs={"texts": ["test"]},
            request_data={"model": "test-model"},
            input_type="response",
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == {
        "error": "blocked by test",
        "guardrail": "t",
        "detection_info": {},
    }


FLAG_CODE = (
    "def apply_guardrail(inputs, request_data, input_type):\n"
    '    return flag("audit hit", metadata={"category": "topic"})\n'
)


@pytest.mark.asyncio
@pytest.mark.parametrize("input_type", ["request", "response"])
async def test_custom_code_flag_passes_content_through_and_records_flagged_entry(input_type):
    """LIT-6894: flag() must not raise, must return the content unchanged and must log
    exactly one guardrail_flagged entry (the decorator must not add a second "success")."""
    guardrail = CustomCodeGuardrail(custom_code=FLAG_CODE, guardrail_name="t", event_hook=["pre_call", "post_call"])
    request_data = {"model": "test-model", "litellm_metadata": {}}

    result = await guardrail.apply_guardrail(
        inputs={"texts": ["hello"]},
        request_data=request_data,
        input_type=input_type,
    )

    assert result == {"texts": ["hello"]}
    entries = request_data["litellm_metadata"]["standard_logging_guardrail_information"]
    assert len(entries) == 1
    entry = entries[0]
    assert entry["guardrail_status"] == "guardrail_flagged"
    assert entry["guardrail_name"] == "t"
    assert entry["guardrail_mode"] == ["pre_call", "post_call"]
    assert entry["guardrail_response"] == {
        "action": "flag",
        "reason": "audit hit",
        "input_type": input_type,
        "metadata": {"category": "topic"},
    }
    assert entry["duration"] is not None and entry["duration"] >= 0


@pytest.mark.asyncio
async def test_custom_code_flag_default_reason_and_empty_metadata():
    code = "def apply_guardrail(inputs, request_data, input_type):\n    return flag('just a note')\n"
    guardrail = _compile(code)
    request_data = {"model": "m", "litellm_metadata": {}}

    await guardrail.apply_guardrail(inputs={"texts": ["x"]}, request_data=request_data, input_type="request")

    entry = request_data["litellm_metadata"]["standard_logging_guardrail_information"][0]
    assert entry["guardrail_response"] == {
        "action": "flag",
        "reason": "just a note",
        "input_type": "request",
        "metadata": {},
    }


@pytest.mark.asyncio
async def test_custom_code_allow_still_records_success_not_flagged():
    code = "def apply_guardrail(inputs, request_data, input_type):\n    return allow()\n"
    guardrail = _compile(code)
    request_data = {"model": "m", "litellm_metadata": {}}

    await guardrail.apply_guardrail(inputs={"texts": ["x"]}, request_data=request_data, input_type="request")

    entries = request_data["litellm_metadata"]["standard_logging_guardrail_information"]
    assert [e["guardrail_status"] for e in entries] == ["success"]


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


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        # Every one of these compiles fine and then raises
        # "NameError: name '_unpack_sequence_' is not defined" at call time when
        # the guard is missing from the sandbox globals.
        ("    a, b = inputs['pair']\n    return a + b\n", 3),
        ("    a, (b, c) = inputs['nested']\n    return a + b + c\n", 6),
        ("    a, *rest = inputs['seq']\n    return rest\n", [2, 3]),
        ("    with inputs['ctx'] as (a, b):\n        return a + b\n", 15),
    ],
)
def test_tuple_unpacking_runs(body: str, expected):
    """Ordinary tuple unpacking must work inside a guardrail, not NameError."""

    class _Ctx:
        def __enter__(self):
            return (7, 8)

        def __exit__(self, *args):
            return False

    guardrail = _compile("def apply_guardrail(inputs, request_data, input_type):\n" + body)
    fn = guardrail._compiled_function
    assert fn is not None
    inputs = {"pair": (1, 2), "nested": (1, (2, 3)), "seq": [1, 2, 3], "ctx": _Ctx()}
    assert fn(inputs, {}, "request") == expected


def _guard_names(source: str) -> set[str]:
    """Names of the RestrictedPython guards the compiled bytecode calls."""
    import types

    from litellm.proxy.guardrails.guardrail_hooks.custom_code.sandbox import (
        compile_sandboxed,
    )

    found: set[str] = set()
    stack = [compile_sandboxed(source)]
    while stack:
        code = stack.pop()
        found |= {n for n in code.co_names + code.co_varnames if n.startswith("_") and n.endswith("_")}
        stack += [c for c in code.co_consts if isinstance(c, types.CodeType)]
    return found


def test_async_with_gets_the_same_guards_as_with():
    """`async with` is the async spelling of `with`; it must not enforce less."""
    sync_guards = _guard_names("def f(x):\n    with x as (a, b):\n        pass\n")
    async_guards = _guard_names("async def f(x):\n    async with x as (a, b):\n        pass\n")

    assert "_unpack_sequence_" in sync_guards, "precondition: `with` unpacking is guarded"
    assert sync_guards <= async_guards


@pytest.mark.asyncio
async def test_async_with_still_executes():
    """Guarding `async with` must not break it."""
    from litellm.proxy.guardrails.guardrail_hooks.custom_code.sandbox import (
        build_sandbox_globals,
        compile_sandboxed,
    )

    class _ACtx:
        def __init__(self, value):
            self.value = value

        async def __aenter__(self):
            return self.value

        async def __aexit__(self, *args):
            return False

    sandbox_globals = build_sandbox_globals()
    exec(  # noqa: S102
        compile_sandboxed(
            "async def f(ctx):\n    async with ctx as (a, b):\n        return a + b\n"
        ),
        sandbox_globals,
    )
    assert await sandbox_globals["f"](_ACtx((5, 6))) == 11


def test_async_for_gets_the_same_guards_as_for():
    """`async for` is the async spelling of `for`; it must not enforce less."""
    sync_guards = _guard_names("def f(x):\n    for a in x:\n        pass\n")
    async_guards = _guard_names("async def f(x):\n    async for a in x:\n        pass\n")

    assert "_getiter_" in sync_guards, "precondition: `for` iteration is guarded"
    assert sync_guards <= async_guards


def test_async_for_unpacking_uses_the_async_unpack_guard():
    """Tuple targets are guarded too, via the async-iterable variant of the helper."""
    guards = _guard_names("async def f(x):\n    async for a, b in x:\n        pass\n")

    assert "_aiter_unpack_sequence_" in guards
    # The sync generator would raise "requires an object with __aiter__".
    assert "_iter_unpack_sequence_" not in guards


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("body", "items", "expected"),
    [
        ("    async for i in src:\n        out.append(i)\n", [1, 2, 3], [1, 2, 3]),
        ("    async for a, b in src:\n        out.append(a + b)\n", [(1, 2), (3, 4)], [3, 7]),
    ],
)
async def test_async_for_still_executes(body: str, items: list, expected: list):
    """Guarding `async for` must not break it, with or without a tuple target."""
    from litellm.proxy.guardrails.guardrail_hooks.custom_code.sandbox import (
        build_sandbox_globals,
        compile_sandboxed,
    )

    class _AIter:
        def __init__(self, values):
            self.values = list(values)

        def __aiter__(self):
            return self

        async def __anext__(self):
            if not self.values:
                raise StopAsyncIteration
            return self.values.pop(0)

    sandbox_globals = build_sandbox_globals()
    exec(  # noqa: S102
        compile_sandboxed("async def f(src):\n    out = []\n" + body + "    return out\n"),
        sandbox_globals,
    )
    assert await sandbox_globals["f"](_AIter(items)) == expected
