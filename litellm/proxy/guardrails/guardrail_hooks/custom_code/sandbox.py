"""
RestrictedPython-based sandbox for custom code guardrails.

User-supplied guardrail source is compiled with ``compile_restricted`` and
executed with curated globals. All attribute access, subscripting, iteration,
and assignment is mediated by RestrictedPython guards, which block dunder
access (``__globals__``, ``__code__``, ``__class__``, ``__setattr__``, etc.)
and reject dangerous AST constructs (``import``, ``exec``, ``eval``,
``compile``, class definitions, etc.) at compile time.

The default ``RestrictingNodeTransformer`` denies ``async def``/``await``,
which breaks the documented async guardrail pattern (``await http_get(...)``).
We subclass it to permit those specific nodes, while keeping every other
restriction intact.
"""

import operator
from typing import Any, Dict

from RestrictedPython import (
    RestrictingNodeTransformer,
    compile_restricted,
    limited_builtins,
    safe_builtins,
    utility_builtins,
)
from RestrictedPython.Eval import default_guarded_getitem, default_guarded_getiter
from RestrictedPython.Guards import (
    full_write_guard,
    guarded_iter_unpack_sequence,
    safer_getattr,
)

from .primitives import get_custom_code_primitives


class AsyncAwareTransformer(RestrictingNodeTransformer):
    """Extend the default transformer to allow ``async def`` and ``await``.

    The base class rejects every async AST node outright. ``AsyncFunctionDef``
    has the same ``_fields`` as ``FunctionDef`` and the same security
    semantics, so we delegate to ``visit_FunctionDef`` — name check, argument
    check, print-scope wrapping, and any future additions to that method are
    inherited automatically. ``AsyncFor``/``AsyncWith``/``Await`` delegate to
    ``node_contents_visit`` so their children still get transformed.
    """

    def visit_AsyncFunctionDef(self, node: Any) -> Any:
        return self.visit_FunctionDef(node)

    def visit_AsyncFor(self, node: Any) -> Any:
        return self.node_contents_visit(node)

    def visit_AsyncWith(self, node: Any) -> Any:
        return self.node_contents_visit(node)

    def visit_Await(self, node: Any) -> Any:
        return self.node_contents_visit(node)


_INPLACE_OPS: Dict[str, Any] = {
    "+=": operator.iadd,
    "-=": operator.isub,
    "*=": operator.imul,
    "/=": operator.itruediv,
    "//=": operator.ifloordiv,
    "%=": operator.imod,
    "**=": operator.ipow,
    "@=": operator.imatmul,
    "&=": operator.iand,
    "|=": operator.ior,
    "^=": operator.ixor,
    "<<=": operator.ilshift,
    ">>=": operator.irshift,
}


def _inplacevar_(op: str, x: Any, y: Any) -> Any:
    # RestrictedPython rewrites ``x += 1`` on a simple name into
    # ``x = _inplacevar_("+=", x, 1)``. The package deliberately ships no
    # default, so we dispatch through ``operator``'s in-place helpers, which
    # honour Python's normal ``__iadd__``/``__add__`` fallback.
    fn = _INPLACE_OPS.get(op)
    if fn is None:
        raise SyntaxError(f"augmented assignment {op!r} is not supported")
    return fn(x, y)


def _build_sandbox_builtins() -> Dict[str, Any]:
    # ``limited_builtins`` overrides ``list``/``tuple``/``range`` from
    # ``safe_builtins`` with bounds-checking variants (e.g. ``limited_range``
    # rejects ``range(10**18)``). ``utility_builtins`` adds ``set``,
    # ``frozenset``, ``math``, ``random``, and a filtered ``string`` delegator.
    return {
        **safe_builtins,
        **limited_builtins,
        **utility_builtins,
    }


def build_sandbox_globals() -> Dict[str, Any]:
    """Assemble the globals dict for executing guardrail code.

    Includes the LiteLLM-provided primitives (``regex_match``, ``http_get``,
    ``allow``/``block``/``modify``, etc.) plus the RestrictedPython guards
    that the compiled bytecode expects to find by name.
    """
    sandbox: Dict[str, Any] = get_custom_code_primitives().copy()
    sandbox["__builtins__"] = _build_sandbox_builtins()
    sandbox["_getattr_"] = safer_getattr
    sandbox["_getitem_"] = default_guarded_getitem
    sandbox["_getiter_"] = default_guarded_getiter
    sandbox["_iter_unpack_sequence_"] = guarded_iter_unpack_sequence
    sandbox["_write_"] = full_write_guard
    sandbox["_inplacevar_"] = _inplacevar_
    return sandbox


def compile_sandboxed(source: str, filename: str = "<guardrail>") -> Any:
    """Compile guardrail source with RestrictedPython's AST transformer.

    Raises ``SyntaxError`` on either a Python syntax error or a restricted
    construct (import, exec, dunder name, etc.).
    """
    return compile_restricted(
        source=source,
        filename=filename,
        mode="exec",
        policy=AsyncAwareTransformer,
    )
