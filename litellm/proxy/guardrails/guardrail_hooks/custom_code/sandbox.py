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

import ast
import operator
from collections.abc import AsyncIterable, AsyncIterator, Callable, Mapping
from types import CodeType
from typing import Final

from RestrictedPython import (
    RestrictingNodeTransformer,
    compile_restricted,
    limited_builtins,
    safe_builtins,
    utility_builtins,
)
from RestrictedPython.Eval import default_guarded_getitem
from RestrictedPython.Guards import (
    full_write_guard,
    guarded_iter_unpack_sequence,
    guarded_unpack_sequence,
    safer_getattr,
)
from RestrictedPython.transformer import copy_locations

from .bounded_execution import budget_ok, budgeted_iter
from .primitives import get_custom_code_primitives


class AsyncAwareTransformer(RestrictingNodeTransformer):
    """Extend the default transformer to allow ``async def`` and ``await``.

    The base class rejects every async AST node outright. ``AsyncFunctionDef``
    has the same ``_fields`` as ``FunctionDef`` and the same security
    semantics, so we delegate to ``visit_FunctionDef`` — name check, argument
    check, print-scope wrapping, and any future additions to that method are
    inherited automatically. ``AsyncWith`` gets the same treatment for the same
    reason: ``node_contents_visit`` only recurses into children, so routing it
    there left ``async with x as (a, b)`` without the unpack guard that
    ``with x as (a, b)`` gets. ``AsyncFor`` likewise delegates to ``visit_For``,
    which is what wraps the loop iterator in ``_getiter_``; the wrapper names it
    inserts are then swapped for async-capable ones, since ``budgeted_iter`` and
    ``_iter_unpack_sequence_`` are plain generators that ``async for`` cannot
    consume. ``Await`` has no synchronous counterpart and only wraps an
    expression, so it stays on ``node_contents_visit``.

    ``visit_While`` rewrites ``while test:`` to ``while _budget_ok_() and test:``
    so a loop that never yields is still stopped at the execution deadline;
    ``for`` loops and comprehensions get the same check through ``_getiter_``.
    """

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        return self.visit_FunctionDef(node)

    def visit_While(self, node: ast.While) -> ast.AST:
        visited: Final = self.node_contents_visit(node)
        budget_check: Final = ast.Call(
            func=ast.Name(id="_budget_ok_", ctx=ast.Load()),
            args=[],
            keywords=[],
        )
        test: Final = ast.BoolOp(
            op=ast.And(),
            values=[budget_check, visited.test],
        )
        copy_locations(test, visited.test)
        bounded: Final = ast.While(test=test, body=visited.body, orelse=visited.orelse)
        copy_locations(bounded, visited)
        return bounded

    def visit_AsyncFor(self, node: ast.AsyncFor) -> ast.AST:
        transformed: Final = self.visit_For(node)
        _use_async_iter_guards(transformed)
        return transformed

    def visit_AsyncWith(self, node: ast.AsyncWith) -> ast.AST:
        return self.visit_With(node)

    def visit_Await(self, node: ast.Await) -> ast.AST:
        return self.node_contents_visit(node)


_GETITER_NAME: Final = "_getiter_"
_ASYNC_GETITER_NAME: Final = "_agetiter_"
_ITER_UNPACK_NAME: Final = "_iter_unpack_sequence_"
_ASYNC_ITER_UNPACK_NAME: Final = "_aiter_unpack_sequence_"


def _use_async_iter_guards(node: ast.AST) -> None:
    """Swap the iterator wrapper ``visit_For`` inserted for its async twin.

    ``visit_For`` rewrites ``for x in y`` into ``for x in _getiter_(y)`` and
    ``for a, b in y`` into
    ``for (a, b) in _iter_unpack_sequence_(y, spec, _getiter_)``. Both wrappers
    are synchronous generators, which ``async for`` cannot consume, so the
    async form needs the async-generator equivalents under their own names.
    Only the wrapper name changes: the guarding each one performs is the same.
    """
    iter_node: Final = getattr(node, "iter", None)
    if not (isinstance(iter_node, ast.Call) and isinstance(iter_node.func, ast.Name)):
        return
    if iter_node.func.id == _ITER_UNPACK_NAME:
        iter_node.func.id = _ASYNC_ITER_UNPACK_NAME
    elif iter_node.func.id == _GETITER_NAME:
        iter_node.func.id = _ASYNC_GETITER_NAME


async def _agetiter_(iterable: AsyncIterable[object]) -> AsyncIterator[object]:
    """``budgeted_iter`` for ``async for``: the same per-item budget check.

    Bound to the name ``_use_async_iter_guards`` substitutes, so an ``async for``
    that never yields is still stopped at the execution deadline, exactly as a
    ``for`` loop is.
    """
    async for item in iterable:
        budget_ok()
        yield item


async def _aiter_unpack_sequence_(
    it: AsyncIterable[object],
    spec: object,
    getiter: Callable[[object], object],
) -> AsyncIterator[object]:
    """``guarded_iter_unpack_sequence`` for ``async for`` targets.

    Same contract as the RestrictedPython helper -- budget the iteration, then
    guard each element's sequence unpacking -- over an async iterator. Unpacking
    an element is synchronous, so the ``_getiter_`` handed in is the synchronous
    one and is passed through untouched.
    """
    async for ob in it:
        budget_ok()
        yield guarded_unpack_sequence(ob, spec, getiter)


_INPLACE_OPS: Final[Mapping[str, Callable[[object, object], object]]] = {
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


def _inplacevar_(op: str, x: object, y: object) -> object:
    # RestrictedPython rewrites ``x += 1`` on a simple name into
    # ``x = _inplacevar_("+=", x, 1)``. The package deliberately ships no
    # default, so we dispatch through ``operator``'s in-place helpers, which
    # honour Python's normal ``__iadd__``/``__add__`` fallback.
    fn: Final = _INPLACE_OPS.get(op)
    if fn is None:
        raise SyntaxError(f"augmented assignment {op!r} is not supported")
    return fn(x, y)


def _build_sandbox_builtins() -> dict[str, object]:
    # ``limited_builtins`` overrides ``list``/``tuple``/``range`` from
    # ``safe_builtins`` with bounds-checking variants (e.g. ``limited_range``
    # rejects ``range(10**18)``). ``utility_builtins`` adds ``set``,
    # ``frozenset``, ``math``, ``random``, and a filtered ``string`` delegator.
    return {
        **safe_builtins,
        **limited_builtins,
        **utility_builtins,
    }


def build_sandbox_globals() -> dict[str, object]:
    """Assemble the globals dict for executing guardrail code.

    Includes the LiteLLM-provided primitives (``regex_match``, ``http_get``,
    ``allow``/``block``/``modify``, etc.) plus the RestrictedPython guards
    that the compiled bytecode expects to find by name.
    """
    return {
        **get_custom_code_primitives(),
        "__builtins__": _build_sandbox_builtins(),
        "_getattr_": safer_getattr,
        "_getitem_": default_guarded_getitem,
        "_getiter_": budgeted_iter,
        "_iter_unpack_sequence_": guarded_iter_unpack_sequence,
        # RestrictedPython emits _unpack_sequence_ for every tuple-unpacking
        # target that is not a for-loop target -- ``a, b = pair``,
        # ``a, *rest = seq``, ``with x as (a, b)`` -- and, like _inplacevar_,
        # ships no default. Without it those statements compile and then raise
        # NameError the first time the guardrail runs.
        "_unpack_sequence_": guarded_unpack_sequence,
        _ASYNC_GETITER_NAME: _agetiter_,
        _ASYNC_ITER_UNPACK_NAME: _aiter_unpack_sequence_,
        "_write_": full_write_guard,
        "_inplacevar_": _inplacevar_,
        "_budget_ok_": budget_ok,
    }


def compile_sandboxed(source: str, filename: str = "<guardrail>") -> CodeType:
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
