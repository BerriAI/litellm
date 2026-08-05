"""Traversal contract for the shared exception-tree walk: the root is yielded first, explicit
links win (the ``raise ... from`` cause subtree, then ExceptionGroup members in raise order,
then the incidental ``__context__`` chain last), and adversarial shapes terminate."""

from litellm.proxy._experimental.mcp_server.faults import iter_exception_tree


def test_yields_the_root_itself_first():
    exc = ValueError("root")
    assert list(iter_exception_tree(exc)) == [exc]


def test_cause_subtree_is_exhausted_before_context():
    deep = KeyError("deep")
    cause = RuntimeError("cause")
    cause.__cause__ = deep
    context = OSError("context")
    root = ValueError("root")
    root.__cause__ = cause
    root.__context__ = context
    assert list(iter_exception_tree(root)) == [root, cause, deep, context]


def test_group_members_yield_in_raise_order_between_cause_and_context():
    first = KeyError("first")
    second = IndexError("second")
    group = BaseExceptionGroup("group", [first, second])
    cause = RuntimeError("cause")
    context = OSError("context")
    group.__cause__ = cause
    group.__context__ = context
    assert list(iter_exception_tree(group)) == [group, cause, first, second, context]


def test_terminates_on_a_cause_cycle():
    a = ValueError("a")
    b = RuntimeError("b")
    a.__cause__ = b
    b.__cause__ = a
    assert list(iter_exception_tree(a)) == [a, b]


def test_node_reachable_as_both_cause_and_context_yields_once():
    inner = KeyError("inner")
    root = ValueError("root")
    root.__cause__ = inner
    root.__context__ = inner
    assert list(iter_exception_tree(root)) == [root, inner]
