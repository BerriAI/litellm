from typing import Any  # noqa:F401
from typing import Dict  # noqa:F401
from typing import List  # noqa:F401
from typing import Optional  # noqa:F401
from typing import Tuple  # noqa:F401
from typing import Union  # noqa:F401


class ArgumentError(Exception):
    """
    This is raised when an argument lookup, either by position or by keyword, is
    not found.
    """


def get_argument_value(
    args: Union[Tuple[Any], List[Any]],
    kwargs: Dict[str, Any],
    pos: int,
    kw: str,
    optional: bool = False,
) -> Optional[Any]:
    """
    This function parses the value of a target function argument that may have been
    passed in as a positional argument or a keyword argument. Because monkey-patched
    functions do not define the same signature as their target function, the value of
    arguments must be inferred from the packed args and kwargs.
    Keyword arguments are prioritized, followed by the positional argument. If the
    argument cannot be resolved, an ``ArgumentError`` exception is raised, which could
    be used, e.g., to handle a default value by the caller.
    :param args: Positional arguments
    :param kwargs: Keyword arguments
    :param pos: The positional index of the argument if passed in as a positional arg
    :param kw: The name of the keyword if passed in as a keyword argument
    :return: The value of the target argument
    """
    try:
        return kwargs[kw]
    except KeyError:
        try:
            return args[pos]
        except IndexError:
            if optional:
                return None
            raise ArgumentError("%s (at position %d)" % (kw, pos))


def set_argument_value(
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    pos: int,
    kw: str,
    value: Any,
    override_unset: bool = False,
) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
    """
    Returns a new args, kwargs with the given value updated
    :param args: Positional arguments
    :param kwargs: Keyword arguments
    :param pos: The positional index of the argument
    :param kw: The name of the keyword
    :param value: The new value of the target argument
    :return: Updated args and kwargs
    """
    if len(args) > pos:
        args = args[:pos] + (value,) + args[pos + 1 :]
    elif kw in kwargs or override_unset:
        kwargs[kw] = value
    else:
        raise ArgumentError("%s (at position %d) is invalid" % (kw, pos))

    return args, kwargs


def _get_metas_to_propagate(context):
    # type: (Any) -> List[Tuple[str, str]]
    metas_to_propagate = []
    # copying context._meta.items() to avoid RuntimeError: dictionary changed size during iteration
    for k, v in list(context._meta.items()):
        if isinstance(k, str) and k.startswith("_dd.p."):
            metas_to_propagate.append((k, v))
    return metas_to_propagate


def get_blocked() -> Optional[Dict[str, Any]]:
    # local import to avoid circular dependency
    from ddtrace.internal import core

    res = core.dispatch_with_results("asm.get_blocked")
    if res and res.block_config:
        return res.block_config.value
    return None


def set_blocked(block_settings: Optional[Dict[str, Any]] = None) -> None:
    # local imports to avoid circular dependency
    from ddtrace.internal import core
    from ddtrace.internal.constants import STATUS_403_TYPE_AUTO

    core.dispatch("asm.set_blocked", (block_settings or STATUS_403_TYPE_AUTO,))
