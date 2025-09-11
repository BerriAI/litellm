"""Decorators for Shapely functions."""

import os
import warnings
from collections.abc import Callable, Iterable
from functools import lru_cache, wraps
from inspect import unwrap

import numpy as np

from shapely import lib
from shapely.errors import UnsupportedGEOSVersionError


class requires_geos:
    """Decorator to require a minimum GEOS version."""

    def __init__(self, version):
        """Create a decorator that requires a minimum GEOS version."""
        if version.count(".") != 2:
            raise ValueError("Version must be <major>.<minor>.<patch> format")
        self.version = tuple(int(x) for x in version.split("."))

    def __call__(self, func):
        """Return the wrapped function."""
        is_compatible = lib.geos_version >= self.version
        is_doc_build = os.environ.get("SPHINX_DOC_BUILD") == "1"  # set in docs/conf.py
        if is_compatible and not is_doc_build:
            return func  # return directly, do not change the docstring

        msg = "'{}' requires at least GEOS {}.{}.{}.".format(
            func.__name__, *self.version
        )
        if is_compatible:

            @wraps(func)
            def wrapped(*args, **kwargs):
                return func(*args, **kwargs)

        else:

            @wraps(func)
            def wrapped(*args, **kwargs):
                raise UnsupportedGEOSVersionError(msg)

        doc = wrapped.__doc__
        if doc:
            # Insert the message at the first double newline
            position = doc.find("\n\n") + 2
            # Figure out the indentation level
            indent = 0
            while True:
                if doc[position + indent] == " ":
                    indent += 1
                else:
                    break
            wrapped.__doc__ = doc.replace(
                "\n\n", "\n\n{}.. note:: {}\n\n".format(" " * indent, msg), 1
            )

        return wrapped


def multithreading_enabled(func):
    """Enable multithreading.

    To do this, the writable flags of object type ndarrays are set to False.

    NB: multithreading also requires the GIL to be released, which is done in
    the C extension (ufuncs.c).
    """

    @wraps(func)
    def wrapped(*args, **kwargs):
        array_args = [
            arg for arg in args if isinstance(arg, np.ndarray) and arg.dtype == object
        ] + [
            arg
            for name, arg in kwargs.items()
            if name not in {"where", "out"}
            and isinstance(arg, np.ndarray)
            and arg.dtype == object
        ]
        old_flags = [arr.flags.writeable for arr in array_args]
        try:
            for arr in array_args:
                arr.flags.writeable = False
            return func(*args, **kwargs)
        finally:
            for arr, old_flag in zip(array_args, old_flags):
                arr.flags.writeable = old_flag

    return wrapped


def deprecate_positional(
    should_be_kwargs: Iterable[str],
    category: type[Warning] = DeprecationWarning,
):
    """Show warning if positional arguments are used that should be keyword.

    Parameters
    ----------
    should_be_kwargs : Iterable[str]
        Names of parameters that should be passed as keyword arguments.
    category : type[Warning], optional (default: DeprecationWarning)
        Warning category to use for deprecation warnings.

    Returns
    -------
    callable
        Decorator function that adds positional argument deprecation warnings.

    Examples
    --------
    >>> from shapely.decorators import deprecate_positional
    >>> @deprecate_positional(['b', 'c'])
    ... def example(a, b, c=None):
    ...     return a, b, c
    ...
    >>> example(1, 2)  # doctest: +SKIP
    DeprecationWarning: positional argument `b` for `example` is deprecated. ...
    (1, 2, None)
    >>> example(1, b=2)  # No warnings
    (1, 2, None)
    """

    def decorator(func: Callable):
        code = unwrap(func).__code__

        # positional parameters are the first co_argcount names
        pos_names = code.co_varnames[: code.co_argcount]
        # build a name -> index map
        name_to_idx = {name: idx for idx, name in enumerate(pos_names)}
        # pick out only those names we care about
        deprecate_positions = [
            (name_to_idx[name], name)
            for name in should_be_kwargs
            if name in name_to_idx
        ]

        # early exit if there are no deprecated positional args
        if not deprecate_positions:
            return func

        # earliest position where a warning could occur
        warn_from = min(deprecate_positions)[0]

        @lru_cache(10)
        def make_msg(n_args: int):
            used = [name for idx, name in deprecate_positions if idx < n_args]

            if len(used) == 1:
                args_txt = f"`{used[0]}`"
                plr = ""
                isare = "is"
            else:
                plr = "s"
                isare = "are"
                if len(used) == 2:
                    args_txt = " and ".join(f"`{u}`" for u in used)
                else:
                    args_txt = ", ".join(f"`{u}`" for u in used[:-1])
                    args_txt += f", and `{used[-1]}`"

            return (
                f"positional argument{plr} {args_txt} for `{func.__name__}` "
                f"{isare} deprecated. Please use keyword argument{plr} instead."
            )

        @wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)

            n = len(args)
            if n > warn_from:
                warnings.warn(make_msg(n), category=category, stacklevel=2)

            return result

        return wrapper

    return decorator
