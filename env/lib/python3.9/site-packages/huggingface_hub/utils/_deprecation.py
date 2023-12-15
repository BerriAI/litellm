import warnings
from functools import wraps
from inspect import Parameter, signature
from typing import Generator, Iterable, Optional


def _deprecate_positional_args(*, version: str):
    """Decorator for methods that issues warnings for positional arguments.
    Using the keyword-only argument syntax in pep 3102, arguments after the
    * will issue a warning when passed as a positional argument.

    Args:
        version (`str`):
            The version when positional arguments will result in error.
    """

    def _inner_deprecate_positional_args(f):
        sig = signature(f)
        kwonly_args = []
        all_args = []
        for name, param in sig.parameters.items():
            if param.kind == Parameter.POSITIONAL_OR_KEYWORD:
                all_args.append(name)
            elif param.kind == Parameter.KEYWORD_ONLY:
                kwonly_args.append(name)

        @wraps(f)
        def inner_f(*args, **kwargs):
            extra_args = len(args) - len(all_args)
            if extra_args <= 0:
                return f(*args, **kwargs)
            # extra_args > 0
            args_msg = [
                f"{name}='{arg}'" if isinstance(arg, str) else f"{name}={arg}"
                for name, arg in zip(kwonly_args[:extra_args], args[-extra_args:])
            ]
            args_msg = ", ".join(args_msg)
            warnings.warn(
                (
                    f"Deprecated positional argument(s) used in '{f.__name__}': pass"
                    f" {args_msg} as keyword args. From version {version} passing these"
                    " as positional arguments will result in an error,"
                ),
                FutureWarning,
            )
            kwargs.update(zip(sig.parameters, args))
            return f(**kwargs)

        return inner_f

    return _inner_deprecate_positional_args


def _deprecate_arguments(
    *,
    version: str,
    deprecated_args: Iterable[str],
    custom_message: Optional[str] = None,
):
    """Decorator to issue warnings when using deprecated arguments.

    TODO: could be useful to be able to set a custom error message.

    Args:
        version (`str`):
            The version when deprecated arguments will result in error.
        deprecated_args (`List[str]`):
            List of the arguments to be deprecated.
        custom_message (`str`, *optional*):
            Warning message that is raised. If not passed, a default warning message
            will be created.
    """

    def _inner_deprecate_positional_args(f):
        sig = signature(f)

        @wraps(f)
        def inner_f(*args, **kwargs):
            # Check for used deprecated arguments
            used_deprecated_args = []
            for _, parameter in zip(args, sig.parameters.values()):
                if parameter.name in deprecated_args:
                    used_deprecated_args.append(parameter.name)
            for kwarg_name, kwarg_value in kwargs.items():
                if (
                    # If argument is deprecated but still used
                    kwarg_name in deprecated_args
                    # And then the value is not the default value
                    and kwarg_value != sig.parameters[kwarg_name].default
                ):
                    used_deprecated_args.append(kwarg_name)

            # Warn and proceed
            if len(used_deprecated_args) > 0:
                message = (
                    f"Deprecated argument(s) used in '{f.__name__}':"
                    f" {', '.join(used_deprecated_args)}. Will not be supported from"
                    f" version '{version}'."
                )
                if custom_message is not None:
                    message += "\n\n" + custom_message
                warnings.warn(message, FutureWarning)
            return f(*args, **kwargs)

        return inner_f

    return _inner_deprecate_positional_args


def _deprecate_method(*, version: str, message: Optional[str] = None):
    """Decorator to issue warnings when using a deprecated method.

    Args:
        version (`str`):
            The version when deprecated arguments will result in error.
        message (`str`, *optional*):
            Warning message that is raised. If not passed, a default warning message
            will be created.
    """

    def _inner_deprecate_method(f):
        @wraps(f)
        def inner_f(*args, **kwargs):
            warning_message = (
                f"'{f.__name__}' (from '{f.__module__}') is deprecated and will be removed from version '{version}'."
            )
            if message is not None:
                warning_message += " " + message
            warnings.warn(warning_message, FutureWarning)
            return f(*args, **kwargs)

        return inner_f

    return _inner_deprecate_method


def _deprecate_list_output(*, version: str):
    """Decorator to deprecate the usage as a list of the output of a method.

    To be used when a method currently returns a list of objects but is planned to return
    an generator instead in the future. Output is still a list but tweaked to issue a
    warning message when it is specifically used as a list (e.g. get/set/del item, get
    length,...).

    Args:
        version (`str`):
            The version when output will start to be an generator.
    """

    def _inner_deprecate_method(f):
        @wraps(f)
        def inner_f(*args, **kwargs):
            list_value = f(*args, **kwargs)
            return DeprecatedList(
                list_value,
                warning_message=(
                    "'{f.__name__}' currently returns a list of objects but is planned"
                    " to be a generator starting from version {version} in order to"
                    " implement pagination. Please avoid to use"
                    " `{f.__name__}(...).{attr_name}` or explicitly convert the output"
                    " to a list first with `[item for item in {f.__name__}(...)]`.".format(
                        f=f,
                        version=version,
                        # Dumb but working workaround to render `attr_name` later
                        # Taken from https://stackoverflow.com/a/35300723
                        attr_name="{attr_name}",
                    )
                ),
            )

        return inner_f

    return _inner_deprecate_method


def _empty_gen() -> Generator:
    # Create an empty generator
    # Taken from https://stackoverflow.com/a/13243870
    return
    yield


# Build the set of attributes that are specific to a List object (and will be deprecated)
_LIST_ONLY_ATTRS = frozenset(set(dir([])) - set(dir(_empty_gen())))


class DeprecateListMetaclass(type):
    """Metaclass that overwrites all list-only methods, including magic ones."""

    def __new__(cls, clsname, bases, attrs):
        # Check consistency
        if "_deprecate" not in attrs:
            raise TypeError("A `_deprecate` method must be implemented to use `DeprecateListMetaclass`.")
        if list not in bases:
            raise TypeError("Class must inherit from `list` to use `DeprecateListMetaclass`.")

        # Create decorator to deprecate list-only methods, including magic ones
        def _with_deprecation(f, name):
            @wraps(f)
            def _inner(self, *args, **kwargs):
                self._deprecate(name)  # Use the `_deprecate`
                return f(self, *args, **kwargs)

            return _inner

        # Deprecate list-only methods
        for attr in _LIST_ONLY_ATTRS:
            attrs[attr] = _with_deprecation(getattr(list, attr), attr)

        return super().__new__(cls, clsname, bases, attrs)


class DeprecatedList(list, metaclass=DeprecateListMetaclass):
    """Custom List class for which all calls to a list-specific method is deprecated.

    Methods that are shared with a generator are not deprecated.
    See `_deprecate_list_output` for more details.
    """

    def __init__(self, iterable, warning_message: str):
        """Initialize the list with a default warning message.

        Warning message will be formatted at runtime with a "{attr_name}" value.
        """
        super().__init__(iterable)
        self._deprecation_msg = warning_message

    def _deprecate(self, attr_name: str) -> None:
        warnings.warn(self._deprecation_msg.format(attr_name=attr_name), FutureWarning)
