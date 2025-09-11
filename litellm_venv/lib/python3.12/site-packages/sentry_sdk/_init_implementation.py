import warnings

from typing import TYPE_CHECKING

import sentry_sdk

if TYPE_CHECKING:
    from typing import Any, ContextManager, Optional

    import sentry_sdk.consts


class _InitGuard:
    _CONTEXT_MANAGER_DEPRECATION_WARNING_MESSAGE = (
        "Using the return value of sentry_sdk.init as a context manager "
        "and manually calling the __enter__ and __exit__ methods on the "
        "return value are deprecated. We are no longer maintaining this "
        "functionality, and we will remove it in the next major release."
    )

    def __init__(self, client):
        # type: (sentry_sdk.Client) -> None
        self._client = client

    def __enter__(self):
        # type: () -> _InitGuard
        warnings.warn(
            self._CONTEXT_MANAGER_DEPRECATION_WARNING_MESSAGE,
            stacklevel=2,
            category=DeprecationWarning,
        )

        return self

    def __exit__(self, exc_type, exc_value, tb):
        # type: (Any, Any, Any) -> None
        warnings.warn(
            self._CONTEXT_MANAGER_DEPRECATION_WARNING_MESSAGE,
            stacklevel=2,
            category=DeprecationWarning,
        )

        c = self._client
        if c is not None:
            c.close()


def _check_python_deprecations():
    # type: () -> None
    # Since we're likely to deprecate Python versions in the future, I'm keeping
    # this handy function around. Use this to detect the Python version used and
    # to output logger.warning()s if it's deprecated.
    pass


def _init(*args, **kwargs):
    # type: (*Optional[str], **Any) -> ContextManager[Any]
    """Initializes the SDK and optionally integrations.

    This takes the same arguments as the client constructor.
    """
    client = sentry_sdk.Client(*args, **kwargs)
    sentry_sdk.get_global_scope().set_client(client)
    _check_python_deprecations()
    rv = _InitGuard(client)
    return rv


if TYPE_CHECKING:
    # Make mypy, PyCharm and other static analyzers think `init` is a type to
    # have nicer autocompletion for params.
    #
    # Use `ClientConstructor` to define the argument types of `init` and
    # `ContextManager[Any]` to tell static analyzers about the return type.

    class init(sentry_sdk.consts.ClientConstructor, _InitGuard):  # noqa: N801
        pass

else:
    # Alias `init` for actual usage. Go through the lambda indirection to throw
    # PyCharm off of the weakly typed signature (it would otherwise discover
    # both the weakly typed signature of `_init` and our faked `init` type).

    init = (lambda: _init)()
