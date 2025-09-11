"""
Create spans from Django middleware invocations
"""

from functools import wraps

from django import VERSION as DJANGO_VERSION

import sentry_sdk
from sentry_sdk.consts import OP
from sentry_sdk.utils import (
    ContextVar,
    transaction_from_function,
    capture_internal_exceptions,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any
    from typing import Callable
    from typing import Optional
    from typing import TypeVar

    from sentry_sdk.tracing import Span

    F = TypeVar("F", bound=Callable[..., Any])

_import_string_should_wrap_middleware = ContextVar(
    "import_string_should_wrap_middleware"
)

DJANGO_SUPPORTS_ASYNC_MIDDLEWARE = DJANGO_VERSION >= (3, 1)

if not DJANGO_SUPPORTS_ASYNC_MIDDLEWARE:
    _asgi_middleware_mixin_factory = lambda _: object
else:
    from .asgi import _asgi_middleware_mixin_factory


def patch_django_middlewares():
    # type: () -> None
    from django.core.handlers import base

    old_import_string = base.import_string

    def sentry_patched_import_string(dotted_path):
        # type: (str) -> Any
        rv = old_import_string(dotted_path)

        if _import_string_should_wrap_middleware.get(None):
            rv = _wrap_middleware(rv, dotted_path)

        return rv

    base.import_string = sentry_patched_import_string

    old_load_middleware = base.BaseHandler.load_middleware

    def sentry_patched_load_middleware(*args, **kwargs):
        # type: (Any, Any) -> Any
        _import_string_should_wrap_middleware.set(True)
        try:
            return old_load_middleware(*args, **kwargs)
        finally:
            _import_string_should_wrap_middleware.set(False)

    base.BaseHandler.load_middleware = sentry_patched_load_middleware


def _wrap_middleware(middleware, middleware_name):
    # type: (Any, str) -> Any
    from sentry_sdk.integrations.django import DjangoIntegration

    def _check_middleware_span(old_method):
        # type: (Callable[..., Any]) -> Optional[Span]
        integration = sentry_sdk.get_client().get_integration(DjangoIntegration)
        if integration is None or not integration.middleware_spans:
            return None

        function_name = transaction_from_function(old_method)

        description = middleware_name
        function_basename = getattr(old_method, "__name__", None)
        if function_basename:
            description = "{}.{}".format(description, function_basename)

        middleware_span = sentry_sdk.start_span(
            op=OP.MIDDLEWARE_DJANGO,
            name=description,
            origin=DjangoIntegration.origin,
        )
        middleware_span.set_tag("django.function_name", function_name)
        middleware_span.set_tag("django.middleware_name", middleware_name)

        return middleware_span

    def _get_wrapped_method(old_method):
        # type: (F) -> F
        with capture_internal_exceptions():

            def sentry_wrapped_method(*args, **kwargs):
                # type: (*Any, **Any) -> Any
                middleware_span = _check_middleware_span(old_method)

                if middleware_span is None:
                    return old_method(*args, **kwargs)

                with middleware_span:
                    return old_method(*args, **kwargs)

            try:
                # fails for __call__ of function on Python 2 (see py2.7-django-1.11)
                sentry_wrapped_method = wraps(old_method)(sentry_wrapped_method)

                # Necessary for Django 3.1
                sentry_wrapped_method.__self__ = old_method.__self__  # type: ignore
            except Exception:
                pass

            return sentry_wrapped_method  # type: ignore

        return old_method

    class SentryWrappingMiddleware(
        _asgi_middleware_mixin_factory(_check_middleware_span)  # type: ignore
    ):
        sync_capable = getattr(middleware, "sync_capable", True)
        async_capable = DJANGO_SUPPORTS_ASYNC_MIDDLEWARE and getattr(
            middleware, "async_capable", False
        )

        def __init__(self, get_response=None, *args, **kwargs):
            # type: (Optional[Callable[..., Any]], *Any, **Any) -> None
            if get_response:
                self._inner = middleware(get_response, *args, **kwargs)
            else:
                self._inner = middleware(*args, **kwargs)
            self.get_response = get_response
            self._call_method = None
            if self.async_capable:
                super().__init__(get_response)

        # We need correct behavior for `hasattr()`, which we can only determine
        # when we have an instance of the middleware we're wrapping.
        def __getattr__(self, method_name):
            # type: (str) -> Any
            if method_name not in (
                "process_request",
                "process_view",
                "process_template_response",
                "process_response",
                "process_exception",
            ):
                raise AttributeError()

            old_method = getattr(self._inner, method_name)
            rv = _get_wrapped_method(old_method)
            self.__dict__[method_name] = rv
            return rv

        def __call__(self, *args, **kwargs):
            # type: (*Any, **Any) -> Any
            if hasattr(self, "async_route_check") and self.async_route_check():
                return self.__acall__(*args, **kwargs)

            f = self._call_method
            if f is None:
                self._call_method = f = self._inner.__call__

            middleware_span = _check_middleware_span(old_method=f)

            if middleware_span is None:
                return f(*args, **kwargs)

            with middleware_span:
                return f(*args, **kwargs)

    for attr in (
        "__name__",
        "__module__",
        "__qualname__",
    ):
        if hasattr(middleware, attr):
            setattr(SentryWrappingMiddleware, attr, getattr(middleware, attr))

    return SentryWrappingMiddleware
