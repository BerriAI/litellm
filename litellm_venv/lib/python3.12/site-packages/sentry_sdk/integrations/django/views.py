import functools

import sentry_sdk
from sentry_sdk.consts import OP

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any


try:
    from asyncio import iscoroutinefunction
except ImportError:
    iscoroutinefunction = None  # type: ignore


try:
    from sentry_sdk.integrations.django.asgi import wrap_async_view
except (ImportError, SyntaxError):
    wrap_async_view = None  # type: ignore


def patch_views():
    # type: () -> None

    from django.core.handlers.base import BaseHandler
    from django.template.response import SimpleTemplateResponse
    from sentry_sdk.integrations.django import DjangoIntegration

    old_make_view_atomic = BaseHandler.make_view_atomic
    old_render = SimpleTemplateResponse.render

    def sentry_patched_render(self):
        # type: (SimpleTemplateResponse) -> Any
        with sentry_sdk.start_span(
            op=OP.VIEW_RESPONSE_RENDER,
            name="serialize response",
            origin=DjangoIntegration.origin,
        ):
            return old_render(self)

    @functools.wraps(old_make_view_atomic)
    def sentry_patched_make_view_atomic(self, *args, **kwargs):
        # type: (Any, *Any, **Any) -> Any
        callback = old_make_view_atomic(self, *args, **kwargs)

        # XXX: The wrapper function is created for every request. Find more
        # efficient way to wrap views (or build a cache?)

        integration = sentry_sdk.get_client().get_integration(DjangoIntegration)
        if integration is not None and integration.middleware_spans:
            is_async_view = (
                iscoroutinefunction is not None
                and wrap_async_view is not None
                and iscoroutinefunction(callback)
            )
            if is_async_view:
                sentry_wrapped_callback = wrap_async_view(callback)
            else:
                sentry_wrapped_callback = _wrap_sync_view(callback)

        else:
            sentry_wrapped_callback = callback

        return sentry_wrapped_callback

    SimpleTemplateResponse.render = sentry_patched_render
    BaseHandler.make_view_atomic = sentry_patched_make_view_atomic


def _wrap_sync_view(callback):
    # type: (Any) -> Any
    from sentry_sdk.integrations.django import DjangoIntegration

    @functools.wraps(callback)
    def sentry_wrapped_callback(request, *args, **kwargs):
        # type: (Any, *Any, **Any) -> Any
        current_scope = sentry_sdk.get_current_scope()
        if current_scope.transaction is not None:
            current_scope.transaction.update_active_thread()

        sentry_scope = sentry_sdk.get_isolation_scope()
        # set the active thread id to the handler thread for sync views
        # this isn't necessary for async views since that runs on main
        if sentry_scope.profile is not None:
            sentry_scope.profile.update_active_thread_id()

        with sentry_sdk.start_span(
            op=OP.VIEW_RENDER,
            name=request.resolver_match.view_name,
            origin=DjangoIntegration.origin,
        ):
            return callback(request, *args, **kwargs)

    return sentry_wrapped_callback
