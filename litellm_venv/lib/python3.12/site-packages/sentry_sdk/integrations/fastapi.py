import asyncio
from copy import deepcopy
from functools import wraps

import sentry_sdk
from sentry_sdk.integrations import DidNotEnable
from sentry_sdk.scope import should_send_default_pii
from sentry_sdk.tracing import SOURCE_FOR_STYLE, TRANSACTION_SOURCE_ROUTE
from sentry_sdk.utils import (
    transaction_from_function,
    logger,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any, Callable, Dict
    from sentry_sdk._types import Event

try:
    from sentry_sdk.integrations.starlette import (
        StarletteIntegration,
        StarletteRequestExtractor,
    )
except DidNotEnable:
    raise DidNotEnable("Starlette is not installed")

try:
    import fastapi  # type: ignore
except ImportError:
    raise DidNotEnable("FastAPI is not installed")


_DEFAULT_TRANSACTION_NAME = "generic FastAPI request"


class FastApiIntegration(StarletteIntegration):
    identifier = "fastapi"

    @staticmethod
    def setup_once():
        # type: () -> None
        patch_get_request_handler()


def _set_transaction_name_and_source(scope, transaction_style, request):
    # type: (sentry_sdk.Scope, str, Any) -> None
    name = ""

    if transaction_style == "endpoint":
        endpoint = request.scope.get("endpoint")
        if endpoint:
            name = transaction_from_function(endpoint) or ""

    elif transaction_style == "url":
        route = request.scope.get("route")
        if route:
            path = getattr(route, "path", None)
            if path is not None:
                name = path

    if not name:
        name = _DEFAULT_TRANSACTION_NAME
        source = TRANSACTION_SOURCE_ROUTE
    else:
        source = SOURCE_FOR_STYLE[transaction_style]

    scope.set_transaction_name(name, source=source)
    logger.debug(
        "[FastAPI] Set transaction name and source on scope: %s / %s", name, source
    )


def patch_get_request_handler():
    # type: () -> None
    old_get_request_handler = fastapi.routing.get_request_handler

    def _sentry_get_request_handler(*args, **kwargs):
        # type: (*Any, **Any) -> Any
        dependant = kwargs.get("dependant")
        if (
            dependant
            and dependant.call is not None
            and not asyncio.iscoroutinefunction(dependant.call)
        ):
            old_call = dependant.call

            @wraps(old_call)
            def _sentry_call(*args, **kwargs):
                # type: (*Any, **Any) -> Any
                current_scope = sentry_sdk.get_current_scope()
                if current_scope.transaction is not None:
                    current_scope.transaction.update_active_thread()

                sentry_scope = sentry_sdk.get_isolation_scope()
                if sentry_scope.profile is not None:
                    sentry_scope.profile.update_active_thread_id()

                return old_call(*args, **kwargs)

            dependant.call = _sentry_call

        old_app = old_get_request_handler(*args, **kwargs)

        async def _sentry_app(*args, **kwargs):
            # type: (*Any, **Any) -> Any
            integration = sentry_sdk.get_client().get_integration(FastApiIntegration)
            if integration is None:
                return await old_app(*args, **kwargs)

            request = args[0]

            _set_transaction_name_and_source(
                sentry_sdk.get_current_scope(), integration.transaction_style, request
            )
            sentry_scope = sentry_sdk.get_isolation_scope()
            extractor = StarletteRequestExtractor(request)
            info = await extractor.extract_request_info()

            def _make_request_event_processor(req, integration):
                # type: (Any, Any) -> Callable[[Event, Dict[str, Any]], Event]
                def event_processor(event, hint):
                    # type: (Event, Dict[str, Any]) -> Event

                    # Extract information from request
                    request_info = event.get("request", {})
                    if info:
                        if "cookies" in info and should_send_default_pii():
                            request_info["cookies"] = info["cookies"]
                        if "data" in info:
                            request_info["data"] = info["data"]
                    event["request"] = deepcopy(request_info)

                    return event

                return event_processor

            sentry_scope._name = FastApiIntegration.identifier
            sentry_scope.add_event_processor(
                _make_request_event_processor(request, integration)
            )

            return await old_app(*args, **kwargs)

        return _sentry_app

    fastapi.routing.get_request_handler = _sentry_get_request_handler
