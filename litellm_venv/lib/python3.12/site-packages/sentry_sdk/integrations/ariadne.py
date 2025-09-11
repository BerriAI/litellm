from importlib import import_module

import sentry_sdk
from sentry_sdk import get_client, capture_event
from sentry_sdk.integrations import _check_minimum_version, DidNotEnable, Integration
from sentry_sdk.integrations.logging import ignore_logger
from sentry_sdk.integrations._wsgi_common import request_body_within_bounds
from sentry_sdk.scope import should_send_default_pii
from sentry_sdk.utils import (
    capture_internal_exceptions,
    ensure_integration_enabled,
    event_from_exception,
    package_version,
)

try:
    # importing like this is necessary due to name shadowing in ariadne
    # (ariadne.graphql is also a function)
    ariadne_graphql = import_module("ariadne.graphql")
except ImportError:
    raise DidNotEnable("ariadne is not installed")

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any, Dict, List, Optional
    from ariadne.types import GraphQLError, GraphQLResult, GraphQLSchema, QueryParser  # type: ignore
    from graphql.language.ast import DocumentNode
    from sentry_sdk._types import Event, EventProcessor


class AriadneIntegration(Integration):
    identifier = "ariadne"

    @staticmethod
    def setup_once():
        # type: () -> None
        version = package_version("ariadne")
        _check_minimum_version(AriadneIntegration, version)

        ignore_logger("ariadne")

        _patch_graphql()


def _patch_graphql():
    # type: () -> None
    old_parse_query = ariadne_graphql.parse_query
    old_handle_errors = ariadne_graphql.handle_graphql_errors
    old_handle_query_result = ariadne_graphql.handle_query_result

    @ensure_integration_enabled(AriadneIntegration, old_parse_query)
    def _sentry_patched_parse_query(context_value, query_parser, data):
        # type: (Optional[Any], Optional[QueryParser], Any) -> DocumentNode
        event_processor = _make_request_event_processor(data)
        sentry_sdk.get_isolation_scope().add_event_processor(event_processor)

        result = old_parse_query(context_value, query_parser, data)
        return result

    @ensure_integration_enabled(AriadneIntegration, old_handle_errors)
    def _sentry_patched_handle_graphql_errors(errors, *args, **kwargs):
        # type: (List[GraphQLError], Any, Any) -> GraphQLResult
        result = old_handle_errors(errors, *args, **kwargs)

        event_processor = _make_response_event_processor(result[1])
        sentry_sdk.get_isolation_scope().add_event_processor(event_processor)

        client = get_client()
        if client.is_active():
            with capture_internal_exceptions():
                for error in errors:
                    event, hint = event_from_exception(
                        error,
                        client_options=client.options,
                        mechanism={
                            "type": AriadneIntegration.identifier,
                            "handled": False,
                        },
                    )
                    capture_event(event, hint=hint)

        return result

    @ensure_integration_enabled(AriadneIntegration, old_handle_query_result)
    def _sentry_patched_handle_query_result(result, *args, **kwargs):
        # type: (Any, Any, Any) -> GraphQLResult
        query_result = old_handle_query_result(result, *args, **kwargs)

        event_processor = _make_response_event_processor(query_result[1])
        sentry_sdk.get_isolation_scope().add_event_processor(event_processor)

        client = get_client()
        if client.is_active():
            with capture_internal_exceptions():
                for error in result.errors or []:
                    event, hint = event_from_exception(
                        error,
                        client_options=client.options,
                        mechanism={
                            "type": AriadneIntegration.identifier,
                            "handled": False,
                        },
                    )
                    capture_event(event, hint=hint)

        return query_result

    ariadne_graphql.parse_query = _sentry_patched_parse_query  # type: ignore
    ariadne_graphql.handle_graphql_errors = _sentry_patched_handle_graphql_errors  # type: ignore
    ariadne_graphql.handle_query_result = _sentry_patched_handle_query_result  # type: ignore


def _make_request_event_processor(data):
    # type: (GraphQLSchema) -> EventProcessor
    """Add request data and api_target to events."""

    def inner(event, hint):
        # type: (Event, dict[str, Any]) -> Event
        if not isinstance(data, dict):
            return event

        with capture_internal_exceptions():
            try:
                content_length = int(
                    (data.get("headers") or {}).get("Content-Length", 0)
                )
            except (TypeError, ValueError):
                return event

            if should_send_default_pii() and request_body_within_bounds(
                get_client(), content_length
            ):
                request_info = event.setdefault("request", {})
                request_info["api_target"] = "graphql"
                request_info["data"] = data

            elif event.get("request", {}).get("data"):
                del event["request"]["data"]

        return event

    return inner


def _make_response_event_processor(response):
    # type: (Dict[str, Any]) -> EventProcessor
    """Add response data to the event's response context."""

    def inner(event, hint):
        # type: (Event, dict[str, Any]) -> Event
        with capture_internal_exceptions():
            if should_send_default_pii() and response.get("errors"):
                contexts = event.setdefault("contexts", {})
                contexts["response"] = {
                    "data": response,
                }

        return event

    return inner
