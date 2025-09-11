import functools
import sys
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from os import environ

import sentry_sdk
from sentry_sdk.api import continue_trace
from sentry_sdk.consts import OP
from sentry_sdk.integrations import Integration
from sentry_sdk.integrations._wsgi_common import _filter_headers
from sentry_sdk.scope import should_send_default_pii
from sentry_sdk.tracing import TRANSACTION_SOURCE_COMPONENT
from sentry_sdk.utils import (
    AnnotatedValue,
    capture_internal_exceptions,
    event_from_exception,
    logger,
    TimeoutThread,
    reraise,
)

from typing import TYPE_CHECKING

# Constants
TIMEOUT_WARNING_BUFFER = 1.5  # Buffer time required to send timeout warning to Sentry
MILLIS_TO_SECONDS = 1000.0

if TYPE_CHECKING:
    from typing import Any
    from typing import TypeVar
    from typing import Callable
    from typing import Optional

    from sentry_sdk._types import EventProcessor, Event, Hint

    F = TypeVar("F", bound=Callable[..., Any])


def _wrap_func(func):
    # type: (F) -> F
    @functools.wraps(func)
    def sentry_func(functionhandler, gcp_event, *args, **kwargs):
        # type: (Any, Any, *Any, **Any) -> Any
        client = sentry_sdk.get_client()

        integration = client.get_integration(GcpIntegration)
        if integration is None:
            return func(functionhandler, gcp_event, *args, **kwargs)

        configured_time = environ.get("FUNCTION_TIMEOUT_SEC")
        if not configured_time:
            logger.debug(
                "The configured timeout could not be fetched from Cloud Functions configuration."
            )
            return func(functionhandler, gcp_event, *args, **kwargs)

        configured_time = int(configured_time)

        initial_time = datetime.now(timezone.utc)

        with sentry_sdk.isolation_scope() as scope:
            with capture_internal_exceptions():
                scope.clear_breadcrumbs()
                scope.add_event_processor(
                    _make_request_event_processor(
                        gcp_event, configured_time, initial_time
                    )
                )
                scope.set_tag("gcp_region", environ.get("FUNCTION_REGION"))
                timeout_thread = None
                if (
                    integration.timeout_warning
                    and configured_time > TIMEOUT_WARNING_BUFFER
                ):
                    waiting_time = configured_time - TIMEOUT_WARNING_BUFFER

                    timeout_thread = TimeoutThread(waiting_time, configured_time)

                    # Starting the thread to raise timeout warning exception
                    timeout_thread.start()

            headers = {}
            if hasattr(gcp_event, "headers"):
                headers = gcp_event.headers

            transaction = continue_trace(
                headers,
                op=OP.FUNCTION_GCP,
                name=environ.get("FUNCTION_NAME", ""),
                source=TRANSACTION_SOURCE_COMPONENT,
                origin=GcpIntegration.origin,
            )
            sampling_context = {
                "gcp_env": {
                    "function_name": environ.get("FUNCTION_NAME"),
                    "function_entry_point": environ.get("ENTRY_POINT"),
                    "function_identity": environ.get("FUNCTION_IDENTITY"),
                    "function_region": environ.get("FUNCTION_REGION"),
                    "function_project": environ.get("GCP_PROJECT"),
                },
                "gcp_event": gcp_event,
            }
            with sentry_sdk.start_transaction(
                transaction, custom_sampling_context=sampling_context
            ):
                try:
                    return func(functionhandler, gcp_event, *args, **kwargs)
                except Exception:
                    exc_info = sys.exc_info()
                    sentry_event, hint = event_from_exception(
                        exc_info,
                        client_options=client.options,
                        mechanism={"type": "gcp", "handled": False},
                    )
                    sentry_sdk.capture_event(sentry_event, hint=hint)
                    reraise(*exc_info)
                finally:
                    if timeout_thread:
                        timeout_thread.stop()
                    # Flush out the event queue
                    client.flush()

    return sentry_func  # type: ignore


class GcpIntegration(Integration):
    identifier = "gcp"
    origin = f"auto.function.{identifier}"

    def __init__(self, timeout_warning=False):
        # type: (bool) -> None
        self.timeout_warning = timeout_warning

    @staticmethod
    def setup_once():
        # type: () -> None
        import __main__ as gcp_functions

        if not hasattr(gcp_functions, "worker_v1"):
            logger.warning(
                "GcpIntegration currently supports only Python 3.7 runtime environment."
            )
            return

        worker1 = gcp_functions.worker_v1

        worker1.FunctionHandler.invoke_user_function = _wrap_func(
            worker1.FunctionHandler.invoke_user_function
        )


def _make_request_event_processor(gcp_event, configured_timeout, initial_time):
    # type: (Any, Any, Any) -> EventProcessor

    def event_processor(event, hint):
        # type: (Event, Hint) -> Optional[Event]

        final_time = datetime.now(timezone.utc)
        time_diff = final_time - initial_time

        execution_duration_in_millis = time_diff / timedelta(milliseconds=1)

        extra = event.setdefault("extra", {})
        extra["google cloud functions"] = {
            "function_name": environ.get("FUNCTION_NAME"),
            "function_entry_point": environ.get("ENTRY_POINT"),
            "function_identity": environ.get("FUNCTION_IDENTITY"),
            "function_region": environ.get("FUNCTION_REGION"),
            "function_project": environ.get("GCP_PROJECT"),
            "execution_duration_in_millis": execution_duration_in_millis,
            "configured_timeout_in_seconds": configured_timeout,
        }

        extra["google cloud logs"] = {
            "url": _get_google_cloud_logs_url(final_time),
        }

        request = event.get("request", {})

        request["url"] = "gcp:///{}".format(environ.get("FUNCTION_NAME"))

        if hasattr(gcp_event, "method"):
            request["method"] = gcp_event.method

        if hasattr(gcp_event, "query_string"):
            request["query_string"] = gcp_event.query_string.decode("utf-8")

        if hasattr(gcp_event, "headers"):
            request["headers"] = _filter_headers(gcp_event.headers)

        if should_send_default_pii():
            if hasattr(gcp_event, "data"):
                request["data"] = gcp_event.data
        else:
            if hasattr(gcp_event, "data"):
                # Unfortunately couldn't find a way to get structured body from GCP
                # event. Meaning every body is unstructured to us.
                request["data"] = AnnotatedValue.removed_because_raw_data()

        event["request"] = deepcopy(request)

        return event

    return event_processor


def _get_google_cloud_logs_url(final_time):
    # type: (datetime) -> str
    """
    Generates a Google Cloud Logs console URL based on the environment variables
    Arguments:
        final_time {datetime} -- Final time
    Returns:
        str -- Google Cloud Logs Console URL to logs.
    """
    hour_ago = final_time - timedelta(hours=1)
    formatstring = "%Y-%m-%dT%H:%M:%SZ"

    url = (
        "https://console.cloud.google.com/logs/viewer?project={project}&resource=cloud_function"
        "%2Ffunction_name%2F{function_name}%2Fregion%2F{region}&minLogLevel=0&expandAll=false"
        "&timestamp={timestamp_end}&customFacets=&limitCustomFacetWidth=true"
        "&dateRangeStart={timestamp_start}&dateRangeEnd={timestamp_end}"
        "&interval=PT1H&scrollTimestamp={timestamp_end}"
    ).format(
        project=environ.get("GCP_PROJECT"),
        function_name=environ.get("FUNCTION_NAME"),
        region=environ.get("FUNCTION_REGION"),
        timestamp_end=final_time.strftime(formatstring),
        timestamp_start=hour_ago.strftime(formatstring),
    )

    return url
