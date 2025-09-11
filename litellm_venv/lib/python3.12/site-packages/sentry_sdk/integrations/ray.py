import inspect
import sys

import sentry_sdk
from sentry_sdk.consts import OP, SPANSTATUS
from sentry_sdk.integrations import _check_minimum_version, DidNotEnable, Integration
from sentry_sdk.tracing import TRANSACTION_SOURCE_TASK
from sentry_sdk.utils import (
    event_from_exception,
    logger,
    package_version,
    qualname_from_function,
    reraise,
)

try:
    import ray  # type: ignore[import-not-found]
except ImportError:
    raise DidNotEnable("Ray not installed.")
import functools

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any, Optional
    from sentry_sdk.utils import ExcInfo


def _check_sentry_initialized():
    # type: () -> None
    if sentry_sdk.get_client().is_active():
        return

    logger.debug(
        "[Tracing] Sentry not initialized in ray cluster worker, performance data will be discarded."
    )


def _patch_ray_remote():
    # type: () -> None
    old_remote = ray.remote

    @functools.wraps(old_remote)
    def new_remote(f, *args, **kwargs):
        # type: (Callable[..., Any], *Any, **Any) -> Callable[..., Any]
        if inspect.isclass(f):
            # Ray Actors
            # (https://docs.ray.io/en/latest/ray-core/actors.html)
            # are not supported
            # (Only Ray Tasks are supported)
            return old_remote(f, *args, *kwargs)

        def _f(*f_args, _tracing=None, **f_kwargs):
            # type: (Any, Optional[dict[str, Any]],  Any) -> Any
            """
            Ray Worker
            """
            _check_sentry_initialized()

            transaction = sentry_sdk.continue_trace(
                _tracing or {},
                op=OP.QUEUE_TASK_RAY,
                name=qualname_from_function(f),
                origin=RayIntegration.origin,
                source=TRANSACTION_SOURCE_TASK,
            )

            with sentry_sdk.start_transaction(transaction) as transaction:
                try:
                    result = f(*f_args, **f_kwargs)
                    transaction.set_status(SPANSTATUS.OK)
                except Exception:
                    transaction.set_status(SPANSTATUS.INTERNAL_ERROR)
                    exc_info = sys.exc_info()
                    _capture_exception(exc_info)
                    reraise(*exc_info)

                return result

        rv = old_remote(_f, *args, *kwargs)
        old_remote_method = rv.remote

        def _remote_method_with_header_propagation(*args, **kwargs):
            # type: (*Any, **Any) -> Any
            """
            Ray Client
            """
            with sentry_sdk.start_span(
                op=OP.QUEUE_SUBMIT_RAY,
                name=qualname_from_function(f),
                origin=RayIntegration.origin,
            ) as span:
                tracing = {
                    k: v
                    for k, v in sentry_sdk.get_current_scope().iter_trace_propagation_headers()
                }
                try:
                    result = old_remote_method(*args, **kwargs, _tracing=tracing)
                    span.set_status(SPANSTATUS.OK)
                except Exception:
                    span.set_status(SPANSTATUS.INTERNAL_ERROR)
                    exc_info = sys.exc_info()
                    _capture_exception(exc_info)
                    reraise(*exc_info)

                return result

        rv.remote = _remote_method_with_header_propagation

        return rv

    ray.remote = new_remote


def _capture_exception(exc_info, **kwargs):
    # type: (ExcInfo, **Any) -> None
    client = sentry_sdk.get_client()

    event, hint = event_from_exception(
        exc_info,
        client_options=client.options,
        mechanism={
            "handled": False,
            "type": RayIntegration.identifier,
        },
    )
    sentry_sdk.capture_event(event, hint=hint)


class RayIntegration(Integration):
    identifier = "ray"
    origin = f"auto.queue.{identifier}"

    @staticmethod
    def setup_once():
        # type: () -> None
        version = package_version("ray")
        _check_minimum_version(RayIntegration, version)

        _patch_ray_remote()
