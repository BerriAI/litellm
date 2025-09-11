# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Manages OpenTelemetry tracing span creation and handling. This is a PREVIEW FEATURE: Coverage and functionality may change."""

import logging
import os

from contextlib import contextmanager

from google.api_core import exceptions as api_exceptions
from google.api_core import retry as api_retry
from google.cloud.storage import __version__
from google.cloud.storage.retry import ConditionalRetryPolicy


ENABLE_OTEL_TRACES_ENV_VAR = "ENABLE_GCS_PYTHON_CLIENT_OTEL_TRACES"
_DEFAULT_ENABLE_OTEL_TRACES_VALUE = False

enable_otel_traces = os.environ.get(
    ENABLE_OTEL_TRACES_ENV_VAR, _DEFAULT_ENABLE_OTEL_TRACES_VALUE
)
logger = logging.getLogger(__name__)

try:
    from opentelemetry import trace

    HAS_OPENTELEMETRY = True

except ImportError:
    logger.debug(
        "This service is instrumented using OpenTelemetry. "
        "OpenTelemetry or one of its components could not be imported; "
        "please add compatible versions of opentelemetry-api and "
        "opentelemetry-instrumentation packages in order to get Storage "
        "Tracing data."
    )
    HAS_OPENTELEMETRY = False

_default_attributes = {
    "rpc.service": "CloudStorage",
    "rpc.system": "http",
    "user_agent.original": f"gcloud-python/{__version__}",
}

_cloud_trace_adoption_attrs = {
    "gcp.client.service": "storage",
    "gcp.client.version": __version__,
    "gcp.client.repo": "googleapis/python-storage",
}


@contextmanager
def create_trace_span(name, attributes=None, client=None, api_request=None, retry=None):
    """Creates a context manager for a new span and set it as the current span
    in the configured tracer. If no configuration exists yields None."""
    if not HAS_OPENTELEMETRY or not enable_otel_traces:
        yield None
        return

    tracer = trace.get_tracer(__name__)
    final_attributes = _get_final_attributes(attributes, client, api_request, retry)
    # Yield new span.
    with tracer.start_as_current_span(
        name=name, kind=trace.SpanKind.CLIENT, attributes=final_attributes
    ) as span:
        try:
            yield span
        except api_exceptions.GoogleAPICallError as error:
            span.set_status(trace.Status(trace.StatusCode.ERROR))
            span.record_exception(error)
            raise


def _get_final_attributes(attributes=None, client=None, api_request=None, retry=None):
    collected_attr = _default_attributes.copy()
    collected_attr.update(_cloud_trace_adoption_attrs)
    if api_request:
        collected_attr.update(_set_api_request_attr(api_request, client))
    if isinstance(retry, api_retry.Retry):
        collected_attr.update(_set_retry_attr(retry))
    if isinstance(retry, ConditionalRetryPolicy):
        collected_attr.update(
            _set_retry_attr(retry.retry_policy, retry.conditional_predicate)
        )
    if attributes:
        collected_attr.update(attributes)
    final_attributes = {k: v for k, v in collected_attr.items() if v is not None}
    return final_attributes


def _set_api_request_attr(request, client):
    attr = {}
    if request.get("method"):
        attr["http.request.method"] = request.get("method")
    if request.get("path"):
        path = request.get("path")
        full_path = f"{client._connection.API_BASE_URL}{path}"
        attr["url.full"] = full_path
    if request.get("timeout"):
        attr["connect_timeout,read_timeout"] = request.get("timeout")
    return attr


def _set_retry_attr(retry, conditional_predicate=None):
    predicate = conditional_predicate if conditional_predicate else retry._predicate
    retry_info = f"multiplier{retry._multiplier}/deadline{retry._deadline}/max{retry._maximum}/initial{retry._initial}/predicate{predicate}"
    return {"retry": retry_info}
