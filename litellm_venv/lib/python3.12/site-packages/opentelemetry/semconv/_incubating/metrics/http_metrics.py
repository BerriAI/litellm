# Copyright The OpenTelemetry Authors
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


from opentelemetry.metrics import Histogram, Meter, UpDownCounter

HTTP_CLIENT_ACTIVE_REQUESTS = "http.client.active_requests"
"""
Number of active HTTP requests
Instrument: updowncounter
Unit: {request}
"""


def create_http_client_active_requests(meter: Meter) -> UpDownCounter:
    """Number of active HTTP requests"""
    return meter.create_up_down_counter(
        name="http.client.active_requests",
        description="Number of active HTTP requests.",
        unit="{request}",
    )


HTTP_CLIENT_CONNECTION_DURATION = "http.client.connection.duration"
"""
The duration of the successfully established outbound HTTP connections
Instrument: histogram
Unit: s
"""


def create_http_client_connection_duration(meter: Meter) -> Histogram:
    """The duration of the successfully established outbound HTTP connections"""
    return meter.create_histogram(
        name="http.client.connection.duration",
        description="The duration of the successfully established outbound HTTP connections.",
        unit="s",
    )


HTTP_CLIENT_OPEN_CONNECTIONS = "http.client.open_connections"
"""
Number of outbound HTTP connections that are currently active or idle on the client
Instrument: updowncounter
Unit: {connection}
"""


def create_http_client_open_connections(meter: Meter) -> UpDownCounter:
    """Number of outbound HTTP connections that are currently active or idle on the client"""
    return meter.create_up_down_counter(
        name="http.client.open_connections",
        description="Number of outbound HTTP connections that are currently active or idle on the client.",
        unit="{connection}",
    )


HTTP_CLIENT_REQUEST_BODY_SIZE = "http.client.request.body.size"
"""
Size of HTTP client request bodies
Instrument: histogram
Unit: By
Note: The size of the request payload body in bytes. This is the number of bytes transferred excluding headers and is often, but not always, present as the [Content-Length](https://www.rfc-editor.org/rfc/rfc9110.html#field.content-length) header. For requests using transport encoding, this should be the compressed size.
"""


def create_http_client_request_body_size(meter: Meter) -> Histogram:
    """Size of HTTP client request bodies"""
    return meter.create_histogram(
        name="http.client.request.body.size",
        description="Size of HTTP client request bodies.",
        unit="By",
    )


HTTP_CLIENT_REQUEST_DURATION = "http.client.request.duration"
"""
Deprecated in favor of stable :py:const:`opentelemetry.semconv.metrics.http_metrics.HTTP_CLIENT_REQUEST_DURATION`.
"""


def create_http_client_request_duration(meter: Meter) -> Histogram:
    """Duration of HTTP client requests"""
    return meter.create_histogram(
        name="http.client.request.duration",
        description="Duration of HTTP client requests.",
        unit="s",
    )


HTTP_CLIENT_RESPONSE_BODY_SIZE = "http.client.response.body.size"
"""
Size of HTTP client response bodies
Instrument: histogram
Unit: By
Note: The size of the response payload body in bytes. This is the number of bytes transferred excluding headers and is often, but not always, present as the [Content-Length](https://www.rfc-editor.org/rfc/rfc9110.html#field.content-length) header. For requests using transport encoding, this should be the compressed size.
"""


def create_http_client_response_body_size(meter: Meter) -> Histogram:
    """Size of HTTP client response bodies"""
    return meter.create_histogram(
        name="http.client.response.body.size",
        description="Size of HTTP client response bodies.",
        unit="By",
    )


HTTP_SERVER_ACTIVE_REQUESTS = "http.server.active_requests"
"""
Number of active HTTP server requests
Instrument: updowncounter
Unit: {request}
"""


def create_http_server_active_requests(meter: Meter) -> UpDownCounter:
    """Number of active HTTP server requests"""
    return meter.create_up_down_counter(
        name="http.server.active_requests",
        description="Number of active HTTP server requests.",
        unit="{request}",
    )


HTTP_SERVER_REQUEST_BODY_SIZE = "http.server.request.body.size"
"""
Size of HTTP server request bodies
Instrument: histogram
Unit: By
Note: The size of the request payload body in bytes. This is the number of bytes transferred excluding headers and is often, but not always, present as the [Content-Length](https://www.rfc-editor.org/rfc/rfc9110.html#field.content-length) header. For requests using transport encoding, this should be the compressed size.
"""


def create_http_server_request_body_size(meter: Meter) -> Histogram:
    """Size of HTTP server request bodies"""
    return meter.create_histogram(
        name="http.server.request.body.size",
        description="Size of HTTP server request bodies.",
        unit="By",
    )


HTTP_SERVER_REQUEST_DURATION = "http.server.request.duration"
"""
Deprecated in favor of stable :py:const:`opentelemetry.semconv.metrics.http_metrics.HTTP_SERVER_REQUEST_DURATION`.
"""


def create_http_server_request_duration(meter: Meter) -> Histogram:
    """Duration of HTTP server requests"""
    return meter.create_histogram(
        name="http.server.request.duration",
        description="Duration of HTTP server requests.",
        unit="s",
    )


HTTP_SERVER_RESPONSE_BODY_SIZE = "http.server.response.body.size"
"""
Size of HTTP server response bodies
Instrument: histogram
Unit: By
Note: The size of the response payload body in bytes. This is the number of bytes transferred excluding headers and is often, but not always, present as the [Content-Length](https://www.rfc-editor.org/rfc/rfc9110.html#field.content-length) header. For requests using transport encoding, this should be the compressed size.
"""


def create_http_server_response_body_size(meter: Meter) -> Histogram:
    """Size of HTTP server response bodies"""
    return meter.create_histogram(
        name="http.server.response.body.size",
        description="Size of HTTP server response bodies.",
        unit="By",
    )
