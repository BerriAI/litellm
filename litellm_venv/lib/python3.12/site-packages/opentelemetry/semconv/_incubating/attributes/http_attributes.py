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


from enum import Enum

from deprecated import deprecated

HTTP_CONNECTION_STATE = "http.connection.state"
"""
State of the HTTP connection in the HTTP connection pool.
"""

HTTP_FLAVOR = "http.flavor"
"""
Deprecated: Replaced by `network.protocol.name`.
"""

HTTP_METHOD = "http.method"
"""
Deprecated: Replaced by `http.request.method`.
"""

HTTP_REQUEST_BODY_SIZE = "http.request.body.size"
"""
The size of the request payload body in bytes. This is the number of bytes transferred excluding headers and is often, but not always, present as the [Content-Length](https://www.rfc-editor.org/rfc/rfc9110.html#field.content-length) header. For requests using transport encoding, this should be the compressed size.
"""

HTTP_REQUEST_HEADER_TEMPLATE = "http.request.header"
"""
Deprecated in favor of stable :py:const:`opentelemetry.semconv.attributes.http_attributes.HTTP_REQUEST_HEADER_TEMPLATE`.
"""

HTTP_REQUEST_METHOD = "http.request.method"
"""
Deprecated in favor of stable :py:const:`opentelemetry.semconv.attributes.http_attributes.HTTP_REQUEST_METHOD`.
"""

HTTP_REQUEST_METHOD_ORIGINAL = "http.request.method_original"
"""
Deprecated in favor of stable :py:const:`opentelemetry.semconv.attributes.http_attributes.HTTP_REQUEST_METHOD_ORIGINAL`.
"""

HTTP_REQUEST_RESEND_COUNT = "http.request.resend_count"
"""
Deprecated in favor of stable :py:const:`opentelemetry.semconv.attributes.http_attributes.HTTP_REQUEST_RESEND_COUNT`.
"""

HTTP_REQUEST_SIZE = "http.request.size"
"""
The total size of the request in bytes. This should be the total number of bytes sent over the wire, including the request line (HTTP/1.1), framing (HTTP/2 and HTTP/3), headers, and request body if any.
"""

HTTP_REQUEST_CONTENT_LENGTH = "http.request_content_length"
"""
Deprecated: Replaced by `http.request.header.content-length`.
"""

HTTP_RESPONSE_BODY_SIZE = "http.response.body.size"
"""
The size of the response payload body in bytes. This is the number of bytes transferred excluding headers and is often, but not always, present as the [Content-Length](https://www.rfc-editor.org/rfc/rfc9110.html#field.content-length) header. For requests using transport encoding, this should be the compressed size.
"""

HTTP_RESPONSE_HEADER_TEMPLATE = "http.response.header"
"""
Deprecated in favor of stable :py:const:`opentelemetry.semconv.attributes.http_attributes.HTTP_RESPONSE_HEADER_TEMPLATE`.
"""

HTTP_RESPONSE_SIZE = "http.response.size"
"""
The total size of the response in bytes. This should be the total number of bytes sent over the wire, including the status line (HTTP/1.1), framing (HTTP/2 and HTTP/3), headers, and response body and trailers if any.
"""

HTTP_RESPONSE_STATUS_CODE = "http.response.status_code"
"""
Deprecated in favor of stable :py:const:`opentelemetry.semconv.attributes.http_attributes.HTTP_RESPONSE_STATUS_CODE`.
"""

HTTP_RESPONSE_CONTENT_LENGTH = "http.response_content_length"
"""
Deprecated: Replaced by `http.response.header.content-length`.
"""

HTTP_ROUTE = "http.route"
"""
Deprecated in favor of stable :py:const:`opentelemetry.semconv.attributes.http_attributes.HTTP_ROUTE`.
"""

HTTP_SCHEME = "http.scheme"
"""
Deprecated: Replaced by `url.scheme` instead.
"""

HTTP_STATUS_CODE = "http.status_code"
"""
Deprecated: Replaced by `http.response.status_code`.
"""

HTTP_TARGET = "http.target"
"""
Deprecated: Split to `url.path` and `url.query.
"""

HTTP_URL = "http.url"
"""
Deprecated: Replaced by `url.full`.
"""

HTTP_USER_AGENT = "http.user_agent"
"""
Deprecated: Replaced by `user_agent.original`.
"""


class HttpConnectionStateValues(Enum):
    ACTIVE = "active"
    """active state."""
    IDLE = "idle"
    """idle state."""


@deprecated(
    reason="The attribute http.flavor is deprecated - Replaced by `network.protocol.name`"
)
class HttpFlavorValues(Enum):
    HTTP_1_0 = "1.0"
    """HTTP/1.0."""
    HTTP_1_1 = "1.1"
    """HTTP/1.1."""
    HTTP_2_0 = "2.0"
    """HTTP/2."""
    HTTP_3_0 = "3.0"
    """HTTP/3."""
    SPDY = "SPDY"
    """SPDY protocol."""
    QUIC = "QUIC"
    """QUIC protocol."""


@deprecated(
    reason="Deprecated in favor of stable :py:const:`opentelemetry.semconv.attributes.http_attributes.HttpRequestMethodValues`."
)
class HttpRequestMethodValues(Enum):
    CONNECT = "CONNECT"
    """CONNECT method."""
    DELETE = "DELETE"
    """DELETE method."""
    GET = "GET"
    """GET method."""
    HEAD = "HEAD"
    """HEAD method."""
    OPTIONS = "OPTIONS"
    """OPTIONS method."""
    PATCH = "PATCH"
    """PATCH method."""
    POST = "POST"
    """POST method."""
    PUT = "PUT"
    """PUT method."""
    TRACE = "TRACE"
    """TRACE method."""
    OTHER = "_OTHER"
    """Any HTTP method that the instrumentation has no prior knowledge of."""
