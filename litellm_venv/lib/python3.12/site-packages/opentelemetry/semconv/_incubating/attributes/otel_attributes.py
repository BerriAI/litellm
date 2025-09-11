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

OTEL_LIBRARY_NAME = "otel.library.name"
"""
Deprecated: use the `otel.scope.name` attribute.
"""

OTEL_LIBRARY_VERSION = "otel.library.version"
"""
Deprecated: use the `otel.scope.version` attribute.
"""

OTEL_SCOPE_NAME = "otel.scope.name"
"""
Deprecated in favor of stable :py:const:`opentelemetry.semconv.attributes.otel_attributes.OTEL_SCOPE_NAME`.
"""

OTEL_SCOPE_VERSION = "otel.scope.version"
"""
Deprecated in favor of stable :py:const:`opentelemetry.semconv.attributes.otel_attributes.OTEL_SCOPE_VERSION`.
"""

OTEL_STATUS_CODE = "otel.status_code"
"""
Deprecated in favor of stable :py:const:`opentelemetry.semconv.attributes.otel_attributes.OTEL_STATUS_CODE`.
"""

OTEL_STATUS_DESCRIPTION = "otel.status_description"
"""
Deprecated in favor of stable :py:const:`opentelemetry.semconv.attributes.otel_attributes.OTEL_STATUS_DESCRIPTION`.
"""


@deprecated(
    reason="Deprecated in favor of stable :py:const:`opentelemetry.semconv.attributes.otel_attributes.OtelStatusCodeValues`."
)
class OtelStatusCodeValues(Enum):
    OK = "OK"
    """The operation has been validated by an Application developer or Operator to have completed successfully."""
    ERROR = "ERROR"
    """The operation contains an error."""
