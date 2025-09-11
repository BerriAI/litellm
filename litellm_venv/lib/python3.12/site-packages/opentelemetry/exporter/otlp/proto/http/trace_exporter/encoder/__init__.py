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

import logging  # noqa: F401
from collections import abc  # noqa: F401
from typing import Any, List, Optional, Sequence  # noqa: F401

from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (  # noqa: F401
    ExportTraceServiceRequest as PB2ExportTraceServiceRequest,
)
from opentelemetry.proto.common.v1.common_pb2 import (  # noqa: F401
    AnyValue as PB2AnyValue,
)
from opentelemetry.proto.common.v1.common_pb2 import (  # noqa: F401
    ArrayValue as PB2ArrayValue,
)
from opentelemetry.proto.common.v1.common_pb2 import (  # noqa: F401
    InstrumentationScope as PB2InstrumentationScope,
)
from opentelemetry.proto.common.v1.common_pb2 import (  # noqa: F401
    KeyValue as PB2KeyValue,
)
from opentelemetry.proto.resource.v1.resource_pb2 import (  # noqa: F401
    Resource as PB2Resource,
)
from opentelemetry.proto.trace.v1.trace_pb2 import (  # noqa: F401
    ScopeSpans as PB2ScopeSpans,
)
from opentelemetry.proto.trace.v1.trace_pb2 import (  # noqa: F401
    ResourceSpans as PB2ResourceSpans,
)
from opentelemetry.proto.trace.v1.trace_pb2 import (  # noqa: F401
    Span as PB2SPan,
)
from opentelemetry.proto.trace.v1.trace_pb2 import (  # noqa: F401
    Status as PB2Status,
)
from opentelemetry.sdk.trace import Event  # noqa: F401
from opentelemetry.sdk.util.instrumentation import (  # noqa: F401
    InstrumentationScope,
)
from opentelemetry.sdk.trace import Resource  # noqa: F401
from opentelemetry.sdk.trace import Span as SDKSpan  # noqa: F401
from opentelemetry.trace import Link  # noqa: F401
from opentelemetry.trace import SpanKind  # noqa: F401
from opentelemetry.trace.span import (  # noqa: F401
    SpanContext,
    TraceState,
    Status,
)
from opentelemetry.util.types import Attributes  # noqa: F401
