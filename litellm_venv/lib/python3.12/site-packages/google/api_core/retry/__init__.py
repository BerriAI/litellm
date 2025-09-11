# Copyright 2017 Google LLC
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

"""Retry implementation for Google API client libraries."""

from .retry_base import exponential_sleep_generator
from .retry_base import if_exception_type
from .retry_base import if_transient_error
from .retry_base import build_retry_error
from .retry_base import RetryFailureReason
from .retry_unary import Retry
from .retry_unary import retry_target
from .retry_unary_async import AsyncRetry
from .retry_unary_async import retry_target as retry_target_async
from .retry_streaming import StreamingRetry
from .retry_streaming import retry_target_stream
from .retry_streaming_async import AsyncStreamingRetry
from .retry_streaming_async import retry_target_stream as retry_target_stream_async

# The following imports are for backwards compatibility with https://github.com/googleapis/python-api-core/blob/4d7d2edee2c108d43deb151e6e0fdceb56b73275/google/api_core/retry.py
#
# TODO: Revert these imports on the next major version release (https://github.com/googleapis/python-api-core/issues/576)
from google.api_core import datetime_helpers  # noqa: F401
from google.api_core import exceptions  # noqa: F401
from google.auth import exceptions as auth_exceptions  # noqa: F401

__all__ = (
    "exponential_sleep_generator",
    "if_exception_type",
    "if_transient_error",
    "build_retry_error",
    "RetryFailureReason",
    "Retry",
    "AsyncRetry",
    "StreamingRetry",
    "AsyncStreamingRetry",
    "retry_target",
    "retry_target_async",
    "retry_target_stream",
    "retry_target_stream_async",
)
