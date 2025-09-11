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
#
# The following imports are for backwards compatibility with https://github.com/googleapis/python-api-core/blob/4d7d2edee2c108d43deb151e6e0fdceb56b73275/google/api_core/retry_async.py
#
# TODO: Revert these imports on the next major version release (https://github.com/googleapis/python-api-core/issues/576)
from google.api_core import datetime_helpers  # noqa: F401
from google.api_core import exceptions  # noqa: F401
from google.api_core.retry import exponential_sleep_generator  # noqa: F401
from google.api_core.retry import if_exception_type  # noqa: F401
from google.api_core.retry import if_transient_error  # noqa: F401
from google.api_core.retry.retry_unary_async import AsyncRetry
from google.api_core.retry.retry_unary_async import retry_target

__all__ = (
    "AsyncRetry",
    "datetime_helpers",
    "exceptions",
    "exponential_sleep_generator",
    "if_exception_type",
    "if_transient_error",
    "retry_target",
)
