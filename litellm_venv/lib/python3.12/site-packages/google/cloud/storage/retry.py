# Copyright 2020 Google LLC
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

"""Helpers for configuring retries with exponential back-off.

See [Retry Strategy for Google Cloud Storage](https://cloud.google.com/storage/docs/retry-strategy#client-libraries)
"""

import requests
import requests.exceptions as requests_exceptions

from google.api_core import exceptions as api_exceptions
from google.api_core import retry
from google.auth import exceptions as auth_exceptions


_RETRYABLE_TYPES = (
    api_exceptions.TooManyRequests,  # 429
    api_exceptions.InternalServerError,  # 500
    api_exceptions.BadGateway,  # 502
    api_exceptions.ServiceUnavailable,  # 503
    api_exceptions.GatewayTimeout,  # 504
    ConnectionError,
    requests.ConnectionError,
    requests_exceptions.ChunkedEncodingError,
    requests_exceptions.Timeout,
)


# Some retriable errors don't have their own custom exception in api_core.
_ADDITIONAL_RETRYABLE_STATUS_CODES = (408,)


def _should_retry(exc):
    """Predicate for determining when to retry."""
    if isinstance(exc, _RETRYABLE_TYPES):
        return True
    elif isinstance(exc, api_exceptions.GoogleAPICallError):
        return exc.code in _ADDITIONAL_RETRYABLE_STATUS_CODES
    elif isinstance(exc, auth_exceptions.TransportError):
        return _should_retry(exc.args[0])
    else:
        return False


DEFAULT_RETRY = retry.Retry(predicate=_should_retry)
"""The default retry object.

This retry setting will retry all _RETRYABLE_TYPES and any status codes from
_ADDITIONAL_RETRYABLE_STATUS_CODES.

To modify the default retry behavior, create a new retry object modeled after
this one by calling it a ``with_XXX`` method. For example, to create a copy of
DEFAULT_RETRY with a deadline of 30 seconds, pass
``retry=DEFAULT_RETRY.with_deadline(30)``. See google-api-core reference
(https://googleapis.dev/python/google-api-core/latest/retry.html) for details.
"""


class ConditionalRetryPolicy(object):
    """A class for use when an API call is only conditionally safe to retry.

    This class is intended for use in inspecting the API call parameters of an
    API call to verify that any flags necessary to make the API call idempotent
    (such as specifying an ``if_generation_match`` or related flag) are present.

    It can be used in place of a ``retry.Retry`` object, in which case
    ``_http.Connection.api_request`` will pass the requested api call keyword
    arguments into the ``conditional_predicate`` and return the ``retry_policy``
    if the conditions are met.

    :type retry_policy: class:`google.api_core.retry.Retry`
    :param retry_policy: A retry object defining timeouts, persistence and which
        exceptions to retry.

    :type conditional_predicate: callable
    :param conditional_predicate: A callable that accepts exactly the number of
        arguments in ``required_kwargs``, in order, and returns True if the
        arguments have sufficient data to determine that the call is safe to
        retry (idempotent).

    :type required_kwargs: list(str)
    :param required_kwargs:
        A list of keyword argument keys that will be extracted from the API call
        and passed into the ``conditional predicate`` in order. For example,
        ``["query_params"]`` is commmonly used for preconditions in query_params.
    """

    def __init__(self, retry_policy, conditional_predicate, required_kwargs):
        self.retry_policy = retry_policy
        self.conditional_predicate = conditional_predicate
        self.required_kwargs = required_kwargs

    def get_retry_policy_if_conditions_met(self, **kwargs):
        if self.conditional_predicate(*[kwargs[key] for key in self.required_kwargs]):
            return self.retry_policy
        return None


def is_generation_specified(query_params):
    """Return True if generation or if_generation_match is specified."""
    generation = query_params.get("generation") is not None
    if_generation_match = query_params.get("ifGenerationMatch") is not None
    return generation or if_generation_match


def is_metageneration_specified(query_params):
    """Return True if if_metageneration_match is specified."""
    if_metageneration_match = query_params.get("ifMetagenerationMatch") is not None
    return if_metageneration_match


def is_etag_in_data(data):
    """Return True if an etag is contained in the request body.

    :type data: dict or None
    :param data: A dict representing the request JSON body. If not passed, returns False.
    """
    return data is not None and "etag" in data


def is_etag_in_json(data):
    """
    ``is_etag_in_json`` is supported for backwards-compatibility reasons only;
    please use ``is_etag_in_data`` instead.
    """
    return is_etag_in_data(data)


DEFAULT_RETRY_IF_GENERATION_SPECIFIED = ConditionalRetryPolicy(
    DEFAULT_RETRY, is_generation_specified, ["query_params"]
)
"""Conditional wrapper for the default retry object.

This retry setting will retry all _RETRYABLE_TYPES and any status codes from
_ADDITIONAL_RETRYABLE_STATUS_CODES, but only if the request included an
``ifGenerationMatch`` header.
"""

DEFAULT_RETRY_IF_METAGENERATION_SPECIFIED = ConditionalRetryPolicy(
    DEFAULT_RETRY, is_metageneration_specified, ["query_params"]
)
"""Conditional wrapper for the default retry object.

This retry setting will retry all _RETRYABLE_TYPES and any status codes from
_ADDITIONAL_RETRYABLE_STATUS_CODES, but only if the request included an
``ifMetagenerationMatch`` header.
"""

DEFAULT_RETRY_IF_ETAG_IN_JSON = ConditionalRetryPolicy(
    DEFAULT_RETRY, is_etag_in_json, ["data"]
)
"""Conditional wrapper for the default retry object.

This retry setting will retry all _RETRYABLE_TYPES and any status codes from
_ADDITIONAL_RETRYABLE_STATUS_CODES, but only if the request included an
``ETAG`` entry in its payload.
"""
