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

"""Helpers for constructing routing headers.

These headers are used by Google infrastructure to determine how to route
requests, especially for services that are regional.

Generally, these headers are specified as gRPC metadata.
"""

import functools
from enum import Enum
from urllib.parse import urlencode

ROUTING_METADATA_KEY = "x-goog-request-params"
# This is the value for the `maxsize` argument of @functools.lru_cache
# https://docs.python.org/3/library/functools.html#functools.lru_cache
# This represents the number of recent function calls to store.
ROUTING_PARAM_CACHE_SIZE = 32


def to_routing_header(params, qualified_enums=True):
    """Returns a routing header string for the given request parameters.

    Args:
        params (Mapping[str, str | bytes | Enum]): A dictionary containing the request
            parameters used for routing.
        qualified_enums (bool): Whether to represent enum values
            as their type-qualified symbol names instead of as their
            unqualified symbol names.

    Returns:
        str: The routing header string.
    """
    tuples = params.items() if isinstance(params, dict) else params
    if not qualified_enums:
        tuples = [(x[0], x[1].name) if isinstance(x[1], Enum) else x for x in tuples]
    return "&".join([_urlencode_param(*t) for t in tuples])


def to_grpc_metadata(params, qualified_enums=True):
    """Returns the gRPC metadata containing the routing headers for the given
    request parameters.

    Args:
        params (Mapping[str, str | bytes | Enum]): A dictionary containing the request
            parameters used for routing.
        qualified_enums (bool): Whether to represent enum values
            as their type-qualified symbol names instead of as their
            unqualified symbol names.

    Returns:
        Tuple(str, str): The gRPC metadata containing the routing header key
            and value.
    """
    return (ROUTING_METADATA_KEY, to_routing_header(params, qualified_enums))


# use caching to avoid repeated computation
@functools.lru_cache(maxsize=ROUTING_PARAM_CACHE_SIZE)
def _urlencode_param(key, value):
    """Cacheable wrapper over urlencode

    Args:
        key (str): The key of the parameter to encode.
        value (str | bytes | Enum): The value of the parameter to encode.

    Returns:
        str: The encoded parameter.
    """
    return urlencode(
        {key: value},
        # Per Google API policy (go/api-url-encoding), / is not encoded.
        safe="/",
    )
