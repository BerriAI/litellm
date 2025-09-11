# -*- coding: utf-8 -*-

# Copyright 2022 Google LLC
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

from typing import Dict, Optional

try:
    import starlette
except ImportError:
    raise ImportError(
        "Starlette is not installed and is required to build model servers. "
        'Please install the SDK using `pip install "google-cloud-aiplatform[prediction]>=1.16.0"`.'
    )

from google.cloud.aiplatform.constants import prediction


def _remove_parameter(value: Optional[str]) -> Optional[str]:
    """Removes the parameter part from the header value.

    Referring to https://www.w3.org/Protocols/rfc2616/rfc2616-sec3.html#sec3.7.

    Args:
        value (str):
            Optional. The original full header value.

    Returns:
        The value without the parameter or None.
    """
    if value is None:
        return None

    return value.split(";")[0]


def get_content_type_from_headers(
    headers: Optional[starlette.datastructures.Headers],
) -> Optional[str]:
    """Gets content type from headers.

    Args:
        headers (starlette.datastructures.Headers):
            Optional. The headers that the content type is retrived from.

    Returns:
        The content type or None.
    """
    if headers is not None:
        for key, value in headers.items():
            if prediction.CONTENT_TYPE_HEADER_REGEX.match(key):
                return _remove_parameter(value)

    return None


def get_accept_from_headers(
    headers: Optional[starlette.datastructures.Headers],
) -> str:
    """Gets accept from headers.

    Default to "application/json" if it is unset.

    Args:
        headers (starlette.datastructures.Headers):
            Optional. The headers that the accept is retrived from.

    Returns:
        The accept.
    """
    if headers is not None:
        for key, value in headers.items():
            if prediction.ACCEPT_HEADER_REGEX.match(key):
                return value

    return prediction.DEFAULT_ACCEPT_VALUE


def parse_accept_header(accept_header: Optional[str]) -> Dict[str, float]:
    """Parses the accept header with quality factors.

    Referring to https://www.w3.org/Protocols/rfc2616/rfc2616-sec14.html.

    The default quality factor is 1.

    Args:
        accept_header (str):
            Optional. The accept header.

    Returns:
        A dictionary with media types pointing to the quality factors.
    """
    if not accept_header:
        return {}

    all_accepts = accept_header.split(",")
    results = {}

    for media_type in all_accepts:
        if media_type.split(";")[0] == media_type:
            # no q => q = 1
            results[media_type.strip()] = 1.0
        else:
            q = media_type.split(";")[1].split("=")[1]
            results[media_type.split(";")[0].strip()] = float(q)

    return results
