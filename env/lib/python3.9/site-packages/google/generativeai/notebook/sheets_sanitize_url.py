# -*- coding: utf-8 -*-
# Copyright 2023 Google LLC
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
"""Utilities for working with URLs."""
from __future__ import annotations

import re
from urllib import parse


def _validate_url_part(part: str) -> None:
    if not re.fullmatch("[a-zA-Z0-9_-]*", part):
        raise ValueError('"{}" is outside the restricted character set'.format(part))


def _validate_url_query_or_fragment(part: str) -> None:
    for key, values in parse.parse_qs(part).items():
        _validate_url_part(key)
        for value in values:
            _validate_url_part(value)


def sanitize_sheets_url(url: str) -> str:
    """Sanitize a Sheets URL.

    Run some saftey checks to check whether `url` is a Sheets URL. This is not a
    general-purpose URL sanitizer. Rather, it makes use of the fact that we know
    the URL has to be for Sheets so we can make a few assumptions about (e.g. the
    domain).

    Args:
      url: The url to sanitize.

    Returns:
      The sanitized url.

    Raises:
      ValueError: If `url` does not match the expected restrictions for a Sheets
      URL.
    """
    parse_result = parse.urlparse(url)
    if parse_result.scheme != "https":
        raise ValueError(
            'Scheme for Sheets url must be "https", got "{}"'.format(parse_result.scheme)
        )
    if parse_result.netloc not in ("docs.google.com", "sheets.googleapis.com"):
        raise ValueError(
            'Domain for Sheets url must be "docs.google.com", got "{}"'.format(parse_result.netloc)
        )

    # Path component.
    try:
        for fragment in parse_result.path.split("/"):
            _validate_url_part(fragment)
    except ValueError as exc:
        raise ValueError('Invalid path for Sheets url, got "{}"'.format(parse_result.path)) from exc

    # Params component.
    if parse_result.params:
        raise ValueError('Params component must be empty, got "{}"'.format(parse_result.params))

    # Query component.
    try:
        _validate_url_query_or_fragment(parse_result.query)
    except ValueError as exc:
        raise ValueError(
            'Invalid query for Sheets url, got "{}"'.format(parse_result.query)
        ) from exc

    # Fragment component.
    try:
        _validate_url_query_or_fragment(parse_result.fragment)
    except ValueError as exc:
        raise ValueError(
            'Invalid fragment for Sheets url, got "{}"'.format(parse_result.fragment)
        ) from exc

    return url
