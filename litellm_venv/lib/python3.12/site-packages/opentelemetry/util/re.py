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

from logging import getLogger
from re import compile, split
from typing import Dict, List, Mapping
from urllib.parse import unquote

from deprecated import deprecated

_logger = getLogger(__name__)

# The following regexes reference this spec: https://github.com/open-telemetry/opentelemetry-specification/blob/main/specification/protocol/exporter.md#specifying-headers-via-environment-variables

# Optional whitespace
_OWS = r"[ \t]*"
# A key contains printable US-ASCII characters except: SP and "(),/:;<=>?@[\]{}
_KEY_FORMAT = r"[\x21\x23-\x27\x2a\x2b\x2d\x2e\x30-\x39\x41-\x5a\x5e-\x7a\x7c\x7e]+"
# A value contains a URL-encoded UTF-8 string. The encoded form can contain any
# printable US-ASCII characters (0x20-0x7f) other than SP, DEL, and ",;/
_VALUE_FORMAT = r"[\x21\x23-\x2b\x2d-\x3a\x3c-\x5b\x5d-\x7e]*"
# A key-value is key=value, with optional whitespace surrounding key and value
_KEY_VALUE_FORMAT = rf"{_OWS}{_KEY_FORMAT}{_OWS}={_OWS}{_VALUE_FORMAT}{_OWS}"

_HEADER_PATTERN = compile(_KEY_VALUE_FORMAT)
_DELIMITER_PATTERN = compile(r"[ \t]*,[ \t]*")

_BAGGAGE_PROPERTY_FORMAT = rf"{_KEY_VALUE_FORMAT}|{_OWS}{_KEY_FORMAT}{_OWS}"


# pylint: disable=invalid-name


@deprecated(version="1.15.0", reason="You should use parse_env_headers")  # type: ignore
def parse_headers(s: str) -> Mapping[str, str]:
    return parse_env_headers(s)


def parse_env_headers(s: str) -> Mapping[str, str]:
    """
    Parse ``s``, which is a ``str`` instance containing HTTP headers encoded
    for use in ENV variables per the W3C Baggage HTTP header format at
    https://www.w3.org/TR/baggage/#baggage-http-header-format, except that
    additional semi-colon delimited metadata is not supported.
    """
    headers: Dict[str, str] = {}
    headers_list: List[str] = split(_DELIMITER_PATTERN, s)
    for header in headers_list:
        if not header:  # empty string
            continue
        match = _HEADER_PATTERN.fullmatch(header.strip())
        if not match:
            _logger.warning(
                "Header format invalid! Header values in environment variables must be "
                "URL encoded per the OpenTelemetry Protocol Exporter specification: %s",
                header,
            )
            continue
        # value may contain any number of `=`
        name, value = match.string.split("=", 1)
        name = unquote(name).strip().lower()
        value = unquote(value).strip()
        headers[name] = value

    return headers
