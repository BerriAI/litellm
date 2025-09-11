#!/usr/bin/python
#
# Copyright The OpenTelemetry Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys

if sys.version_info.major <= 2:
    import urllib

    url_quote_fn = urllib.quote
else:
    import urllib.parse

    url_quote_fn = urllib.parse.quote

KEY_VALUE_DELIMITER = ","


def generate_sql_comment(**meta):
    """
    Return a SQL comment with comma delimited key=value pairs created from
    **meta kwargs.
    """
    if not meta:  # No entries added.
        return ""

    # Sort the keywords to ensure that caching works and that testing is
    # deterministic. It eases visual inspection as well.
    return (
        " /*"
        + KEY_VALUE_DELIMITER.join(
            "{}={!r}".format(url_quote(key), url_quote(value))
            for key, value in sorted(meta.items())
            if value is not None
        )
        + "*/"
    )


def url_quote(s):
    if not isinstance(s, (str, bytes)):
        return s
    quoted = url_quote_fn(s)
    # Since SQL uses '%' as a keyword, '%' is a by-product of url quoting
    # e.g. foo,bar --> foo%2Cbar
    # thus in our quoting, we need to escape it too to finally give
    #      foo,bar --> foo%%2Cbar
    return quoted.replace("%", "%%")
