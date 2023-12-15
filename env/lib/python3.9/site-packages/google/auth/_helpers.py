# Copyright 2015 Google Inc.
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

"""Helper functions for commonly used utilities."""

import base64
import calendar
import datetime
from email.message import Message
import sys
import urllib

from google.auth import exceptions

# The smallest MDS cache used by this library stores tokens until 4 minutes from
# expiry.
REFRESH_THRESHOLD = datetime.timedelta(minutes=3, seconds=45)


def copy_docstring(source_class):
    """Decorator that copies a method's docstring from another class.

    Args:
        source_class (type): The class that has the documented method.

    Returns:
        Callable: A decorator that will copy the docstring of the same
            named method in the source class to the decorated method.
    """

    def decorator(method):
        """Decorator implementation.

        Args:
            method (Callable): The method to copy the docstring to.

        Returns:
            Callable: the same method passed in with an updated docstring.

        Raises:
            google.auth.exceptions.InvalidOperation: if the method already has a docstring.
        """
        if method.__doc__:
            raise exceptions.InvalidOperation("Method already has a docstring.")

        source_method = getattr(source_class, method.__name__)
        method.__doc__ = source_method.__doc__

        return method

    return decorator


def parse_content_type(header_value):
    """Parse a 'content-type' header value to get just the plain media-type (without parameters).

    This is done using the class Message from email.message as suggested in PEP 594
        (because the cgi is now deprecated and will be removed in python 3.13,
        see https://peps.python.org/pep-0594/#cgi).

    Args:
        header_value (str): The value of a 'content-type' header as a string.

    Returns:
        str: A string with just the lowercase media-type from the parsed 'content-type' header.
            If the provided content-type is not parsable, returns 'text/plain',
            the default value for textual files.
    """
    m = Message()
    m["content-type"] = header_value
    return (
        m.get_content_type()
    )  # Despite the name, actually returns just the media-type


def utcnow():
    """Returns the current UTC datetime.

    Returns:
        datetime: The current time in UTC.
    """
    # We used datetime.utcnow() before, since it's deprecated from python 3.12,
    # we are using datetime.now(timezone.utc) now. "utcnow()" is offset-native
    # (no timezone info), but "now()" is offset-aware (with timezone info).
    # This will cause datetime comparison problem. For backward compatibility,
    # we need to remove the timezone info.
    now = datetime.datetime.now(datetime.timezone.utc)
    now = now.replace(tzinfo=None)
    return now


def datetime_to_secs(value):
    """Convert a datetime object to the number of seconds since the UNIX epoch.

    Args:
        value (datetime): The datetime to convert.

    Returns:
        int: The number of seconds since the UNIX epoch.
    """
    return calendar.timegm(value.utctimetuple())


def to_bytes(value, encoding="utf-8"):
    """Converts a string value to bytes, if necessary.

    Args:
        value (Union[str, bytes]): The value to be converted.
        encoding (str): The encoding to use to convert unicode to bytes.
            Defaults to "utf-8".

    Returns:
        bytes: The original value converted to bytes (if unicode) or as
            passed in if it started out as bytes.

    Raises:
        google.auth.exceptions.InvalidValue: If the value could not be converted to bytes.
    """
    result = value.encode(encoding) if isinstance(value, str) else value
    if isinstance(result, bytes):
        return result
    else:
        raise exceptions.InvalidValue(
            "{0!r} could not be converted to bytes".format(value)
        )


def from_bytes(value):
    """Converts bytes to a string value, if necessary.

    Args:
        value (Union[str, bytes]): The value to be converted.

    Returns:
        str: The original value converted to unicode (if bytes) or as passed in
            if it started out as unicode.

    Raises:
        google.auth.exceptions.InvalidValue: If the value could not be converted to unicode.
    """
    result = value.decode("utf-8") if isinstance(value, bytes) else value
    if isinstance(result, str):
        return result
    else:
        raise exceptions.InvalidValue(
            "{0!r} could not be converted to unicode".format(value)
        )


def update_query(url, params, remove=None):
    """Updates a URL's query parameters.

    Replaces any current values if they are already present in the URL.

    Args:
        url (str): The URL to update.
        params (Mapping[str, str]): A mapping of query parameter
            keys to values.
        remove (Sequence[str]): Parameters to remove from the query string.

    Returns:
        str: The URL with updated query parameters.

    Examples:

        >>> url = 'http://example.com?a=1'
        >>> update_query(url, {'a': '2'})
        http://example.com?a=2
        >>> update_query(url, {'b': '3'})
        http://example.com?a=1&b=3
        >> update_query(url, {'b': '3'}, remove=['a'])
        http://example.com?b=3

    """
    if remove is None:
        remove = []

    # Split the URL into parts.
    parts = urllib.parse.urlparse(url)
    # Parse the query string.
    query_params = urllib.parse.parse_qs(parts.query)
    # Update the query parameters with the new parameters.
    query_params.update(params)
    # Remove any values specified in remove.
    query_params = {
        key: value for key, value in query_params.items() if key not in remove
    }
    # Re-encoded the query string.
    new_query = urllib.parse.urlencode(query_params, doseq=True)
    # Unsplit the url.
    new_parts = parts._replace(query=new_query)
    return urllib.parse.urlunparse(new_parts)


def scopes_to_string(scopes):
    """Converts scope value to a string suitable for sending to OAuth 2.0
    authorization servers.

    Args:
        scopes (Sequence[str]): The sequence of scopes to convert.

    Returns:
        str: The scopes formatted as a single string.
    """
    return " ".join(scopes)


def string_to_scopes(scopes):
    """Converts stringifed scopes value to a list.

    Args:
        scopes (Union[Sequence, str]): The string of space-separated scopes
            to convert.
    Returns:
        Sequence(str): The separated scopes.
    """
    if not scopes:
        return []

    return scopes.split(" ")


def padded_urlsafe_b64decode(value):
    """Decodes base64 strings lacking padding characters.

    Google infrastructure tends to omit the base64 padding characters.

    Args:
        value (Union[str, bytes]): The encoded value.

    Returns:
        bytes: The decoded value
    """
    b64string = to_bytes(value)
    padded = b64string + b"=" * (-len(b64string) % 4)
    return base64.urlsafe_b64decode(padded)


def unpadded_urlsafe_b64encode(value):
    """Encodes base64 strings removing any padding characters.

    `rfc 7515`_ defines Base64url to NOT include any padding
    characters, but the stdlib doesn't do that by default.

    _rfc7515: https://tools.ietf.org/html/rfc7515#page-6

    Args:
        value (Union[str|bytes]): The bytes-like value to encode

    Returns:
        Union[str|bytes]: The encoded value
    """
    return base64.urlsafe_b64encode(value).rstrip(b"=")


def is_python_3():
    """Check if the Python interpreter is Python 2 or 3.

    Returns:
        bool: True if the Python interpreter is Python 3 and False otherwise.
    """
    return sys.version_info > (3, 0)
