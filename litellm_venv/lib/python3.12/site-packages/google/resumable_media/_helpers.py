# Copyright 2017 Google Inc.
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

"""Shared utilities used by both downloads and uploads."""

from __future__ import absolute_import

import base64
import hashlib
import logging
import random
import warnings

from urllib.parse import parse_qs
from urllib.parse import urlencode
from urllib.parse import urlsplit
from urllib.parse import urlunsplit

from google.resumable_media import common


RANGE_HEADER = "range"
CONTENT_RANGE_HEADER = "content-range"
CONTENT_ENCODING_HEADER = "content-encoding"

_SLOW_CRC32C_WARNING = (
    "Currently using crcmod in pure python form. This is a slow "
    "implementation. Python 3 has a faster implementation, `google-crc32c`, "
    "which will be used if it is installed."
)
_GENERATION_HEADER = "x-goog-generation"
_HASH_HEADER = "x-goog-hash"
_STORED_CONTENT_ENCODING_HEADER = "x-goog-stored-content-encoding"

_MISSING_CHECKSUM = """\
No {checksum_type} checksum was returned from the service while downloading {}
(which happens for composite objects), so client-side content integrity
checking is not being performed."""
_LOGGER = logging.getLogger(__name__)


def do_nothing():
    """Simple default callback."""


def header_required(response, name, get_headers, callback=do_nothing):
    """Checks that a specific header is in a headers dictionary.

    Args:
        response (object): An HTTP response object, expected to have a
            ``headers`` attribute that is a ``Mapping[str, str]``.
        name (str): The name of a required header.
        get_headers (Callable[Any, Mapping[str, str]]): Helper to get headers
            from an HTTP response.
        callback (Optional[Callable]): A callback that takes no arguments,
            to be executed when an exception is being raised.

    Returns:
        str: The desired header.

    Raises:
        ~google.resumable_media.common.InvalidResponse: If the header
            is missing.
    """
    headers = get_headers(response)
    if name not in headers:
        callback()
        raise common.InvalidResponse(
            response, "Response headers must contain header", name
        )

    return headers[name]


def require_status_code(response, status_codes, get_status_code, callback=do_nothing):
    """Require a response has a status code among a list.

    Args:
        response (object): The HTTP response object.
        status_codes (tuple): The acceptable status codes.
        get_status_code (Callable[Any, int]): Helper to get a status code
            from a response.
        callback (Optional[Callable]): A callback that takes no arguments,
            to be executed when an exception is being raised.

    Returns:
        int: The status code.

    Raises:
        ~google.resumable_media.common.InvalidResponse: If the status code
            is not one of the values in ``status_codes``.
    """
    status_code = get_status_code(response)
    if status_code not in status_codes:
        if status_code not in common.RETRYABLE:
            callback()
        raise common.InvalidResponse(
            response,
            "Request failed with status code",
            status_code,
            "Expected one of",
            *status_codes
        )
    return status_code


def calculate_retry_wait(base_wait, max_sleep, multiplier=2.0):
    """Calculate the amount of time to wait before a retry attempt.

    Wait time grows exponentially with the number of attempts, until
    ``max_sleep``.

    A random amount of jitter (between 0 and 1 seconds) is added to spread out
    retry attempts from different clients.

    Args:
        base_wait (float): The "base" wait time (i.e. without any jitter)
            that will be multiplied until it reaches the maximum sleep.
        max_sleep (float): Maximum value that a sleep time is allowed to be.
        multiplier (float): Multiplier to apply to the base wait.

    Returns:
        Tuple[float, float]: The new base wait time as well as the wait time
        to be applied (with a random amount of jitter between 0 and 1 seconds
        added).
    """
    new_base_wait = multiplier * base_wait
    if new_base_wait > max_sleep:
        new_base_wait = max_sleep

    jitter_ms = random.randint(0, 1000)
    return new_base_wait, new_base_wait + 0.001 * jitter_ms


def _get_crc32c_object():
    """Get crc32c object
    Attempt to use the Google-CRC32c package. If it isn't available, try
    to use CRCMod. CRCMod might be using a 'slow' varietal. If so, warn...
    """
    try:
        import google_crc32c  # type: ignore

        crc_obj = google_crc32c.Checksum()
    except ImportError:
        try:
            import crcmod  # type: ignore

            crc_obj = crcmod.predefined.Crc("crc-32c")
            _is_fast_crcmod()

        except ImportError:
            raise ImportError("Failed to import either `google-crc32c` or `crcmod`")

    return crc_obj


def _is_fast_crcmod():
    # Determine if this is using the slow form of crcmod.
    nested_crcmod = __import__(
        "crcmod.crcmod",
        globals(),
        locals(),
        ["_usingExtension"],
        0,
    )
    fast_crc = getattr(nested_crcmod, "_usingExtension", False)
    if not fast_crc:
        warnings.warn(_SLOW_CRC32C_WARNING, RuntimeWarning, stacklevel=2)
    return fast_crc


def _get_metadata_key(checksum_type):
    if checksum_type == "md5":
        return "md5Hash"
    else:
        return checksum_type


def prepare_checksum_digest(digest_bytestring):
    """Convert a checksum object into a digest encoded for an HTTP header.

    Args:
        bytes: A checksum digest bytestring.

    Returns:
        str: A base64 string representation of the input.
    """
    encoded_digest = base64.b64encode(digest_bytestring)
    # NOTE: ``b64encode`` returns ``bytes``, but HTTP headers expect ``str``.
    return encoded_digest.decode("utf-8")


def _get_expected_checksum(response, get_headers, media_url, checksum_type):
    """Get the expected checksum and checksum object for the download response.

    Args:
        response (~requests.Response): The HTTP response object.
        get_headers (callable: response->dict): returns response headers.
        media_url (str): The URL containing the media to be downloaded.
        checksum_type Optional(str): The checksum type to read from the headers,
            exactly as it will appear in the headers (case-sensitive). Must be
            "md5", "crc32c" or None.

    Returns:
        Tuple (Optional[str], object): The expected checksum of the response,
        if it can be detected from the ``X-Goog-Hash`` header, and the
        appropriate checksum object for the expected checksum.
    """
    if checksum_type not in ["md5", "crc32c", None]:
        raise ValueError("checksum must be ``'md5'``, ``'crc32c'`` or ``None``")
    elif checksum_type in ["md5", "crc32c"]:
        headers = get_headers(response)
        expected_checksum = _parse_checksum_header(
            headers.get(_HASH_HEADER), response, checksum_label=checksum_type
        )

        if expected_checksum is None:
            msg = _MISSING_CHECKSUM.format(
                media_url, checksum_type=checksum_type.upper()
            )
            _LOGGER.info(msg)
            checksum_object = _DoNothingHash()
        else:
            if checksum_type == "md5":
                checksum_object = hashlib.md5()
            else:
                checksum_object = _get_crc32c_object()
    else:
        expected_checksum = None
        checksum_object = _DoNothingHash()

    return (expected_checksum, checksum_object)


def _get_uploaded_checksum_from_headers(response, get_headers, checksum_type):
    """Get the computed checksum and checksum object from the response headers.

    Args:
        response (~requests.Response): The HTTP response object.
        get_headers (callable: response->dict): returns response headers.
        checksum_type Optional(str): The checksum type to read from the headers,
            exactly as it will appear in the headers (case-sensitive). Must be
            "md5", "crc32c" or None.

    Returns:
        Tuple (Optional[str], object): The checksum of the response,
        if it can be detected from the ``X-Goog-Hash`` header, and the
        appropriate checksum object for the expected checksum.
    """
    if checksum_type not in ["md5", "crc32c", None]:
        raise ValueError("checksum must be ``'md5'``, ``'crc32c'`` or ``None``")
    elif checksum_type in ["md5", "crc32c"]:
        headers = get_headers(response)
        remote_checksum = _parse_checksum_header(
            headers.get(_HASH_HEADER), response, checksum_label=checksum_type
        )
    else:
        remote_checksum = None

    return remote_checksum


def _parse_checksum_header(header_value, response, checksum_label):
    """Parses the checksum header from an ``X-Goog-Hash`` value.

    .. _header reference: https://cloud.google.com/storage/docs/\
                          xml-api/reference-headers#xgooghash

    Expects ``header_value`` (if not :data:`None`) to be in one of the three
    following formats:

    * ``crc32c=n03x6A==``
    * ``md5=Ojk9c3dhfxgoKVVHYwFbHQ==``
    * ``crc32c=n03x6A==,md5=Ojk9c3dhfxgoKVVHYwFbHQ==``

    See the `header reference`_ for more information.

    Args:
        header_value (Optional[str]): The ``X-Goog-Hash`` header from
            a download response.
        response (~requests.Response): The HTTP response object.
        checksum_label (str): The label of the header value to read, as in the
            examples above. Typically "md5" or "crc32c"

    Returns:
        Optional[str]: The expected checksum of the response, if it
        can be detected from the ``X-Goog-Hash`` header; otherwise, None.

    Raises:
        ~google.resumable_media.common.InvalidResponse: If there are
            multiple checksums of the requested type in ``header_value``.
    """
    if header_value is None:
        return None

    matches = []
    for checksum in header_value.split(","):
        name, value = checksum.split("=", 1)
        # Official docs say "," is the separator, but real-world responses have encountered ", "
        if name.lstrip() == checksum_label:
            matches.append(value)

    if len(matches) == 0:
        return None
    elif len(matches) == 1:
        return matches[0]
    else:
        raise common.InvalidResponse(
            response,
            "X-Goog-Hash header had multiple ``{}`` values.".format(checksum_label),
            header_value,
            matches,
        )


def _get_checksum_object(checksum_type):
    """Respond with a checksum object for a supported type, if not None.

    Raises ValueError if checksum_type is unsupported.
    """
    if checksum_type == "md5":
        return hashlib.md5()
    elif checksum_type == "crc32c":
        return _get_crc32c_object()
    elif checksum_type is None:
        return None
    else:
        raise ValueError("checksum must be ``'md5'``, ``'crc32c'`` or ``None``")


def _parse_generation_header(response, get_headers):
    """Parses the generation header from an ``X-Goog-Generation`` value.

    Args:
        response (~requests.Response): The HTTP response object.
        get_headers (callable: response->dict): returns response headers.

    Returns:
        Optional[long]: The object generation from the response, if it
        can be detected from the ``X-Goog-Generation`` header; otherwise, None.
    """
    headers = get_headers(response)
    object_generation = headers.get(_GENERATION_HEADER, None)

    if object_generation is None:
        return None
    else:
        return int(object_generation)


def _get_generation_from_url(media_url):
    """Retrieve the object generation query param specified in the media url.

    Args:
        media_url (str): The URL containing the media to be downloaded.

    Returns:
        long: The object generation from the media url if exists; otherwise, None.
    """

    _, _, _, query, _ = urlsplit(media_url)
    query_params = parse_qs(query)
    object_generation = query_params.get("generation", None)

    if object_generation is None:
        return None
    else:
        return int(object_generation[0])


def add_query_parameters(media_url, query_params):
    """Add query parameters to a base url.

    Args:
        media_url (str): The URL containing the media to be downloaded.
        query_params (dict): Names and values of the query parameters to add.

    Returns:
        str: URL with additional query strings appended.
    """

    if len(query_params) == 0:
        return media_url

    scheme, netloc, path, query, frag = urlsplit(media_url)
    params = parse_qs(query)
    new_params = {**params, **query_params}
    query = urlencode(new_params, doseq=True)
    return urlunsplit((scheme, netloc, path, query, frag))


def _is_decompressive_transcoding(response, get_headers):
    """Returns True if the object was served decompressed. This happens when the
    "x-goog-stored-content-encoding" header is "gzip" and "content-encoding" header
    is not "gzip". See more at: https://cloud.google.com/storage/docs/transcoding#transcoding_and_gzip
    Args:
        response (~requests.Response): The HTTP response object.
        get_headers (callable: response->dict): returns response headers.
    Returns:
        bool: Returns True if decompressive transcoding has occurred; otherwise, False.
    """
    headers = get_headers(response)
    return (
        headers.get(_STORED_CONTENT_ENCODING_HEADER) == "gzip"
        and headers.get(CONTENT_ENCODING_HEADER) != "gzip"
    )


class _DoNothingHash(object):
    """Do-nothing hash object.

    Intended as a stand-in for ``hashlib.md5`` or a crc32c checksum
    implementation in cases where it isn't necessary to compute the hash.
    """

    def update(self, unused_chunk):
        """Do-nothing ``update`` method.

        Intended to match the interface of ``hashlib.md5`` and other checksums.

        Args:
            unused_chunk (bytes): A chunk of data.
        """
