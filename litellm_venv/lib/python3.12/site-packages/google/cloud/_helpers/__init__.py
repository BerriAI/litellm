# Copyright 2014 Google LLC
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

"""Shared helpers for Google Cloud packages.

This module is not part of the public API surface.
"""

from __future__ import absolute_import

import calendar
import datetime
import http.client
import os
import re
from threading import local as Local
from typing import Union

import google.auth
import google.auth.transport.requests
from google.protobuf import duration_pb2
from google.protobuf import timestamp_pb2

try:
    import grpc
    import google.auth.transport.grpc
except ImportError:  # pragma: NO COVER
    grpc = None

# `google.cloud._helpers._NOW` is deprecated
_NOW = datetime.datetime.utcnow
UTC = datetime.timezone.utc  # Singleton instance to be used throughout.
_EPOCH = datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc)

_RFC3339_MICROS = "%Y-%m-%dT%H:%M:%S.%fZ"
_RFC3339_NO_FRACTION = "%Y-%m-%dT%H:%M:%S"
_TIMEONLY_W_MICROS = "%H:%M:%S.%f"
_TIMEONLY_NO_FRACTION = "%H:%M:%S"
# datetime.strptime cannot handle nanosecond precision:  parse w/ regex
_RFC3339_NANOS = re.compile(
    r"""
    (?P<no_fraction>
        \d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}  # YYYY-MM-DDTHH:MM:SS
    )
    (                                        # Optional decimal part
     \.                                      # decimal point
     (?P<nanos>\d{1,9})                      # nanoseconds, maybe truncated
    )?
    Z                                        # Zulu
""",
    re.VERBOSE,
)
# NOTE: Catching this ImportError is a workaround for GAE not supporting the
#       "pwd" module which is imported lazily when "expanduser" is called.
_USER_ROOT: Union[str, None]
try:
    _USER_ROOT = os.path.expanduser("~")
except ImportError:  # pragma: NO COVER
    _USER_ROOT = None
_GCLOUD_CONFIG_FILE = os.path.join("gcloud", "configurations", "config_default")
_GCLOUD_CONFIG_SECTION = "core"
_GCLOUD_CONFIG_KEY = "project"


class _LocalStack(Local):
    """Manage a thread-local LIFO stack of resources.

    Intended for use in :class:`google.cloud.datastore.batch.Batch.__enter__`,
    :class:`google.cloud.storage.batch.Batch.__enter__`, etc.
    """

    def __init__(self):
        super(_LocalStack, self).__init__()
        self._stack = []

    def __iter__(self):
        """Iterate the stack in LIFO order."""
        return iter(reversed(self._stack))

    def push(self, resource):
        """Push a resource onto our stack."""
        self._stack.append(resource)

    def pop(self):
        """Pop a resource from our stack.

        :rtype: object
        :returns: the top-most resource, after removing it.
        :raises IndexError: if the stack is empty.
        """
        return self._stack.pop()

    @property
    def top(self):
        """Get the top-most resource

        :rtype: object
        :returns: the top-most item, or None if the stack is empty.
        """
        if self._stack:
            return self._stack[-1]


def _ensure_tuple_or_list(arg_name, tuple_or_list):
    """Ensures an input is a tuple or list.

    This effectively reduces the iterable types allowed to a very short
    allowlist: list and tuple.

    :type arg_name: str
    :param arg_name: Name of argument to use in error message.

    :type tuple_or_list: sequence of str
    :param tuple_or_list: Sequence to be verified.

    :rtype: list of str
    :returns: The ``tuple_or_list`` passed in cast to a ``list``.
    :raises TypeError: if the ``tuple_or_list`` is not a tuple or list.
    """
    if not isinstance(tuple_or_list, (tuple, list)):
        raise TypeError(
            "Expected %s to be a tuple or list. "
            "Received %r" % (arg_name, tuple_or_list)
        )
    return list(tuple_or_list)


def _determine_default_project(project=None):
    """Determine default project ID explicitly or implicitly as fall-back.

    See :func:`google.auth.default` for details on how the default project
    is determined.

    :type project: str
    :param project: Optional. The project name to use as default.

    :rtype: str or ``NoneType``
    :returns: Default project if it can be determined.
    """
    if project is None:
        _, project = google.auth.default()
    return project


def _millis(when):
    """Convert a zone-aware datetime to integer milliseconds.

    :type when: :class:`datetime.datetime`
    :param when: the datetime to convert

    :rtype: int
    :returns: milliseconds since epoch for ``when``
    """
    micros = _microseconds_from_datetime(when)
    return micros // 1000


def _datetime_from_microseconds(value):
    """Convert timestamp to datetime, assuming UTC.

    :type value: float
    :param value: The timestamp to convert

    :rtype: :class:`datetime.datetime`
    :returns: The datetime object created from the value.
    """
    return _EPOCH + datetime.timedelta(microseconds=value)


def _microseconds_from_datetime(value):
    """Convert non-none datetime to microseconds.

    :type value: :class:`datetime.datetime`
    :param value: The timestamp to convert.

    :rtype: int
    :returns: The timestamp, in microseconds.
    """
    if not value.tzinfo:
        value = value.replace(tzinfo=UTC)
    # Regardless of what timezone is on the value, convert it to UTC.
    value = value.astimezone(UTC)
    # Convert the datetime to a microsecond timestamp.
    return int(calendar.timegm(value.timetuple()) * 1e6) + value.microsecond


def _millis_from_datetime(value):
    """Convert non-none datetime to timestamp, assuming UTC.

    :type value: :class:`datetime.datetime`
    :param value: (Optional) the timestamp

    :rtype: int, or ``NoneType``
    :returns: the timestamp, in milliseconds, or None
    """
    if value is not None:
        return _millis(value)


def _date_from_iso8601_date(value):
    """Convert a ISO8601 date string to native datetime date

    :type value: str
    :param value: The date string to convert

    :rtype: :class:`datetime.date`
    :returns: A datetime date object created from the string

    """
    return datetime.datetime.strptime(value, "%Y-%m-%d").date()


def _time_from_iso8601_time_naive(value):
    """Convert a zoneless ISO8601 time string to naive datetime time

    :type value: str
    :param value: The time string to convert

    :rtype: :class:`datetime.time`
    :returns: A datetime time object created from the string
    :raises ValueError: if the value does not match a known format.
    """
    if len(value) == 8:  # HH:MM:SS
        fmt = _TIMEONLY_NO_FRACTION
    elif len(value) == 15:  # HH:MM:SS.micros
        fmt = _TIMEONLY_W_MICROS
    else:
        raise ValueError("Unknown time format: {}".format(value))
    return datetime.datetime.strptime(value, fmt).time()


def _rfc3339_to_datetime(dt_str):
    """Convert a microsecond-precision timestamp to a native datetime.

    :type dt_str: str
    :param dt_str: The string to convert.

    :rtype: :class:`datetime.datetime`
    :returns: The datetime object created from the string.
    """
    return datetime.datetime.strptime(dt_str, _RFC3339_MICROS).replace(tzinfo=UTC)


def _rfc3339_nanos_to_datetime(dt_str):
    """Convert a nanosecond-precision timestamp to a native datetime.

    .. note::

       Python datetimes do not support nanosecond precision;  this function
       therefore truncates such values to microseconds.

    :type dt_str: str
    :param dt_str: The string to convert.

    :rtype: :class:`datetime.datetime`
    :returns: The datetime object created from the string.
    :raises ValueError: If the timestamp does not match the RFC 3339
                        regular expression.
    """
    with_nanos = _RFC3339_NANOS.match(dt_str)
    if with_nanos is None:
        raise ValueError(
            "Timestamp: %r, does not match pattern: %r"
            % (dt_str, _RFC3339_NANOS.pattern)
        )
    bare_seconds = datetime.datetime.strptime(
        with_nanos.group("no_fraction"), _RFC3339_NO_FRACTION
    )
    fraction = with_nanos.group("nanos")
    if fraction is None:
        micros = 0
    else:
        scale = 9 - len(fraction)
        nanos = int(fraction) * (10**scale)
        micros = nanos // 1000
    return bare_seconds.replace(microsecond=micros, tzinfo=UTC)


def _datetime_to_rfc3339(value, ignore_zone=True):
    """Convert a timestamp to a string.

    :type value: :class:`datetime.datetime`
    :param value: The datetime object to be converted to a string.

    :type ignore_zone: bool
    :param ignore_zone: If True, then the timezone (if any) of the datetime
                        object is ignored.

    :rtype: str
    :returns: The string representing the datetime stamp.
    """
    if not ignore_zone and value.tzinfo is not None:
        # Convert to UTC and remove the time zone info.
        value = value.replace(tzinfo=None) - value.utcoffset()

    return value.strftime(_RFC3339_MICROS)


def _to_bytes(value, encoding="ascii"):
    """Converts a string value to bytes, if necessary.

    :type value: str / bytes or unicode
    :param value: The string/bytes value to be converted.

    :type encoding: str
    :param encoding: The encoding to use to convert unicode to bytes. Defaults
                     to "ascii", which will not allow any characters from
                     ordinals larger than 127. Other useful values are
                     "latin-1", which which will only allows byte ordinals
                     (up to 255) and "utf-8", which will encode any unicode
                     that needs to be.

    :rtype: str / bytes
    :returns: The original value converted to bytes (if unicode) or as passed
              in if it started out as bytes.
    :raises TypeError: if the value could not be converted to bytes.
    """
    result = value.encode(encoding) if isinstance(value, str) else value
    if isinstance(result, bytes):
        return result
    else:
        raise TypeError("%r could not be converted to bytes" % (value,))


def _bytes_to_unicode(value):
    """Converts bytes to a unicode value, if necessary.

    :type value: bytes
    :param value: bytes value to attempt string conversion on.

    :rtype: str
    :returns: The original value converted to unicode (if bytes) or as passed
              in if it started out as unicode.

    :raises ValueError: if the value could not be converted to unicode.
    """
    result = value.decode("utf-8") if isinstance(value, bytes) else value
    if isinstance(result, str):
        return result
    else:
        raise ValueError("%r could not be converted to unicode" % (value,))


def _from_any_pb(pb_type, any_pb):
    """Converts an Any protobuf to the specified message type

    Args:
        pb_type (type): the type of the message that any_pb stores an instance
            of.
        any_pb (google.protobuf.any_pb2.Any): the object to be converted.

    Returns:
        pb_type: An instance of the pb_type message.

    Raises:
        TypeError: if the message could not be converted.
    """
    msg = pb_type()
    if not any_pb.Unpack(msg):
        raise TypeError(
            "Could not convert {} to {}".format(
                any_pb.__class__.__name__, pb_type.__name__
            )
        )

    return msg


def _pb_timestamp_to_datetime(timestamp_pb):
    """Convert a Timestamp protobuf to a datetime object.

    :type timestamp_pb: :class:`google.protobuf.timestamp_pb2.Timestamp`
    :param timestamp_pb: A Google returned timestamp protobuf.

    :rtype: :class:`datetime.datetime`
    :returns: A UTC datetime object converted from a protobuf timestamp.
    """
    return _EPOCH + datetime.timedelta(
        seconds=timestamp_pb.seconds, microseconds=(timestamp_pb.nanos / 1000.0)
    )


def _pb_timestamp_to_rfc3339(timestamp_pb):
    """Convert a Timestamp protobuf to an RFC 3339 string.

    :type timestamp_pb: :class:`google.protobuf.timestamp_pb2.Timestamp`
    :param timestamp_pb: A Google returned timestamp protobuf.

    :rtype: str
    :returns: An RFC 3339 formatted timestamp string.
    """
    timestamp = _pb_timestamp_to_datetime(timestamp_pb)
    return _datetime_to_rfc3339(timestamp)


def _datetime_to_pb_timestamp(when):
    """Convert a datetime object to a Timestamp protobuf.

    :type when: :class:`datetime.datetime`
    :param when: the datetime to convert

    :rtype: :class:`google.protobuf.timestamp_pb2.Timestamp`
    :returns: A timestamp protobuf corresponding to the object.
    """
    ms_value = _microseconds_from_datetime(when)
    seconds, micros = divmod(ms_value, 10**6)
    nanos = micros * 10**3
    return timestamp_pb2.Timestamp(seconds=seconds, nanos=nanos)


def _timedelta_to_duration_pb(timedelta_val):
    """Convert a Python timedelta object to a duration protobuf.

    .. note::

        The Python timedelta has a granularity of microseconds while
        the protobuf duration type has a duration of nanoseconds.

    :type timedelta_val: :class:`datetime.timedelta`
    :param timedelta_val: A timedelta object.

    :rtype: :class:`google.protobuf.duration_pb2.Duration`
    :returns: A duration object equivalent to the time delta.
    """
    duration_pb = duration_pb2.Duration()
    duration_pb.FromTimedelta(timedelta_val)
    return duration_pb


def _duration_pb_to_timedelta(duration_pb):
    """Convert a duration protobuf to a Python timedelta object.

    .. note::

        The Python timedelta has a granularity of microseconds while
        the protobuf duration type has a duration of nanoseconds.

    :type duration_pb: :class:`google.protobuf.duration_pb2.Duration`
    :param duration_pb: A protobuf duration object.

    :rtype: :class:`datetime.timedelta`
    :returns: The converted timedelta object.
    """
    return datetime.timedelta(
        seconds=duration_pb.seconds, microseconds=(duration_pb.nanos / 1000.0)
    )


def _name_from_project_path(path, project, template):
    """Validate a URI path and get the leaf object's name.

    :type path: str
    :param path: URI path containing the name.

    :type project: str
    :param project: (Optional) The project associated with the request. It is
                    included for validation purposes.  If passed as None,
                    disables validation.

    :type template: str
    :param template: Template regex describing the expected form of the path.
                     The regex must have two named groups, 'project' and
                     'name'.

    :rtype: str
    :returns: Name parsed from ``path``.
    :raises ValueError: if the ``path`` is ill-formed or if the project from
                        the ``path`` does not agree with the ``project``
                        passed in.
    """
    if isinstance(template, str):
        template = re.compile(template)

    match = template.match(path)

    if not match:
        raise ValueError(
            'path "%s" did not match expected pattern "%s"' % (path, template.pattern)
        )

    if project is not None:
        found_project = match.group("project")
        if found_project != project:
            raise ValueError(
                "Project from client (%s) should agree with "
                "project from resource(%s)." % (project, found_project)
            )

    return match.group("name")


def make_secure_channel(credentials, user_agent, host, extra_options=()):
    """Makes a secure channel for an RPC service.

    Uses / depends on gRPC.

    :type credentials: :class:`google.auth.credentials.Credentials`
    :param credentials: The OAuth2 Credentials to use for creating
                        access tokens.

    :type user_agent: str
    :param user_agent: The user agent to be used with API requests.

    :type host: str
    :param host: The host for the service.

    :type extra_options: tuple
    :param extra_options: (Optional) Extra gRPC options used when creating the
                          channel.

    :rtype: :class:`grpc._channel.Channel`
    :returns: gRPC secure channel with credentials attached.
    """
    target = "%s:%d" % (host, http.client.HTTPS_PORT)
    http_request = google.auth.transport.requests.Request()

    user_agent_option = ("grpc.primary_user_agent", user_agent)
    options = (user_agent_option,) + extra_options
    return google.auth.transport.grpc.secure_authorized_channel(
        credentials, http_request, target, options=options
    )


def make_secure_stub(credentials, user_agent, stub_class, host, extra_options=()):
    """Makes a secure stub for an RPC service.

    Uses / depends on gRPC.

    :type credentials: :class:`google.auth.credentials.Credentials`
    :param credentials: The OAuth2 Credentials to use for creating
                        access tokens.

    :type user_agent: str
    :param user_agent: The user agent to be used with API requests.

    :type stub_class: type
    :param stub_class: A gRPC stub type for a given service.

    :type host: str
    :param host: The host for the service.

    :type extra_options: tuple
    :param extra_options: (Optional) Extra gRPC options passed when creating
                          the channel.

    :rtype: object, instance of ``stub_class``
    :returns: The stub object used to make gRPC requests to a given API.
    """
    channel = make_secure_channel(
        credentials, user_agent, host, extra_options=extra_options
    )
    return stub_class(channel)


def make_insecure_stub(stub_class, host, port=None):
    """Makes an insecure stub for an RPC service.

    Uses / depends on gRPC.

    :type stub_class: type
    :param stub_class: A gRPC stub type for a given service.

    :type host: str
    :param host: The host for the service. May also include the port
                 if ``port`` is unspecified.

    :type port: int
    :param port: (Optional) The port for the service.

    :rtype: object, instance of ``stub_class``
    :returns: The stub object used to make gRPC requests to a given API.
    """
    if port is None:
        target = host
    else:
        # NOTE: This assumes port != http.client.HTTPS_PORT:
        target = "%s:%d" % (host, port)
    channel = grpc.insecure_channel(target)
    return stub_class(channel)
