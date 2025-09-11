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

"""Helpers for :mod:`datetime`."""

import calendar
import datetime
import re

from google.protobuf import timestamp_pb2


_UTC_EPOCH = datetime.datetime.fromtimestamp(0, datetime.timezone.utc)

_RFC3339_MICROS = "%Y-%m-%dT%H:%M:%S.%fZ"
_RFC3339_NO_FRACTION = "%Y-%m-%dT%H:%M:%S"
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


def _from_microseconds(value):
    """Convert timestamp in microseconds since the unix epoch to datetime.

    Args:
        value (float): The timestamp to convert, in microseconds.

    Returns:
        datetime.datetime: The datetime object equivalent to the timestamp in
            UTC.
    """
    return _UTC_EPOCH + datetime.timedelta(microseconds=value)


def _to_rfc3339(value, ignore_zone=True):
    """Convert a datetime to an RFC3339 timestamp string.

    Args:
        value (datetime.datetime):
            The datetime object to be converted to a string.
        ignore_zone (bool): If True, then the timezone (if any) of the
            datetime object is ignored and the datetime is treated as UTC.

    Returns:
        str: The RFC3339 formatted string representing the datetime.
    """
    if not ignore_zone and value.tzinfo is not None:
        # Convert to UTC and remove the time zone info.
        value = value.replace(tzinfo=None) - value.utcoffset()

    return value.strftime(_RFC3339_MICROS)


class DatetimeWithNanoseconds(datetime.datetime):
    """Track nanosecond in addition to normal datetime attrs.

    Nanosecond can be passed only as a keyword argument.
    """

    __slots__ = ("_nanosecond",)

    # pylint: disable=arguments-differ
    def __new__(cls, *args, **kw):
        nanos = kw.pop("nanosecond", 0)
        if nanos > 0:
            if "microsecond" in kw:
                raise TypeError("Specify only one of 'microsecond' or 'nanosecond'")
            kw["microsecond"] = nanos // 1000
        inst = datetime.datetime.__new__(cls, *args, **kw)
        inst._nanosecond = nanos or 0
        return inst

    # pylint: disable=arguments-differ
    def replace(self, *args, **kw):
        """Return a date with the same value, except for those parameters given
        new values by whichever keyword arguments are specified. For example,
        if d == date(2002, 12, 31), then
        d.replace(day=26) == date(2002, 12, 26).
        NOTE: nanosecond and microsecond are mutually exclusive arguments.
        """

        ms_provided = "microsecond" in kw
        ns_provided = "nanosecond" in kw
        provided_ns = kw.pop("nanosecond", 0)

        prev_nanos = self.nanosecond

        if ms_provided and ns_provided:
            raise TypeError("Specify only one of 'microsecond' or 'nanosecond'")

        if ns_provided:
            # if nanos were provided, manipulate microsecond kw arg to super
            kw["microsecond"] = provided_ns // 1000
        inst = super().replace(*args, **kw)

        if ms_provided:
            # ms were provided, nanos are invalid, build from ms
            inst._nanosecond = inst.microsecond * 1000
        elif ns_provided:
            # ns were provided, replace nanoseconds to match after calling super
            inst._nanosecond = provided_ns
        else:
            # if neither ms or ns were provided, passthru previous nanos.
            inst._nanosecond = prev_nanos

        return inst

    @property
    def nanosecond(self):
        """Read-only: nanosecond precision."""
        return self._nanosecond or self.microsecond * 1000

    def rfc3339(self):
        """Return an RFC3339-compliant timestamp.

        Returns:
            (str): Timestamp string according to RFC3339 spec.
        """
        if self._nanosecond == 0:
            return _to_rfc3339(self)
        nanos = str(self._nanosecond).rjust(9, "0").rstrip("0")
        return "{}.{}Z".format(self.strftime(_RFC3339_NO_FRACTION), nanos)

    @classmethod
    def from_rfc3339(cls, stamp):
        """Parse RFC3339-compliant timestamp, preserving nanoseconds.

        Args:
            stamp (str): RFC3339 stamp, with up to nanosecond precision

        Returns:
            :class:`DatetimeWithNanoseconds`:
                an instance matching the timestamp string

        Raises:
            ValueError: if `stamp` does not match the expected format
        """
        with_nanos = _RFC3339_NANOS.match(stamp)
        if with_nanos is None:
            raise ValueError(
                "Timestamp: {}, does not match pattern: {}".format(
                    stamp, _RFC3339_NANOS.pattern
                )
            )
        bare = datetime.datetime.strptime(
            with_nanos.group("no_fraction"), _RFC3339_NO_FRACTION
        )
        fraction = with_nanos.group("nanos")
        if fraction is None:
            nanos = 0
        else:
            scale = 9 - len(fraction)
            nanos = int(fraction) * (10**scale)
        return cls(
            bare.year,
            bare.month,
            bare.day,
            bare.hour,
            bare.minute,
            bare.second,
            nanosecond=nanos,
            tzinfo=datetime.timezone.utc,
        )

    def timestamp_pb(self):
        """Return a timestamp message.

        Returns:
            (:class:`~google.protobuf.timestamp_pb2.Timestamp`): Timestamp message
        """
        inst = (
            self
            if self.tzinfo is not None
            else self.replace(tzinfo=datetime.timezone.utc)
        )
        delta = inst - _UTC_EPOCH
        seconds = int(delta.total_seconds())
        nanos = self._nanosecond or self.microsecond * 1000
        return timestamp_pb2.Timestamp(seconds=seconds, nanos=nanos)

    @classmethod
    def from_timestamp_pb(cls, stamp):
        """Parse RFC3339-compliant timestamp, preserving nanoseconds.

        Args:
            stamp (:class:`~google.protobuf.timestamp_pb2.Timestamp`): timestamp message

        Returns:
            :class:`DatetimeWithNanoseconds`:
                an instance matching the timestamp message
        """
        microseconds = int(stamp.seconds * 1e6)
        bare = _from_microseconds(microseconds)
        return cls(
            bare.year,
            bare.month,
            bare.day,
            bare.hour,
            bare.minute,
            bare.second,
            nanosecond=stamp.nanos,
            tzinfo=datetime.timezone.utc,
        )
