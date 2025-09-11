# Copyright 2017 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Types used in the Google BigQuery DB-API.

See `PEP-249`_ for details.

.. _PEP-249:
    https://www.python.org/dev/peps/pep-0249/#type-objects-and-constructors
"""

import datetime


Date = datetime.date
Time = datetime.time
Timestamp = datetime.datetime
DateFromTicks = datetime.date.fromtimestamp
TimestampFromTicks = datetime.datetime.fromtimestamp


def Binary(data):
    """Contruct a DB-API binary value.

    Args:
        data (bytes-like): An object containing binary data and that
                           can be converted to bytes with the `bytes` builtin.

    Returns:
        bytes: The binary data as a bytes object.
    """
    if isinstance(data, int):
        # This is not the conversion we're looking for, because it
        # will simply create a bytes object of the given size.
        raise TypeError("cannot convert `int` object to binary")

    try:
        return bytes(data)
    except TypeError:
        if isinstance(data, str):
            return data.encode("utf-8")
        else:
            raise


def TimeFromTicks(ticks, tz=None):
    """Construct a DB-API time value from the given ticks value.

    Args:
        ticks (float):
            a number of seconds since the epoch; see the documentation of the
            standard Python time module for details.

        tz (datetime.tzinfo): (Optional) time zone to use for conversion

    Returns:
        datetime.time: time represented by ticks.
    """
    dt = datetime.datetime.fromtimestamp(ticks, tz=tz)
    return dt.timetz()


class _DBAPITypeObject(object):
    """DB-API type object which compares equal to many different strings.

    See `PEP-249`_ for details.

    .. _PEP-249:
        https://www.python.org/dev/peps/pep-0249/#implementation-hints-for-module-authors
    """

    def __init__(self, *values):
        self.values = values

    def __eq__(self, other):
        return other in self.values


STRING = "STRING"
BINARY = _DBAPITypeObject("BYTES", "RECORD", "STRUCT")
NUMBER = _DBAPITypeObject(
    "INTEGER", "INT64", "FLOAT", "FLOAT64", "NUMERIC", "BIGNUMERIC", "BOOLEAN", "BOOL"
)
DATETIME = _DBAPITypeObject("TIMESTAMP", "DATE", "TIME", "DATETIME")
ROWID = "ROWID"
