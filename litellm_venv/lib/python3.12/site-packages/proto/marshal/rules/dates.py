# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from datetime import datetime
from datetime import timedelta
from datetime import timezone

from google.protobuf import duration_pb2
from google.protobuf import timestamp_pb2
from proto import datetime_helpers, utils


class TimestampRule:
    """A marshal between Python datetimes and protobuf timestamps.

    Note: Python datetimes are less precise than protobuf datetimes
    (microsecond vs. nanosecond level precision). If nanosecond-level
    precision matters, it is recommended to interact with the internal
    proto directly.
    """

    def to_python(
        self, value, *, absent: bool = None
    ) -> datetime_helpers.DatetimeWithNanoseconds:
        if isinstance(value, timestamp_pb2.Timestamp):
            if absent:
                return None
            return datetime_helpers.DatetimeWithNanoseconds.from_timestamp_pb(value)
        return value

    def to_proto(self, value) -> timestamp_pb2.Timestamp:
        if isinstance(value, datetime_helpers.DatetimeWithNanoseconds):
            return value.timestamp_pb()
        if isinstance(value, datetime):
            return timestamp_pb2.Timestamp(
                seconds=int(value.timestamp()),
                nanos=value.microsecond * 1000,
            )
        if isinstance(value, str):
            timestamp_value = timestamp_pb2.Timestamp()
            timestamp_value.FromJsonString(value=value)
            return timestamp_value
        return value


class DurationRule:
    """A marshal between Python timedeltas and protobuf durations.

    Note: Python timedeltas are less precise than protobuf durations
    (microsecond vs. nanosecond level precision). If nanosecond-level
    precision matters, it is recommended to interact with the internal
    proto directly.
    """

    def to_python(self, value, *, absent: bool = None) -> timedelta:
        if isinstance(value, duration_pb2.Duration):
            return timedelta(
                days=value.seconds // 86400,
                seconds=value.seconds % 86400,
                microseconds=value.nanos // 1000,
            )
        return value

    def to_proto(self, value) -> duration_pb2.Duration:
        if isinstance(value, timedelta):
            return duration_pb2.Duration(
                seconds=value.days * 86400 + value.seconds,
                nanos=value.microseconds * 1000,
            )
        if isinstance(value, str):
            duration_value = duration_pb2.Duration()
            duration_value.FromJsonString(value=value)
            return duration_value
        return value
