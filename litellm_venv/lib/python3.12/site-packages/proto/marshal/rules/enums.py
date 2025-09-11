# Copyright 2019 Google LLC
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

from typing import Type
import enum
import warnings


class EnumRule:
    """A marshal for converting between integer values and enum values."""

    def __init__(self, enum_class: Type[enum.IntEnum]):
        self._enum = enum_class

    def to_python(self, value, *, absent: bool = None):
        if isinstance(value, int) and not isinstance(value, self._enum):
            try:
                # Coerce the int on the wire to the enum value.
                return self._enum(value)
            except ValueError:
                # Since it is possible to add values to enums, we do
                # not want to flatly error on this.
                #
                # However, it is useful to make some noise about it so
                # the user realizes that an unexpected value came along.
                warnings.warn(
                    "Unrecognized {name} enum value: {value}".format(
                        name=self._enum.__name__,
                        value=value,
                    )
                )
        return value

    def to_proto(self, value):
        # Accept enum values and coerce to the pure integer.
        # This is not strictly necessary (protocol buffers can take these
        # objects as they subclass int) but nevertheless seems like the
        # right thing to do.
        if isinstance(value, self._enum):
            return value.value

        # If a string is provided that matches an enum value, coerce it
        # to the enum value.
        if isinstance(value, str):
            return self._enum[value].value

        # We got a pure integer; pass it on.
        return value
