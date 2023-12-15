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

import collections.abc

from google.protobuf import struct_pb2

from proto.marshal.collections import maps
from proto.marshal.collections import repeated


class ValueRule:
    """A rule to marshal between google.protobuf.Value and Python values."""

    def __init__(self, *, marshal):
        self._marshal = marshal

    def to_python(self, value, *, absent: bool = None):
        """Coerce the given value to the appropriate Python type.

        Note that both NullValue and absent fields return None.
        In order to disambiguate between these two options,
        use containment check,
        E.g.
        "value" in foo
        which is True for NullValue and False for an absent value.
        """
        kind = value.WhichOneof("kind")
        if kind == "null_value" or absent:
            return None
        if kind == "bool_value":
            return bool(value.bool_value)
        if kind == "number_value":
            return float(value.number_value)
        if kind == "string_value":
            return str(value.string_value)
        if kind == "struct_value":
            return self._marshal.to_python(
                struct_pb2.Struct,
                value.struct_value,
                absent=False,
            )
        if kind == "list_value":
            return self._marshal.to_python(
                struct_pb2.ListValue,
                value.list_value,
                absent=False,
            )
        # If more variants are ever added, we want to fail loudly
        # instead of tacitly returning None.
        raise ValueError("Unexpected kind: %s" % kind)  # pragma: NO COVER

    def to_proto(self, value) -> struct_pb2.Value:
        """Return a protobuf Value object representing this value."""
        if isinstance(value, struct_pb2.Value):
            return value
        if value is None:
            return struct_pb2.Value(null_value=0)
        if isinstance(value, bool):
            return struct_pb2.Value(bool_value=value)
        if isinstance(value, (int, float)):
            return struct_pb2.Value(number_value=float(value))
        if isinstance(value, str):
            return struct_pb2.Value(string_value=value)
        if isinstance(value, collections.abc.Sequence):
            return struct_pb2.Value(
                list_value=self._marshal.to_proto(struct_pb2.ListValue, value),
            )
        if isinstance(value, collections.abc.Mapping):
            return struct_pb2.Value(
                struct_value=self._marshal.to_proto(struct_pb2.Struct, value),
            )
        raise ValueError("Unable to coerce value: %r" % value)


class ListValueRule:
    """A rule translating google.protobuf.ListValue and list-like objects."""

    def __init__(self, *, marshal):
        self._marshal = marshal

    def to_python(self, value, *, absent: bool = None):
        """Coerce the given value to a Python sequence."""
        return (
            None
            if absent
            else repeated.RepeatedComposite(value.values, marshal=self._marshal)
        )

    def to_proto(self, value) -> struct_pb2.ListValue:
        # We got a proto, or else something we sent originally.
        # Preserve the instance we have.
        if isinstance(value, struct_pb2.ListValue):
            return value
        if isinstance(value, repeated.RepeatedComposite):
            return struct_pb2.ListValue(values=[v for v in value.pb])

        # We got a list (or something list-like); convert it.
        return struct_pb2.ListValue(
            values=[self._marshal.to_proto(struct_pb2.Value, v) for v in value]
        )


class StructRule:
    """A rule translating google.protobuf.Struct and dict-like objects."""

    def __init__(self, *, marshal):
        self._marshal = marshal

    def to_python(self, value, *, absent: bool = None):
        """Coerce the given value to a Python mapping."""
        return (
            None if absent else maps.MapComposite(value.fields, marshal=self._marshal)
        )

    def to_proto(self, value) -> struct_pb2.Struct:
        # We got a proto, or else something we sent originally.
        # Preserve the instance we have.
        if isinstance(value, struct_pb2.Struct):
            return value
        if isinstance(value, maps.MapComposite):
            return struct_pb2.Struct(
                fields={k: v for k, v in value.pb.items()},
            )

        # We got a dict (or something dict-like); convert it.
        answer = struct_pb2.Struct(
            fields={
                k: self._marshal.to_proto(struct_pb2.Value, v) for k, v in value.items()
            }
        )
        return answer
