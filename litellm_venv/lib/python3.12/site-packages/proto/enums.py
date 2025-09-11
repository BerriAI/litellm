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

import enum

from google.protobuf import descriptor_pb2

from proto import _file_info
from proto import _package_info
from proto.marshal.rules.enums import EnumRule


class ProtoEnumMeta(enum.EnumMeta):
    """A metaclass for building and registering protobuf enums."""

    def __new__(mcls, name, bases, attrs):
        # Do not do any special behavior for `proto.Enum` itself.
        if bases[0] == enum.IntEnum:
            return super().__new__(mcls, name, bases, attrs)

        # Get the essential information about the proto package, and where
        # this component belongs within the file.
        package, marshal = _package_info.compile(name, attrs)

        # Determine the local path of this proto component within the file.
        local_path = tuple(attrs.get("__qualname__", name).split("."))

        # Sanity check: We get the wrong full name if a class is declared
        # inside a function local scope; correct this.
        if "<locals>" in local_path:
            ix = local_path.index("<locals>")
            local_path = local_path[: ix - 1] + local_path[ix + 1 :]

        # Determine the full name in protocol buffers.
        full_name = ".".join((package,) + local_path).lstrip(".")
        filename = _file_info._FileInfo.proto_file_name(
            attrs.get("__module__", name.lower())
        )

        # Retrieve any enum options.
        # We expect something that looks like an EnumOptions message,
        # either an actual instance or a dict-like representation.
        pb_options = "_pb_options"
        opts = attrs.pop(pb_options, {})
        # This is the only portable way to remove the _pb_options name
        # from the enum attrs.
        # In 3.7 onwards, we can define an _ignore_ attribute and do some
        # mucking around with that.
        if pb_options in attrs._member_names:
            if isinstance(attrs._member_names, list):
                idx = attrs._member_names.index(pb_options)
                attrs._member_names.pop(idx)
            elif isinstance(attrs._member_names, set):  # PyPy
                attrs._member_names.discard(pb_options)
            else:  # Python 3.11.0b3
                del attrs._member_names[pb_options]

        # Make the descriptor.
        enum_desc = descriptor_pb2.EnumDescriptorProto(
            name=name,
            # Note: the superclass ctor removes the variants, so get them now.
            # Note: proto3 requires that the first variant value be zero.
            value=sorted(
                (
                    descriptor_pb2.EnumValueDescriptorProto(name=name, number=number)
                    # Minor hack to get all the enum variants out.
                    # Use the `_member_names` property to get only the enum members
                    # See https://github.com/googleapis/proto-plus-python/issues/490
                    for name, number in attrs.items()
                    if name in attrs._member_names and isinstance(number, int)
                ),
                key=lambda v: v.number,
            ),
            options=opts,
        )

        file_info = _file_info._FileInfo.maybe_add_descriptor(filename, package)
        if len(local_path) == 1:
            file_info.descriptor.enum_type.add().MergeFrom(enum_desc)
        else:
            file_info.nested_enum[local_path] = enum_desc

        # Run the superclass constructor.
        cls = super().__new__(mcls, name, bases, attrs)

        # We can't just add a "_meta" element to attrs because the Enum
        # machinery doesn't know what to do with a non-int value.
        # The pb is set later, in generate_file_pb
        cls._meta = _EnumInfo(full_name=full_name, pb=None)

        file_info.enums[full_name] = cls

        # Register the enum with the marshal.
        marshal.register(cls, EnumRule(cls))

        # Generate the descriptor for the file if it is ready.
        if file_info.ready(new_class=cls):
            file_info.generate_file_pb(new_class=cls, fallback_salt=full_name)

        # Done; return the class.
        return cls


class Enum(enum.IntEnum, metaclass=ProtoEnumMeta):
    """A enum object that also builds a protobuf enum descriptor."""

    def _comparable(self, other):
        # Avoid 'isinstance' to prevent other IntEnums from matching
        return type(other) in (type(self), int)

    def __hash__(self):
        return hash(self.value)

    def __eq__(self, other):
        if not self._comparable(other):
            return NotImplemented

        return self.value == int(other)

    def __ne__(self, other):
        if not self._comparable(other):
            return NotImplemented

        return self.value != int(other)

    def __lt__(self, other):
        if not self._comparable(other):
            return NotImplemented

        return self.value < int(other)

    def __le__(self, other):
        if not self._comparable(other):
            return NotImplemented

        return self.value <= int(other)

    def __ge__(self, other):
        if not self._comparable(other):
            return NotImplemented

        return self.value >= int(other)

    def __gt__(self, other):
        if not self._comparable(other):
            return NotImplemented

        return self.value > int(other)


class _EnumInfo:
    def __init__(self, *, full_name: str, pb):
        self.full_name = full_name
        self.pb = pb
