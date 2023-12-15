# Copyright (C) 2021  Google LLC
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

from proto.primitives import ProtoType


class StringyNumberRule:
    """A marshal between certain numeric types and strings

    This is a necessary hack to allow round trip conversion
    from messages to dicts back to messages.

    See https://github.com/protocolbuffers/protobuf/issues/2679
    and
    https://developers.google.com/protocol-buffers/docs/proto3#json
    for more details.
    """

    def to_python(self, value, *, absent: bool = None):
        return value

    def to_proto(self, value):
        if value is not None:
            return self._python_type(value)

        return None


class Int64Rule(StringyNumberRule):
    _python_type = int
    _proto_type = ProtoType.INT64


class UInt64Rule(StringyNumberRule):
    _python_type = int
    _proto_type = ProtoType.UINT64


class SInt64Rule(StringyNumberRule):
    _python_type = int
    _proto_type = ProtoType.SINT64


class Fixed64Rule(StringyNumberRule):
    _python_type = int
    _proto_type = ProtoType.FIXED64


class SFixed64Rule(StringyNumberRule):
    _python_type = int
    _proto_type = ProtoType.SFIXED64


STRINGY_NUMBER_RULES = [
    Int64Rule,
    UInt64Rule,
    SInt64Rule,
    Fixed64Rule,
    SFixed64Rule,
]
