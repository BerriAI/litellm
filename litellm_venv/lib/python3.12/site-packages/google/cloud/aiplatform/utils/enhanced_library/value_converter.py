# Copyright 2020 Google LLC
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
from __future__ import absolute_import
from google.protobuf.struct_pb2 import Value
from google.protobuf import json_format
from proto.marshal.collections.maps import MapComposite
from proto.marshal import Marshal
from proto import Message
from proto.message import MessageMeta


def to_value(self: Message) -> Value:
    """Converts a message type to a :class:`~google.protobuf.struct_pb2.Value` object.

    Args:
      message: the message to convert

    Returns:
      the message as a :class:`~google.protobuf.struct_pb2.Value` object
    """
    tmp_dict = json_format.MessageToDict(self._pb)
    return json_format.ParseDict(tmp_dict, Value())


def from_value(cls: MessageMeta, value: Value) -> Message:
    """Creates instance of class from a :class:`~google.protobuf.struct_pb2.Value` object.

    Args:
      value: a :class:`~google.protobuf.struct_pb2.Value` object

    Returns:
      Instance of class
    """
    value_dict = json_format.MessageToDict(value)
    return json_format.ParseDict(value_dict, cls()._pb)


def from_map(cls: MessageMeta, map_: MapComposite) -> Message:
    """Creates instance of class from a :class:`~proto.marshal.collections.maps.MapComposite` object.

    Args:
      map_: a :class:`~proto.marshal.collections.maps.MapComposite` object

    Returns:
      Instance of class
    """
    marshal = Marshal(name="marshal")
    pb = marshal.to_proto(Value, map_)
    return from_value(cls, pb)
