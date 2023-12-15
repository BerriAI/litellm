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

from .enums import Enum
from .fields import Field
from .fields import MapField
from .fields import RepeatedField
from .marshal import Marshal
from .message import Message
from .modules import define_module as module
from .primitives import ProtoType
from .version import __version__


DOUBLE = ProtoType.DOUBLE
FLOAT = ProtoType.FLOAT
INT64 = ProtoType.INT64
UINT64 = ProtoType.UINT64
INT32 = ProtoType.INT32
FIXED64 = ProtoType.FIXED64
FIXED32 = ProtoType.FIXED32
BOOL = ProtoType.BOOL
STRING = ProtoType.STRING
MESSAGE = ProtoType.MESSAGE
BYTES = ProtoType.BYTES
UINT32 = ProtoType.UINT32
ENUM = ProtoType.ENUM
SFIXED32 = ProtoType.SFIXED32
SFIXED64 = ProtoType.SFIXED64
SINT32 = ProtoType.SINT32
SINT64 = ProtoType.SINT64


__all__ = (
    "__version__",
    "Enum",
    "Field",
    "MapField",
    "RepeatedField",
    "Marshal",
    "Message",
    "module",
    # Expose the types directly.
    "DOUBLE",
    "FLOAT",
    "INT64",
    "UINT64",
    "INT32",
    "FIXED64",
    "FIXED32",
    "BOOL",
    "STRING",
    "MESSAGE",
    "BYTES",
    "UINT32",
    "ENUM",
    "SFIXED32",
    "SFIXED64",
    "SINT32",
    "SINT64",
)
