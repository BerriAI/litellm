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

# This file pulls in the container types from internal protocol buffers,
# and exports the types available.
#
# If the C extensions were not installed, then their container types will
# not be included.

from google.protobuf.internal import containers

# Import all message types to ensure that pyext types are recognized
# when upb types exist. Conda's protobuf defaults to pyext despite upb existing.
# See https://github.com/googleapis/proto-plus-python/issues/470
try:
    from google._upb import _message as _message_upb
except ImportError:
    _message_upb = None

try:
    from google.protobuf.pyext import _message as _message_pyext
except ImportError:
    _message_pyext = None


repeated_composite_types = (containers.RepeatedCompositeFieldContainer,)
repeated_scalar_types = (containers.RepeatedScalarFieldContainer,)
map_composite_types = (containers.MessageMap,)

# In `proto/marshal.py`, for compatibility with protobuf 5.x,
# we'll use `map_composite_type_names` to check whether
# the name of the class of a protobuf type is
# `MessageMapContainer`, and, if `True`, return a MapComposite.
# See https://github.com/protocolbuffers/protobuf/issues/16596
map_composite_type_names = ("MessageMapContainer",)

for message in [_message_upb, _message_pyext]:
    if message:
        repeated_composite_types += (message.RepeatedCompositeContainer,)
        repeated_scalar_types += (message.RepeatedScalarContainer,)

        try:
            map_composite_types += (message.MessageMapContainer,)
        except AttributeError:
            # The `MessageMapContainer` attribute is not available in Protobuf 5.x+
            pass

__all__ = (
    "repeated_composite_types",
    "repeated_scalar_types",
    "map_composite_types",
    "map_composite_type_names",
)
