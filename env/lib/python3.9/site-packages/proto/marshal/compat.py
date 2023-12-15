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

# Import protobuf 4.xx first and fallback to earlier version
# if not present.
try:
    from google._upb import _message
except ImportError:
    _message = None

if not _message:
    try:
        from google.protobuf.pyext import _message
    except ImportError:
        _message = None

repeated_composite_types = (containers.RepeatedCompositeFieldContainer,)
repeated_scalar_types = (containers.RepeatedScalarFieldContainer,)
map_composite_types = (containers.MessageMap,)

if _message:
    repeated_composite_types += (_message.RepeatedCompositeContainer,)
    repeated_scalar_types += (_message.RepeatedScalarContainer,)
    map_composite_types += (_message.MessageMapContainer,)


__all__ = (
    "repeated_composite_types",
    "repeated_scalar_types",
    "map_composite_types",
)
