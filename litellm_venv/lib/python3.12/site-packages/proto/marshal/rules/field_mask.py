# Copyright 2022 Google LLC
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

from google.protobuf import field_mask_pb2


class FieldMaskRule:
    """A marshal between FieldMask and strings.

    See https://github.com/googleapis/proto-plus-python/issues/333
    and
    https://developers.google.com/protocol-buffers/docs/proto3#json
    for more details.
    """

    def to_python(self, value, *, absent: bool = None):
        return value

    def to_proto(self, value):
        if isinstance(value, str):
            field_mask_value = field_mask_pb2.FieldMask()
            field_mask_value.FromJsonString(value=value)
            return field_mask_value

        return value
