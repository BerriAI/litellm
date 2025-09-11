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


import base64


class BytesRule:
    """A marshal between Python strings and protobuf bytes.

    Note: this conversion is asymmetric because Python does have a bytes type.
    It is sometimes necessary to convert proto bytes fields to strings, e.g. for
    JSON encoding, marshalling a message to a dict. Because bytes fields can
    represent arbitrary data, bytes fields are base64 encoded when they need to
    be represented as strings.

    It is necessary to have the conversion be bidirectional, i.e.
    my_message == MyMessage(MyMessage.to_dict(my_message))

    To accomplish this, we need to intercept assignments from strings and
    base64 decode them back into bytes.
    """

    def to_python(self, value, *, absent: bool = None):
        return value

    def to_proto(self, value):
        if isinstance(value, str):
            value = value.encode("utf-8")
            value += b"=" * (4 - len(value) % 4)  # padding
            value = base64.urlsafe_b64decode(value)

        return value
