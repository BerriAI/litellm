# Copyright 2024 Google LLC
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

"""Helpers for server-side streaming in REST."""

from collections import deque
import string
from typing import Deque, Union
import types

import proto
import google.protobuf.message
from google.protobuf.json_format import Parse


class BaseResponseIterator:
    """Base Iterator over REST API responses. This class should not be used directly.

    Args:
        response_message_cls (Union[proto.Message, google.protobuf.message.Message]): A response
        class expected to be returned from an API.

    Raises:
        ValueError: If `response_message_cls` is not a subclass of `proto.Message` or `google.protobuf.message.Message`.
    """

    def __init__(
        self,
        response_message_cls: Union[proto.Message, google.protobuf.message.Message],
    ):
        self._response_message_cls = response_message_cls
        # Contains a list of JSON responses ready to be sent to user.
        self._ready_objs: Deque[str] = deque()
        # Current JSON response being built.
        self._obj = ""
        # Keeps track of the nesting level within a JSON object.
        self._level = 0
        # Keeps track whether HTTP response is currently sending values
        # inside of a string value.
        self._in_string = False
        # Whether an escape symbol "\" was encountered.
        self._escape_next = False

        self._grab = types.MethodType(self._create_grab(), self)

    def _process_chunk(self, chunk: str):
        if self._level == 0:
            if chunk[0] != "[":
                raise ValueError(
                    "Can only parse array of JSON objects, instead got %s" % chunk
                )
        for char in chunk:
            if char == "{":
                if self._level == 1:
                    # Level 1 corresponds to the outermost JSON object
                    # (i.e. the one we care about).
                    self._obj = ""
                if not self._in_string:
                    self._level += 1
                self._obj += char
            elif char == "}":
                self._obj += char
                if not self._in_string:
                    self._level -= 1
                if not self._in_string and self._level == 1:
                    self._ready_objs.append(self._obj)
            elif char == '"':
                # Helps to deal with an escaped quotes inside of a string.
                if not self._escape_next:
                    self._in_string = not self._in_string
                self._obj += char
            elif char in string.whitespace:
                if self._in_string:
                    self._obj += char
            elif char == "[":
                if self._level == 0:
                    self._level += 1
                else:
                    self._obj += char
            elif char == "]":
                if self._level == 1:
                    self._level -= 1
                else:
                    self._obj += char
            else:
                self._obj += char
            self._escape_next = not self._escape_next if char == "\\" else False

    def _create_grab(self):
        if issubclass(self._response_message_cls, proto.Message):

            def grab(this):
                return this._response_message_cls.from_json(
                    this._ready_objs.popleft(), ignore_unknown_fields=True
                )

            return grab
        elif issubclass(self._response_message_cls, google.protobuf.message.Message):

            def grab(this):
                return Parse(this._ready_objs.popleft(), this._response_message_cls())

            return grab
        else:
            raise ValueError(
                "Response message class must be a subclass of proto.Message or google.protobuf.message.Message."
            )
