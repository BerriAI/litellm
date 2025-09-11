# -*- coding: utf-8 -*-
# Copyright 2023 Google LLC
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
from __future__ import annotations

import datetime

from google.generativeai.client import get_default_file_client

import google.ai.generativelanguage as glm


class File:
    def __init__(self, proto: glm.File | File | dict):
        if isinstance(proto, File):
            proto = proto.to_proto()
        self._proto = glm.File(proto)

    def to_proto(self):
        return self._proto

    @property
    def name(self) -> str:
        return self._proto.name

    @property
    def display_name(self) -> str:
        return self._proto.display_name

    @property
    def mime_type(self) -> str:
        return self._proto.mime_type

    @property
    def size_bytes(self) -> int:
        return self._proto.size_bytes

    @property
    def create_time(self) -> datetime.datetime:
        return self._proto.create_time

    @property
    def update_time(self) -> datetime.datetime:
        return self._proto.update_time

    @property
    def expiration_time(self) -> datetime.datetime:
        return self._proto.expiration_time

    @property
    def update_time(self) -> datetime.datetime:
        return self._proto.update_time

    @property
    def sha256_hash(self) -> bytes:
        return self._proto.sha256_hash

    @property
    def uri(self) -> str:
        return self._proto.uri

    def delete(self):
        client = get_default_file_client()
        client.delete_file(name=self.name)
