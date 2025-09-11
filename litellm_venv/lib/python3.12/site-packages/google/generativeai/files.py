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

import os
import pathlib
import mimetypes
from typing import Iterable
import logging
import google.ai.generativelanguage as glm
from itertools import islice

from google.generativeai.types import file_types

from google.generativeai.client import get_default_file_client

__all__ = ["upload_file", "get_file", "list_files", "delete_file"]


def upload_file(
    path: str | pathlib.Path | os.PathLike,
    *,
    mime_type: str | None = None,
    name: str | None = None,
    display_name: str | None = None,
) -> file_types.File:
    client = get_default_file_client()

    path = pathlib.Path(os.fspath(path))

    if mime_type is None:
        mime_type, _ = mimetypes.guess_type(path)

    if name is not None and "/" not in name:
        name = f"files/{name}"

    if display_name is None:
        display_name = path.name

    response = client.create_file(
        path=path, mime_type=mime_type, name=name, display_name=display_name
    )
    return file_types.File(response)


def list_files(page_size=100) -> Iterable[file_types.File]:
    client = get_default_file_client()

    response = client.list_files(glm.ListFilesRequest(page_size=page_size))
    for proto in response:
        yield file_types.File(proto)


def get_file(name) -> file_types.File:
    client = get_default_file_client()
    return file_types.File(client.get_file(name=name))


def delete_file(name):
    if isinstance(name, (file_types.File, glm.File)):
        name = name.name
    request = glm.DeleteFileRequest(name=name)
    client = get_default_file_client()
    client.delete_file(request=request)
