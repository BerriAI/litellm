# -*- coding: utf-8 -*-
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
#
from __future__ import annotations

from typing import MutableMapping, MutableSequence

import proto  # type: ignore

from google.ai.generativelanguage_v1beta.types import file as gag_file

__protobuf__ = proto.module(
    package="google.ai.generativelanguage.v1beta",
    manifest={
        "CreateFileRequest",
        "CreateFileResponse",
        "ListFilesRequest",
        "ListFilesResponse",
        "GetFileRequest",
        "DeleteFileRequest",
    },
)


class CreateFileRequest(proto.Message):
    r"""Request for ``CreateFile``.

    Attributes:
        file (google.ai.generativelanguage_v1beta.types.File):
            Optional. Metadata for the file to create.
    """

    file: gag_file.File = proto.Field(
        proto.MESSAGE,
        number=1,
        message=gag_file.File,
    )


class CreateFileResponse(proto.Message):
    r"""Response for ``CreateFile``.

    Attributes:
        file (google.ai.generativelanguage_v1beta.types.File):
            Metadata for the created file.
    """

    file: gag_file.File = proto.Field(
        proto.MESSAGE,
        number=1,
        message=gag_file.File,
    )


class ListFilesRequest(proto.Message):
    r"""Request for ``ListFiles``.

    Attributes:
        page_size (int):
            Optional. Maximum number of ``File``\ s to return per page.
            If unspecified, defaults to 10. Maximum ``page_size`` is
            100.
        page_token (str):
            Optional. A page token from a previous ``ListFiles`` call.
    """

    page_size: int = proto.Field(
        proto.INT32,
        number=1,
    )
    page_token: str = proto.Field(
        proto.STRING,
        number=3,
    )


class ListFilesResponse(proto.Message):
    r"""Response for ``ListFiles``.

    Attributes:
        files (MutableSequence[google.ai.generativelanguage_v1beta.types.File]):
            The list of ``File``\ s.
        next_page_token (str):
            A token that can be sent as a ``page_token`` into a
            subsequent ``ListFiles`` call.
    """

    @property
    def raw_page(self):
        return self

    files: MutableSequence[gag_file.File] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=gag_file.File,
    )
    next_page_token: str = proto.Field(
        proto.STRING,
        number=2,
    )


class GetFileRequest(proto.Message):
    r"""Request for ``GetFile``.

    Attributes:
        name (str):
            Required. The name of the ``File`` to get. Example:
            ``files/abc-123``
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class DeleteFileRequest(proto.Message):
    r"""Request for ``DeleteFile``.

    Attributes:
        name (str):
            Required. The name of the ``File`` to delete. Example:
            ``files/abc-123``
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
