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
#
from __future__ import annotations

from typing import MutableMapping, MutableSequence

import proto  # type: ignore

__protobuf__ = proto.module(
    package="google.ai.generativelanguage.v1beta3",
    manifest={
        "CitationMetadata",
        "CitationSource",
    },
)


class CitationMetadata(proto.Message):
    r"""A collection of source attributions for a piece of content.

    Attributes:
        citation_sources (MutableSequence[google.ai.generativelanguage_v1beta3.types.CitationSource]):
            Citations to sources for a specific response.
    """

    citation_sources: MutableSequence["CitationSource"] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message="CitationSource",
    )


class CitationSource(proto.Message):
    r"""A citation to a source for a portion of a specific response.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        start_index (int):
            Optional. Start of segment of the response
            that is attributed to this source.

            Index indicates the start of the segment,
            measured in bytes.

            This field is a member of `oneof`_ ``_start_index``.
        end_index (int):
            Optional. End of the attributed segment,
            exclusive.

            This field is a member of `oneof`_ ``_end_index``.
        uri (str):
            Optional. URI that is attributed as a source
            for a portion of the text.

            This field is a member of `oneof`_ ``_uri``.
        license_ (str):
            Optional. License for the GitHub project that
            is attributed as a source for segment.

            License info is required for code citations.

            This field is a member of `oneof`_ ``_license``.
    """

    start_index: int = proto.Field(
        proto.INT32,
        number=1,
        optional=True,
    )
    end_index: int = proto.Field(
        proto.INT32,
        number=2,
        optional=True,
    )
    uri: str = proto.Field(
        proto.STRING,
        number=3,
        optional=True,
    )
    license_: str = proto.Field(
        proto.STRING,
        number=4,
        optional=True,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
