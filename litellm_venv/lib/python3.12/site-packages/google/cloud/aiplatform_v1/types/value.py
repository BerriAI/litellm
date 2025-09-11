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


__protobuf__ = proto.module(
    package="google.cloud.aiplatform.v1",
    manifest={
        "Value",
    },
)


class Value(proto.Message):
    r"""Value is the value of the field.

    This message has `oneof`_ fields (mutually exclusive fields).
    For each oneof, at most one member field can be set at the same time.
    Setting any member of the oneof automatically clears all other
    members.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        int_value (int):
            An integer value.

            This field is a member of `oneof`_ ``value``.
        double_value (float):
            A double value.

            This field is a member of `oneof`_ ``value``.
        string_value (str):
            A string value.

            This field is a member of `oneof`_ ``value``.
    """

    int_value: int = proto.Field(
        proto.INT64,
        number=1,
        oneof="value",
    )
    double_value: float = proto.Field(
        proto.DOUBLE,
        number=2,
        oneof="value",
    )
    string_value: str = proto.Field(
        proto.STRING,
        number=3,
        oneof="value",
    )


__all__ = tuple(sorted(__protobuf__.manifest))
