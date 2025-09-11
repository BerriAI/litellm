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
    package="google.cloud.aiplatform.v1beta1",
    manifest={
        "IdMatcher",
        "FeatureSelector",
    },
)


class IdMatcher(proto.Message):
    r"""Matcher for Features of an EntityType by Feature ID.

    Attributes:
        ids (MutableSequence[str]):
            Required. The following are accepted as ``ids``:

            -  A single-element list containing only ``*``, which
               selects all Features in the target EntityType, or
            -  A list containing only Feature IDs, which selects only
               Features with those IDs in the target EntityType.
    """

    ids: MutableSequence[str] = proto.RepeatedField(
        proto.STRING,
        number=1,
    )


class FeatureSelector(proto.Message):
    r"""Selector for Features of an EntityType.

    Attributes:
        id_matcher (google.cloud.aiplatform_v1beta1.types.IdMatcher):
            Required. Matches Features based on ID.
    """

    id_matcher: "IdMatcher" = proto.Field(
        proto.MESSAGE,
        number=1,
        message="IdMatcher",
    )


__all__ = tuple(sorted(__protobuf__.manifest))
