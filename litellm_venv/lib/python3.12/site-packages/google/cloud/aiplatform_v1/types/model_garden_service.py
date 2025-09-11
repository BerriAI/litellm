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
        "PublisherModelView",
        "GetPublisherModelRequest",
    },
)


class PublisherModelView(proto.Enum):
    r"""View enumeration of PublisherModel.

    Values:
        PUBLISHER_MODEL_VIEW_UNSPECIFIED (0):
            The default / unset value. The API will
            default to the BASIC view.
        PUBLISHER_MODEL_VIEW_BASIC (1):
            Include basic metadata about the publisher
            model, but not the full contents.
        PUBLISHER_MODEL_VIEW_FULL (2):
            Include everything.
        PUBLISHER_MODEL_VERSION_VIEW_BASIC (3):
            Include: VersionId, ModelVersionExternalName,
            and SupportedActions.
    """
    PUBLISHER_MODEL_VIEW_UNSPECIFIED = 0
    PUBLISHER_MODEL_VIEW_BASIC = 1
    PUBLISHER_MODEL_VIEW_FULL = 2
    PUBLISHER_MODEL_VERSION_VIEW_BASIC = 3


class GetPublisherModelRequest(proto.Message):
    r"""Request message for
    [ModelGardenService.GetPublisherModel][google.cloud.aiplatform.v1.ModelGardenService.GetPublisherModel]

    Attributes:
        name (str):
            Required. The name of the PublisherModel resource. Format:
            ``publishers/{publisher}/models/{publisher_model}``
        language_code (str):
            Optional. The IETF BCP-47 language code
            representing the language in which the publisher
            model's text information should be written in
            (see go/bcp47).
        view (google.cloud.aiplatform_v1.types.PublisherModelView):
            Optional. PublisherModel view specifying
            which fields to read.
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    language_code: str = proto.Field(
        proto.STRING,
        number=2,
    )
    view: "PublisherModelView" = proto.Field(
        proto.ENUM,
        number=3,
        enum="PublisherModelView",
    )


__all__ = tuple(sorted(__protobuf__.manifest))
