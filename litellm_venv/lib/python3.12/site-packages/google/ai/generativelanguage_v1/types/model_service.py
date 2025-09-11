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

from google.ai.generativelanguage_v1.types import model

__protobuf__ = proto.module(
    package="google.ai.generativelanguage.v1",
    manifest={
        "GetModelRequest",
        "ListModelsRequest",
        "ListModelsResponse",
    },
)


class GetModelRequest(proto.Message):
    r"""Request for getting information about a specific Model.

    Attributes:
        name (str):
            Required. The resource name of the model.

            This name should match a model name returned by the
            ``ListModels`` method.

            Format: ``models/{model}``
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class ListModelsRequest(proto.Message):
    r"""Request for listing all Models.

    Attributes:
        page_size (int):
            The maximum number of ``Models`` to return (per page).

            The service may return fewer models. If unspecified, at most
            50 models will be returned per page. This method returns at
            most 1000 models per page, even if you pass a larger
            page_size.
        page_token (str):
            A page token, received from a previous ``ListModels`` call.

            Provide the ``page_token`` returned by one request as an
            argument to the next request to retrieve the next page.

            When paginating, all other parameters provided to
            ``ListModels`` must match the call that provided the page
            token.
    """

    page_size: int = proto.Field(
        proto.INT32,
        number=2,
    )
    page_token: str = proto.Field(
        proto.STRING,
        number=3,
    )


class ListModelsResponse(proto.Message):
    r"""Response from ``ListModel`` containing a paginated list of Models.

    Attributes:
        models (MutableSequence[google.ai.generativelanguage_v1.types.Model]):
            The returned Models.
        next_page_token (str):
            A token, which can be sent as ``page_token`` to retrieve the
            next page.

            If this field is omitted, there are no more pages.
    """

    @property
    def raw_page(self):
        return self

    models: MutableSequence[model.Model] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=model.Model,
    )
    next_page_token: str = proto.Field(
        proto.STRING,
        number=2,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
