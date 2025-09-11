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

from google.protobuf import field_mask_pb2  # type: ignore
import proto  # type: ignore

from google.ai.generativelanguage_v1beta.types import tuned_model as gag_tuned_model
from google.ai.generativelanguage_v1beta.types import model

__protobuf__ = proto.module(
    package="google.ai.generativelanguage.v1beta",
    manifest={
        "GetModelRequest",
        "ListModelsRequest",
        "ListModelsResponse",
        "GetTunedModelRequest",
        "ListTunedModelsRequest",
        "ListTunedModelsResponse",
        "CreateTunedModelRequest",
        "CreateTunedModelMetadata",
        "UpdateTunedModelRequest",
        "DeleteTunedModelRequest",
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
        models (MutableSequence[google.ai.generativelanguage_v1beta.types.Model]):
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


class GetTunedModelRequest(proto.Message):
    r"""Request for getting information about a specific Model.

    Attributes:
        name (str):
            Required. The resource name of the model.

            Format: ``tunedModels/my-model-id``
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class ListTunedModelsRequest(proto.Message):
    r"""Request for listing TunedModels.

    Attributes:
        page_size (int):
            Optional. The maximum number of ``TunedModels`` to return
            (per page). The service may return fewer tuned models.

            If unspecified, at most 10 tuned models will be returned.
            This method returns at most 1000 models per page, even if
            you pass a larger page_size.
        page_token (str):
            Optional. A page token, received from a previous
            ``ListTunedModels`` call.

            Provide the ``page_token`` returned by one request as an
            argument to the next request to retrieve the next page.

            When paginating, all other parameters provided to
            ``ListTunedModels`` must match the call that provided the
            page token.
        filter (str):
            Optional. A filter is a full text search over
            the tuned model's description and display name.
            By default, results will not include tuned
            models shared with everyone.

            Additional operators:

              - owner:me
              - writers:me
              - readers:me
              - readers:everyone

            Examples:

              "owner:me" returns all tuned models to which
            caller has owner role   "readers:me" returns all
            tuned models to which caller has reader role
            "readers:everyone" returns all tuned models that
            are shared with everyone
    """

    page_size: int = proto.Field(
        proto.INT32,
        number=1,
    )
    page_token: str = proto.Field(
        proto.STRING,
        number=2,
    )
    filter: str = proto.Field(
        proto.STRING,
        number=3,
    )


class ListTunedModelsResponse(proto.Message):
    r"""Response from ``ListTunedModels`` containing a paginated list of
    Models.

    Attributes:
        tuned_models (MutableSequence[google.ai.generativelanguage_v1beta.types.TunedModel]):
            The returned Models.
        next_page_token (str):
            A token, which can be sent as ``page_token`` to retrieve the
            next page.

            If this field is omitted, there are no more pages.
    """

    @property
    def raw_page(self):
        return self

    tuned_models: MutableSequence[gag_tuned_model.TunedModel] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=gag_tuned_model.TunedModel,
    )
    next_page_token: str = proto.Field(
        proto.STRING,
        number=2,
    )


class CreateTunedModelRequest(proto.Message):
    r"""Request to create a TunedModel.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        tuned_model_id (str):
            Optional. The unique id for the tuned model if specified.
            This value should be up to 40 characters, the first
            character must be a letter, the last could be a letter or a
            number. The id must match the regular expression:
            `a-z <[a-z0-9-]{0,38}[a-z0-9]>`__?.

            This field is a member of `oneof`_ ``_tuned_model_id``.
        tuned_model (google.ai.generativelanguage_v1beta.types.TunedModel):
            Required. The tuned model to create.
    """

    tuned_model_id: str = proto.Field(
        proto.STRING,
        number=1,
        optional=True,
    )
    tuned_model: gag_tuned_model.TunedModel = proto.Field(
        proto.MESSAGE,
        number=2,
        message=gag_tuned_model.TunedModel,
    )


class CreateTunedModelMetadata(proto.Message):
    r"""Metadata about the state and progress of creating a tuned
    model returned from the long-running operation

    Attributes:
        tuned_model (str):
            Name of the tuned model associated with the
            tuning operation.
        total_steps (int):
            The total number of tuning steps.
        completed_steps (int):
            The number of steps completed.
        completed_percent (float):
            The completed percentage for the tuning
            operation.
        snapshots (MutableSequence[google.ai.generativelanguage_v1beta.types.TuningSnapshot]):
            Metrics collected during tuning.
    """

    tuned_model: str = proto.Field(
        proto.STRING,
        number=5,
    )
    total_steps: int = proto.Field(
        proto.INT32,
        number=1,
    )
    completed_steps: int = proto.Field(
        proto.INT32,
        number=2,
    )
    completed_percent: float = proto.Field(
        proto.FLOAT,
        number=3,
    )
    snapshots: MutableSequence[gag_tuned_model.TuningSnapshot] = proto.RepeatedField(
        proto.MESSAGE,
        number=4,
        message=gag_tuned_model.TuningSnapshot,
    )


class UpdateTunedModelRequest(proto.Message):
    r"""Request to update a TunedModel.

    Attributes:
        tuned_model (google.ai.generativelanguage_v1beta.types.TunedModel):
            Required. The tuned model to update.
        update_mask (google.protobuf.field_mask_pb2.FieldMask):
            Required. The list of fields to update.
    """

    tuned_model: gag_tuned_model.TunedModel = proto.Field(
        proto.MESSAGE,
        number=1,
        message=gag_tuned_model.TunedModel,
    )
    update_mask: field_mask_pb2.FieldMask = proto.Field(
        proto.MESSAGE,
        number=2,
        message=field_mask_pb2.FieldMask,
    )


class DeleteTunedModelRequest(proto.Message):
    r"""Request to delete a TunedModel.

    Attributes:
        name (str):
            Required. The resource name of the model. Format:
            ``tunedModels/my-model-id``
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
