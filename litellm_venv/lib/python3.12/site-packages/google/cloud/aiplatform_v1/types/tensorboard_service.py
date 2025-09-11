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

from google.cloud.aiplatform_v1.types import operation
from google.cloud.aiplatform_v1.types import tensorboard as gca_tensorboard
from google.cloud.aiplatform_v1.types import tensorboard_data
from google.cloud.aiplatform_v1.types import (
    tensorboard_experiment as gca_tensorboard_experiment,
)
from google.cloud.aiplatform_v1.types import tensorboard_run as gca_tensorboard_run
from google.cloud.aiplatform_v1.types import (
    tensorboard_time_series as gca_tensorboard_time_series,
)
from google.protobuf import field_mask_pb2  # type: ignore


__protobuf__ = proto.module(
    package="google.cloud.aiplatform.v1",
    manifest={
        "CreateTensorboardRequest",
        "GetTensorboardRequest",
        "ListTensorboardsRequest",
        "ListTensorboardsResponse",
        "UpdateTensorboardRequest",
        "DeleteTensorboardRequest",
        "ReadTensorboardUsageRequest",
        "ReadTensorboardUsageResponse",
        "ReadTensorboardSizeRequest",
        "ReadTensorboardSizeResponse",
        "CreateTensorboardExperimentRequest",
        "GetTensorboardExperimentRequest",
        "ListTensorboardExperimentsRequest",
        "ListTensorboardExperimentsResponse",
        "UpdateTensorboardExperimentRequest",
        "DeleteTensorboardExperimentRequest",
        "BatchCreateTensorboardRunsRequest",
        "BatchCreateTensorboardRunsResponse",
        "CreateTensorboardRunRequest",
        "GetTensorboardRunRequest",
        "ReadTensorboardBlobDataRequest",
        "ReadTensorboardBlobDataResponse",
        "ListTensorboardRunsRequest",
        "ListTensorboardRunsResponse",
        "UpdateTensorboardRunRequest",
        "DeleteTensorboardRunRequest",
        "BatchCreateTensorboardTimeSeriesRequest",
        "BatchCreateTensorboardTimeSeriesResponse",
        "CreateTensorboardTimeSeriesRequest",
        "GetTensorboardTimeSeriesRequest",
        "ListTensorboardTimeSeriesRequest",
        "ListTensorboardTimeSeriesResponse",
        "UpdateTensorboardTimeSeriesRequest",
        "DeleteTensorboardTimeSeriesRequest",
        "BatchReadTensorboardTimeSeriesDataRequest",
        "BatchReadTensorboardTimeSeriesDataResponse",
        "ReadTensorboardTimeSeriesDataRequest",
        "ReadTensorboardTimeSeriesDataResponse",
        "WriteTensorboardExperimentDataRequest",
        "WriteTensorboardExperimentDataResponse",
        "WriteTensorboardRunDataRequest",
        "WriteTensorboardRunDataResponse",
        "ExportTensorboardTimeSeriesDataRequest",
        "ExportTensorboardTimeSeriesDataResponse",
        "CreateTensorboardOperationMetadata",
        "UpdateTensorboardOperationMetadata",
    },
)


class CreateTensorboardRequest(proto.Message):
    r"""Request message for
    [TensorboardService.CreateTensorboard][google.cloud.aiplatform.v1.TensorboardService.CreateTensorboard].

    Attributes:
        parent (str):
            Required. The resource name of the Location to create the
            Tensorboard in. Format:
            ``projects/{project}/locations/{location}``
        tensorboard (google.cloud.aiplatform_v1.types.Tensorboard):
            Required. The Tensorboard to create.
    """

    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )
    tensorboard: gca_tensorboard.Tensorboard = proto.Field(
        proto.MESSAGE,
        number=2,
        message=gca_tensorboard.Tensorboard,
    )


class GetTensorboardRequest(proto.Message):
    r"""Request message for
    [TensorboardService.GetTensorboard][google.cloud.aiplatform.v1.TensorboardService.GetTensorboard].

    Attributes:
        name (str):
            Required. The name of the Tensorboard resource. Format:
            ``projects/{project}/locations/{location}/tensorboards/{tensorboard}``
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class ListTensorboardsRequest(proto.Message):
    r"""Request message for
    [TensorboardService.ListTensorboards][google.cloud.aiplatform.v1.TensorboardService.ListTensorboards].

    Attributes:
        parent (str):
            Required. The resource name of the Location to list
            Tensorboards. Format:
            ``projects/{project}/locations/{location}``
        filter (str):
            Lists the Tensorboards that match the filter
            expression.
        page_size (int):
            The maximum number of Tensorboards to return.
            The service may return fewer than this value. If
            unspecified, at most 100 Tensorboards are
            returned. The maximum value is 100; values above
            100 are coerced to 100.
        page_token (str):
            A page token, received from a previous
            [TensorboardService.ListTensorboards][google.cloud.aiplatform.v1.TensorboardService.ListTensorboards]
            call. Provide this to retrieve the subsequent page.

            When paginating, all other parameters provided to
            [TensorboardService.ListTensorboards][google.cloud.aiplatform.v1.TensorboardService.ListTensorboards]
            must match the call that provided the page token.
        order_by (str):
            Field to use to sort the list.
        read_mask (google.protobuf.field_mask_pb2.FieldMask):
            Mask specifying which fields to read.
    """

    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )
    filter: str = proto.Field(
        proto.STRING,
        number=2,
    )
    page_size: int = proto.Field(
        proto.INT32,
        number=3,
    )
    page_token: str = proto.Field(
        proto.STRING,
        number=4,
    )
    order_by: str = proto.Field(
        proto.STRING,
        number=5,
    )
    read_mask: field_mask_pb2.FieldMask = proto.Field(
        proto.MESSAGE,
        number=6,
        message=field_mask_pb2.FieldMask,
    )


class ListTensorboardsResponse(proto.Message):
    r"""Response message for
    [TensorboardService.ListTensorboards][google.cloud.aiplatform.v1.TensorboardService.ListTensorboards].

    Attributes:
        tensorboards (MutableSequence[google.cloud.aiplatform_v1.types.Tensorboard]):
            The Tensorboards mathching the request.
        next_page_token (str):
            A token, which can be sent as
            [ListTensorboardsRequest.page_token][google.cloud.aiplatform.v1.ListTensorboardsRequest.page_token]
            to retrieve the next page. If this field is omitted, there
            are no subsequent pages.
    """

    @property
    def raw_page(self):
        return self

    tensorboards: MutableSequence[gca_tensorboard.Tensorboard] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=gca_tensorboard.Tensorboard,
    )
    next_page_token: str = proto.Field(
        proto.STRING,
        number=2,
    )


class UpdateTensorboardRequest(proto.Message):
    r"""Request message for
    [TensorboardService.UpdateTensorboard][google.cloud.aiplatform.v1.TensorboardService.UpdateTensorboard].

    Attributes:
        update_mask (google.protobuf.field_mask_pb2.FieldMask):
            Required. Field mask is used to specify the fields to be
            overwritten in the Tensorboard resource by the update. The
            fields specified in the update_mask are relative to the
            resource, not the full request. A field is overwritten if
            it's in the mask. If the user does not provide a mask then
            all fields are overwritten if new values are specified.
        tensorboard (google.cloud.aiplatform_v1.types.Tensorboard):
            Required. The Tensorboard's ``name`` field is used to
            identify the Tensorboard to be updated. Format:
            ``projects/{project}/locations/{location}/tensorboards/{tensorboard}``
    """

    update_mask: field_mask_pb2.FieldMask = proto.Field(
        proto.MESSAGE,
        number=1,
        message=field_mask_pb2.FieldMask,
    )
    tensorboard: gca_tensorboard.Tensorboard = proto.Field(
        proto.MESSAGE,
        number=2,
        message=gca_tensorboard.Tensorboard,
    )


class DeleteTensorboardRequest(proto.Message):
    r"""Request message for
    [TensorboardService.DeleteTensorboard][google.cloud.aiplatform.v1.TensorboardService.DeleteTensorboard].

    Attributes:
        name (str):
            Required. The name of the Tensorboard to be deleted. Format:
            ``projects/{project}/locations/{location}/tensorboards/{tensorboard}``
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class ReadTensorboardUsageRequest(proto.Message):
    r"""Request message for
    [TensorboardService.ReadTensorboardUsage][google.cloud.aiplatform.v1.TensorboardService.ReadTensorboardUsage].

    Attributes:
        tensorboard (str):
            Required. The name of the Tensorboard resource. Format:
            ``projects/{project}/locations/{location}/tensorboards/{tensorboard}``
    """

    tensorboard: str = proto.Field(
        proto.STRING,
        number=1,
    )


class ReadTensorboardUsageResponse(proto.Message):
    r"""Response message for
    [TensorboardService.ReadTensorboardUsage][google.cloud.aiplatform.v1.TensorboardService.ReadTensorboardUsage].

    Attributes:
        monthly_usage_data (MutableMapping[str, google.cloud.aiplatform_v1.types.ReadTensorboardUsageResponse.PerMonthUsageData]):
            Maps year-month (YYYYMM) string to per month
            usage data.
    """

    class PerUserUsageData(proto.Message):
        r"""Per user usage data.

        Attributes:
            username (str):
                User's username
            view_count (int):
                Number of times the user has read data within
                the Tensorboard.
        """

        username: str = proto.Field(
            proto.STRING,
            number=1,
        )
        view_count: int = proto.Field(
            proto.INT64,
            number=2,
        )

    class PerMonthUsageData(proto.Message):
        r"""Per month usage data

        Attributes:
            user_usage_data (MutableSequence[google.cloud.aiplatform_v1.types.ReadTensorboardUsageResponse.PerUserUsageData]):
                Usage data for each user in the given month.
        """

        user_usage_data: MutableSequence[
            "ReadTensorboardUsageResponse.PerUserUsageData"
        ] = proto.RepeatedField(
            proto.MESSAGE,
            number=1,
            message="ReadTensorboardUsageResponse.PerUserUsageData",
        )

    monthly_usage_data: MutableMapping[str, PerMonthUsageData] = proto.MapField(
        proto.STRING,
        proto.MESSAGE,
        number=1,
        message=PerMonthUsageData,
    )


class ReadTensorboardSizeRequest(proto.Message):
    r"""Request message for
    [TensorboardService.ReadTensorboardSize][google.cloud.aiplatform.v1.TensorboardService.ReadTensorboardSize].

    Attributes:
        tensorboard (str):
            Required. The name of the Tensorboard resource. Format:
            ``projects/{project}/locations/{location}/tensorboards/{tensorboard}``
    """

    tensorboard: str = proto.Field(
        proto.STRING,
        number=1,
    )


class ReadTensorboardSizeResponse(proto.Message):
    r"""Response message for
    [TensorboardService.ReadTensorboardSize][google.cloud.aiplatform.v1.TensorboardService.ReadTensorboardSize].

    Attributes:
        storage_size_byte (int):
            Payload storage size for the TensorBoard
    """

    storage_size_byte: int = proto.Field(
        proto.INT64,
        number=1,
    )


class CreateTensorboardExperimentRequest(proto.Message):
    r"""Request message for
    [TensorboardService.CreateTensorboardExperiment][google.cloud.aiplatform.v1.TensorboardService.CreateTensorboardExperiment].

    Attributes:
        parent (str):
            Required. The resource name of the Tensorboard to create the
            TensorboardExperiment in. Format:
            ``projects/{project}/locations/{location}/tensorboards/{tensorboard}``
        tensorboard_experiment (google.cloud.aiplatform_v1.types.TensorboardExperiment):
            The TensorboardExperiment to create.
        tensorboard_experiment_id (str):
            Required. The ID to use for the Tensorboard experiment,
            which becomes the final component of the Tensorboard
            experiment's resource name.

            This value should be 1-128 characters, and valid characters
            are ``/[a-z][0-9]-/``.
    """

    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )
    tensorboard_experiment: gca_tensorboard_experiment.TensorboardExperiment = (
        proto.Field(
            proto.MESSAGE,
            number=2,
            message=gca_tensorboard_experiment.TensorboardExperiment,
        )
    )
    tensorboard_experiment_id: str = proto.Field(
        proto.STRING,
        number=3,
    )


class GetTensorboardExperimentRequest(proto.Message):
    r"""Request message for
    [TensorboardService.GetTensorboardExperiment][google.cloud.aiplatform.v1.TensorboardService.GetTensorboardExperiment].

    Attributes:
        name (str):
            Required. The name of the TensorboardExperiment resource.
            Format:
            ``projects/{project}/locations/{location}/tensorboards/{tensorboard}/experiments/{experiment}``
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class ListTensorboardExperimentsRequest(proto.Message):
    r"""Request message for
    [TensorboardService.ListTensorboardExperiments][google.cloud.aiplatform.v1.TensorboardService.ListTensorboardExperiments].

    Attributes:
        parent (str):
            Required. The resource name of the Tensorboard to list
            TensorboardExperiments. Format:
            ``projects/{project}/locations/{location}/tensorboards/{tensorboard}``
        filter (str):
            Lists the TensorboardExperiments that match
            the filter expression.
        page_size (int):
            The maximum number of TensorboardExperiments
            to return. The service may return fewer than
            this value. If unspecified, at most 50
            TensorboardExperiments are returned. The maximum
            value is 1000; values above 1000 are coerced to
            1000.
        page_token (str):
            A page token, received from a previous
            [TensorboardService.ListTensorboardExperiments][google.cloud.aiplatform.v1.TensorboardService.ListTensorboardExperiments]
            call. Provide this to retrieve the subsequent page.

            When paginating, all other parameters provided to
            [TensorboardService.ListTensorboardExperiments][google.cloud.aiplatform.v1.TensorboardService.ListTensorboardExperiments]
            must match the call that provided the page token.
        order_by (str):
            Field to use to sort the list.
        read_mask (google.protobuf.field_mask_pb2.FieldMask):
            Mask specifying which fields to read.
    """

    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )
    filter: str = proto.Field(
        proto.STRING,
        number=2,
    )
    page_size: int = proto.Field(
        proto.INT32,
        number=3,
    )
    page_token: str = proto.Field(
        proto.STRING,
        number=4,
    )
    order_by: str = proto.Field(
        proto.STRING,
        number=5,
    )
    read_mask: field_mask_pb2.FieldMask = proto.Field(
        proto.MESSAGE,
        number=6,
        message=field_mask_pb2.FieldMask,
    )


class ListTensorboardExperimentsResponse(proto.Message):
    r"""Response message for
    [TensorboardService.ListTensorboardExperiments][google.cloud.aiplatform.v1.TensorboardService.ListTensorboardExperiments].

    Attributes:
        tensorboard_experiments (MutableSequence[google.cloud.aiplatform_v1.types.TensorboardExperiment]):
            The TensorboardExperiments mathching the
            request.
        next_page_token (str):
            A token, which can be sent as
            [ListTensorboardExperimentsRequest.page_token][google.cloud.aiplatform.v1.ListTensorboardExperimentsRequest.page_token]
            to retrieve the next page. If this field is omitted, there
            are no subsequent pages.
    """

    @property
    def raw_page(self):
        return self

    tensorboard_experiments: MutableSequence[
        gca_tensorboard_experiment.TensorboardExperiment
    ] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=gca_tensorboard_experiment.TensorboardExperiment,
    )
    next_page_token: str = proto.Field(
        proto.STRING,
        number=2,
    )


class UpdateTensorboardExperimentRequest(proto.Message):
    r"""Request message for
    [TensorboardService.UpdateTensorboardExperiment][google.cloud.aiplatform.v1.TensorboardService.UpdateTensorboardExperiment].

    Attributes:
        update_mask (google.protobuf.field_mask_pb2.FieldMask):
            Required. Field mask is used to specify the fields to be
            overwritten in the TensorboardExperiment resource by the
            update. The fields specified in the update_mask are relative
            to the resource, not the full request. A field is
            overwritten if it's in the mask. If the user does not
            provide a mask then all fields are overwritten if new values
            are specified.
        tensorboard_experiment (google.cloud.aiplatform_v1.types.TensorboardExperiment):
            Required. The TensorboardExperiment's ``name`` field is used
            to identify the TensorboardExperiment to be updated. Format:
            ``projects/{project}/locations/{location}/tensorboards/{tensorboard}/experiments/{experiment}``
    """

    update_mask: field_mask_pb2.FieldMask = proto.Field(
        proto.MESSAGE,
        number=1,
        message=field_mask_pb2.FieldMask,
    )
    tensorboard_experiment: gca_tensorboard_experiment.TensorboardExperiment = (
        proto.Field(
            proto.MESSAGE,
            number=2,
            message=gca_tensorboard_experiment.TensorboardExperiment,
        )
    )


class DeleteTensorboardExperimentRequest(proto.Message):
    r"""Request message for
    [TensorboardService.DeleteTensorboardExperiment][google.cloud.aiplatform.v1.TensorboardService.DeleteTensorboardExperiment].

    Attributes:
        name (str):
            Required. The name of the TensorboardExperiment to be
            deleted. Format:
            ``projects/{project}/locations/{location}/tensorboards/{tensorboard}/experiments/{experiment}``
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class BatchCreateTensorboardRunsRequest(proto.Message):
    r"""Request message for
    [TensorboardService.BatchCreateTensorboardRuns][google.cloud.aiplatform.v1.TensorboardService.BatchCreateTensorboardRuns].

    Attributes:
        parent (str):
            Required. The resource name of the TensorboardExperiment to
            create the TensorboardRuns in. Format:
            ``projects/{project}/locations/{location}/tensorboards/{tensorboard}/experiments/{experiment}``
            The parent field in the CreateTensorboardRunRequest messages
            must match this field.
        requests (MutableSequence[google.cloud.aiplatform_v1.types.CreateTensorboardRunRequest]):
            Required. The request message specifying the
            TensorboardRuns to create. A maximum of 1000
            TensorboardRuns can be created in a batch.
    """

    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )
    requests: MutableSequence["CreateTensorboardRunRequest"] = proto.RepeatedField(
        proto.MESSAGE,
        number=2,
        message="CreateTensorboardRunRequest",
    )


class BatchCreateTensorboardRunsResponse(proto.Message):
    r"""Response message for
    [TensorboardService.BatchCreateTensorboardRuns][google.cloud.aiplatform.v1.TensorboardService.BatchCreateTensorboardRuns].

    Attributes:
        tensorboard_runs (MutableSequence[google.cloud.aiplatform_v1.types.TensorboardRun]):
            The created TensorboardRuns.
    """

    tensorboard_runs: MutableSequence[
        gca_tensorboard_run.TensorboardRun
    ] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=gca_tensorboard_run.TensorboardRun,
    )


class CreateTensorboardRunRequest(proto.Message):
    r"""Request message for
    [TensorboardService.CreateTensorboardRun][google.cloud.aiplatform.v1.TensorboardService.CreateTensorboardRun].

    Attributes:
        parent (str):
            Required. The resource name of the TensorboardExperiment to
            create the TensorboardRun in. Format:
            ``projects/{project}/locations/{location}/tensorboards/{tensorboard}/experiments/{experiment}``
        tensorboard_run (google.cloud.aiplatform_v1.types.TensorboardRun):
            Required. The TensorboardRun to create.
        tensorboard_run_id (str):
            Required. The ID to use for the Tensorboard run, which
            becomes the final component of the Tensorboard run's
            resource name.

            This value should be 1-128 characters, and valid characters
            are ``/[a-z][0-9]-/``.
    """

    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )
    tensorboard_run: gca_tensorboard_run.TensorboardRun = proto.Field(
        proto.MESSAGE,
        number=2,
        message=gca_tensorboard_run.TensorboardRun,
    )
    tensorboard_run_id: str = proto.Field(
        proto.STRING,
        number=3,
    )


class GetTensorboardRunRequest(proto.Message):
    r"""Request message for
    [TensorboardService.GetTensorboardRun][google.cloud.aiplatform.v1.TensorboardService.GetTensorboardRun].

    Attributes:
        name (str):
            Required. The name of the TensorboardRun resource. Format:
            ``projects/{project}/locations/{location}/tensorboards/{tensorboard}/experiments/{experiment}/runs/{run}``
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class ReadTensorboardBlobDataRequest(proto.Message):
    r"""Request message for
    [TensorboardService.ReadTensorboardBlobData][google.cloud.aiplatform.v1.TensorboardService.ReadTensorboardBlobData].

    Attributes:
        time_series (str):
            Required. The resource name of the TensorboardTimeSeries to
            list Blobs. Format:
            ``projects/{project}/locations/{location}/tensorboards/{tensorboard}/experiments/{experiment}/runs/{run}/timeSeries/{time_series}``
        blob_ids (MutableSequence[str]):
            IDs of the blobs to read.
    """

    time_series: str = proto.Field(
        proto.STRING,
        number=1,
    )
    blob_ids: MutableSequence[str] = proto.RepeatedField(
        proto.STRING,
        number=2,
    )


class ReadTensorboardBlobDataResponse(proto.Message):
    r"""Response message for
    [TensorboardService.ReadTensorboardBlobData][google.cloud.aiplatform.v1.TensorboardService.ReadTensorboardBlobData].

    Attributes:
        blobs (MutableSequence[google.cloud.aiplatform_v1.types.TensorboardBlob]):
            Blob messages containing blob bytes.
    """

    blobs: MutableSequence[tensorboard_data.TensorboardBlob] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=tensorboard_data.TensorboardBlob,
    )


class ListTensorboardRunsRequest(proto.Message):
    r"""Request message for
    [TensorboardService.ListTensorboardRuns][google.cloud.aiplatform.v1.TensorboardService.ListTensorboardRuns].

    Attributes:
        parent (str):
            Required. The resource name of the TensorboardExperiment to
            list TensorboardRuns. Format:
            ``projects/{project}/locations/{location}/tensorboards/{tensorboard}/experiments/{experiment}``
        filter (str):
            Lists the TensorboardRuns that match the
            filter expression.
        page_size (int):
            The maximum number of TensorboardRuns to
            return. The service may return fewer than this
            value. If unspecified, at most 50
            TensorboardRuns are returned. The maximum value
            is 1000; values above 1000 are coerced to 1000.
        page_token (str):
            A page token, received from a previous
            [TensorboardService.ListTensorboardRuns][google.cloud.aiplatform.v1.TensorboardService.ListTensorboardRuns]
            call. Provide this to retrieve the subsequent page.

            When paginating, all other parameters provided to
            [TensorboardService.ListTensorboardRuns][google.cloud.aiplatform.v1.TensorboardService.ListTensorboardRuns]
            must match the call that provided the page token.
        order_by (str):
            Field to use to sort the list.
        read_mask (google.protobuf.field_mask_pb2.FieldMask):
            Mask specifying which fields to read.
    """

    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )
    filter: str = proto.Field(
        proto.STRING,
        number=2,
    )
    page_size: int = proto.Field(
        proto.INT32,
        number=3,
    )
    page_token: str = proto.Field(
        proto.STRING,
        number=4,
    )
    order_by: str = proto.Field(
        proto.STRING,
        number=5,
    )
    read_mask: field_mask_pb2.FieldMask = proto.Field(
        proto.MESSAGE,
        number=6,
        message=field_mask_pb2.FieldMask,
    )


class ListTensorboardRunsResponse(proto.Message):
    r"""Response message for
    [TensorboardService.ListTensorboardRuns][google.cloud.aiplatform.v1.TensorboardService.ListTensorboardRuns].

    Attributes:
        tensorboard_runs (MutableSequence[google.cloud.aiplatform_v1.types.TensorboardRun]):
            The TensorboardRuns mathching the request.
        next_page_token (str):
            A token, which can be sent as
            [ListTensorboardRunsRequest.page_token][google.cloud.aiplatform.v1.ListTensorboardRunsRequest.page_token]
            to retrieve the next page. If this field is omitted, there
            are no subsequent pages.
    """

    @property
    def raw_page(self):
        return self

    tensorboard_runs: MutableSequence[
        gca_tensorboard_run.TensorboardRun
    ] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=gca_tensorboard_run.TensorboardRun,
    )
    next_page_token: str = proto.Field(
        proto.STRING,
        number=2,
    )


class UpdateTensorboardRunRequest(proto.Message):
    r"""Request message for
    [TensorboardService.UpdateTensorboardRun][google.cloud.aiplatform.v1.TensorboardService.UpdateTensorboardRun].

    Attributes:
        update_mask (google.protobuf.field_mask_pb2.FieldMask):
            Required. Field mask is used to specify the fields to be
            overwritten in the TensorboardRun resource by the update.
            The fields specified in the update_mask are relative to the
            resource, not the full request. A field is overwritten if
            it's in the mask. If the user does not provide a mask then
            all fields are overwritten if new values are specified.
        tensorboard_run (google.cloud.aiplatform_v1.types.TensorboardRun):
            Required. The TensorboardRun's ``name`` field is used to
            identify the TensorboardRun to be updated. Format:
            ``projects/{project}/locations/{location}/tensorboards/{tensorboard}/experiments/{experiment}/runs/{run}``
    """

    update_mask: field_mask_pb2.FieldMask = proto.Field(
        proto.MESSAGE,
        number=1,
        message=field_mask_pb2.FieldMask,
    )
    tensorboard_run: gca_tensorboard_run.TensorboardRun = proto.Field(
        proto.MESSAGE,
        number=2,
        message=gca_tensorboard_run.TensorboardRun,
    )


class DeleteTensorboardRunRequest(proto.Message):
    r"""Request message for
    [TensorboardService.DeleteTensorboardRun][google.cloud.aiplatform.v1.TensorboardService.DeleteTensorboardRun].

    Attributes:
        name (str):
            Required. The name of the TensorboardRun to be deleted.
            Format:
            ``projects/{project}/locations/{location}/tensorboards/{tensorboard}/experiments/{experiment}/runs/{run}``
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class BatchCreateTensorboardTimeSeriesRequest(proto.Message):
    r"""Request message for
    [TensorboardService.BatchCreateTensorboardTimeSeries][google.cloud.aiplatform.v1.TensorboardService.BatchCreateTensorboardTimeSeries].

    Attributes:
        parent (str):
            Required. The resource name of the TensorboardExperiment to
            create the TensorboardTimeSeries in. Format:
            ``projects/{project}/locations/{location}/tensorboards/{tensorboard}/experiments/{experiment}``
            The TensorboardRuns referenced by the parent fields in the
            CreateTensorboardTimeSeriesRequest messages must be sub
            resources of this TensorboardExperiment.
        requests (MutableSequence[google.cloud.aiplatform_v1.types.CreateTensorboardTimeSeriesRequest]):
            Required. The request message specifying the
            TensorboardTimeSeries to create. A maximum of
            1000 TensorboardTimeSeries can be created in a
            batch.
    """

    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )
    requests: MutableSequence[
        "CreateTensorboardTimeSeriesRequest"
    ] = proto.RepeatedField(
        proto.MESSAGE,
        number=2,
        message="CreateTensorboardTimeSeriesRequest",
    )


class BatchCreateTensorboardTimeSeriesResponse(proto.Message):
    r"""Response message for
    [TensorboardService.BatchCreateTensorboardTimeSeries][google.cloud.aiplatform.v1.TensorboardService.BatchCreateTensorboardTimeSeries].

    Attributes:
        tensorboard_time_series (MutableSequence[google.cloud.aiplatform_v1.types.TensorboardTimeSeries]):
            The created TensorboardTimeSeries.
    """

    tensorboard_time_series: MutableSequence[
        gca_tensorboard_time_series.TensorboardTimeSeries
    ] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=gca_tensorboard_time_series.TensorboardTimeSeries,
    )


class CreateTensorboardTimeSeriesRequest(proto.Message):
    r"""Request message for
    [TensorboardService.CreateTensorboardTimeSeries][google.cloud.aiplatform.v1.TensorboardService.CreateTensorboardTimeSeries].

    Attributes:
        parent (str):
            Required. The resource name of the TensorboardRun to create
            the TensorboardTimeSeries in. Format:
            ``projects/{project}/locations/{location}/tensorboards/{tensorboard}/experiments/{experiment}/runs/{run}``
        tensorboard_time_series_id (str):
            Optional. The user specified unique ID to use for the
            TensorboardTimeSeries, which becomes the final component of
            the TensorboardTimeSeries's resource name. This value should
            match "[a-z0-9][a-z0-9-]{0, 127}".
        tensorboard_time_series (google.cloud.aiplatform_v1.types.TensorboardTimeSeries):
            Required. The TensorboardTimeSeries to
            create.
    """

    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )
    tensorboard_time_series_id: str = proto.Field(
        proto.STRING,
        number=3,
    )
    tensorboard_time_series: gca_tensorboard_time_series.TensorboardTimeSeries = (
        proto.Field(
            proto.MESSAGE,
            number=2,
            message=gca_tensorboard_time_series.TensorboardTimeSeries,
        )
    )


class GetTensorboardTimeSeriesRequest(proto.Message):
    r"""Request message for
    [TensorboardService.GetTensorboardTimeSeries][google.cloud.aiplatform.v1.TensorboardService.GetTensorboardTimeSeries].

    Attributes:
        name (str):
            Required. The name of the TensorboardTimeSeries resource.
            Format:
            ``projects/{project}/locations/{location}/tensorboards/{tensorboard}/experiments/{experiment}/runs/{run}/timeSeries/{time_series}``
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class ListTensorboardTimeSeriesRequest(proto.Message):
    r"""Request message for
    [TensorboardService.ListTensorboardTimeSeries][google.cloud.aiplatform.v1.TensorboardService.ListTensorboardTimeSeries].

    Attributes:
        parent (str):
            Required. The resource name of the TensorboardRun to list
            TensorboardTimeSeries. Format:
            ``projects/{project}/locations/{location}/tensorboards/{tensorboard}/experiments/{experiment}/runs/{run}``
        filter (str):
            Lists the TensorboardTimeSeries that match
            the filter expression.
        page_size (int):
            The maximum number of TensorboardTimeSeries
            to return. The service may return fewer than
            this value. If unspecified, at most 50
            TensorboardTimeSeries are returned. The maximum
            value is 1000; values above 1000 are coerced to
            1000.
        page_token (str):
            A page token, received from a previous
            [TensorboardService.ListTensorboardTimeSeries][google.cloud.aiplatform.v1.TensorboardService.ListTensorboardTimeSeries]
            call. Provide this to retrieve the subsequent page.

            When paginating, all other parameters provided to
            [TensorboardService.ListTensorboardTimeSeries][google.cloud.aiplatform.v1.TensorboardService.ListTensorboardTimeSeries]
            must match the call that provided the page token.
        order_by (str):
            Field to use to sort the list.
        read_mask (google.protobuf.field_mask_pb2.FieldMask):
            Mask specifying which fields to read.
    """

    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )
    filter: str = proto.Field(
        proto.STRING,
        number=2,
    )
    page_size: int = proto.Field(
        proto.INT32,
        number=3,
    )
    page_token: str = proto.Field(
        proto.STRING,
        number=4,
    )
    order_by: str = proto.Field(
        proto.STRING,
        number=5,
    )
    read_mask: field_mask_pb2.FieldMask = proto.Field(
        proto.MESSAGE,
        number=6,
        message=field_mask_pb2.FieldMask,
    )


class ListTensorboardTimeSeriesResponse(proto.Message):
    r"""Response message for
    [TensorboardService.ListTensorboardTimeSeries][google.cloud.aiplatform.v1.TensorboardService.ListTensorboardTimeSeries].

    Attributes:
        tensorboard_time_series (MutableSequence[google.cloud.aiplatform_v1.types.TensorboardTimeSeries]):
            The TensorboardTimeSeries mathching the
            request.
        next_page_token (str):
            A token, which can be sent as
            [ListTensorboardTimeSeriesRequest.page_token][google.cloud.aiplatform.v1.ListTensorboardTimeSeriesRequest.page_token]
            to retrieve the next page. If this field is omitted, there
            are no subsequent pages.
    """

    @property
    def raw_page(self):
        return self

    tensorboard_time_series: MutableSequence[
        gca_tensorboard_time_series.TensorboardTimeSeries
    ] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=gca_tensorboard_time_series.TensorboardTimeSeries,
    )
    next_page_token: str = proto.Field(
        proto.STRING,
        number=2,
    )


class UpdateTensorboardTimeSeriesRequest(proto.Message):
    r"""Request message for
    [TensorboardService.UpdateTensorboardTimeSeries][google.cloud.aiplatform.v1.TensorboardService.UpdateTensorboardTimeSeries].

    Attributes:
        update_mask (google.protobuf.field_mask_pb2.FieldMask):
            Required. Field mask is used to specify the fields to be
            overwritten in the TensorboardTimeSeries resource by the
            update. The fields specified in the update_mask are relative
            to the resource, not the full request. A field is
            overwritten if it's in the mask. If the user does not
            provide a mask then all fields are overwritten if new values
            are specified.
        tensorboard_time_series (google.cloud.aiplatform_v1.types.TensorboardTimeSeries):
            Required. The TensorboardTimeSeries' ``name`` field is used
            to identify the TensorboardTimeSeries to be updated. Format:
            ``projects/{project}/locations/{location}/tensorboards/{tensorboard}/experiments/{experiment}/runs/{run}/timeSeries/{time_series}``
    """

    update_mask: field_mask_pb2.FieldMask = proto.Field(
        proto.MESSAGE,
        number=1,
        message=field_mask_pb2.FieldMask,
    )
    tensorboard_time_series: gca_tensorboard_time_series.TensorboardTimeSeries = (
        proto.Field(
            proto.MESSAGE,
            number=2,
            message=gca_tensorboard_time_series.TensorboardTimeSeries,
        )
    )


class DeleteTensorboardTimeSeriesRequest(proto.Message):
    r"""Request message for
    [TensorboardService.DeleteTensorboardTimeSeries][google.cloud.aiplatform.v1.TensorboardService.DeleteTensorboardTimeSeries].

    Attributes:
        name (str):
            Required. The name of the TensorboardTimeSeries to be
            deleted. Format:
            ``projects/{project}/locations/{location}/tensorboards/{tensorboard}/experiments/{experiment}/runs/{run}/timeSeries/{time_series}``
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class BatchReadTensorboardTimeSeriesDataRequest(proto.Message):
    r"""Request message for
    [TensorboardService.BatchReadTensorboardTimeSeriesData][google.cloud.aiplatform.v1.TensorboardService.BatchReadTensorboardTimeSeriesData].

    Attributes:
        tensorboard (str):
            Required. The resource name of the Tensorboard containing
            TensorboardTimeSeries to read data from. Format:
            ``projects/{project}/locations/{location}/tensorboards/{tensorboard}``.
            The TensorboardTimeSeries referenced by
            [time_series][google.cloud.aiplatform.v1.BatchReadTensorboardTimeSeriesDataRequest.time_series]
            must be sub resources of this Tensorboard.
        time_series (MutableSequence[str]):
            Required. The resource names of the TensorboardTimeSeries to
            read data from. Format:
            ``projects/{project}/locations/{location}/tensorboards/{tensorboard}/experiments/{experiment}/runs/{run}/timeSeries/{time_series}``
    """

    tensorboard: str = proto.Field(
        proto.STRING,
        number=1,
    )
    time_series: MutableSequence[str] = proto.RepeatedField(
        proto.STRING,
        number=2,
    )


class BatchReadTensorboardTimeSeriesDataResponse(proto.Message):
    r"""Response message for
    [TensorboardService.BatchReadTensorboardTimeSeriesData][google.cloud.aiplatform.v1.TensorboardService.BatchReadTensorboardTimeSeriesData].

    Attributes:
        time_series_data (MutableSequence[google.cloud.aiplatform_v1.types.TimeSeriesData]):
            The returned time series data.
    """

    time_series_data: MutableSequence[
        tensorboard_data.TimeSeriesData
    ] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=tensorboard_data.TimeSeriesData,
    )


class ReadTensorboardTimeSeriesDataRequest(proto.Message):
    r"""Request message for
    [TensorboardService.ReadTensorboardTimeSeriesData][google.cloud.aiplatform.v1.TensorboardService.ReadTensorboardTimeSeriesData].

    Attributes:
        tensorboard_time_series (str):
            Required. The resource name of the TensorboardTimeSeries to
            read data from. Format:
            ``projects/{project}/locations/{location}/tensorboards/{tensorboard}/experiments/{experiment}/runs/{run}/timeSeries/{time_series}``
        max_data_points (int):
            The maximum number of TensorboardTimeSeries'
            data to return.
            This value should be a positive integer.
            This value can be set to -1 to return all data.
        filter (str):
            Reads the TensorboardTimeSeries' data that
            match the filter expression.
    """

    tensorboard_time_series: str = proto.Field(
        proto.STRING,
        number=1,
    )
    max_data_points: int = proto.Field(
        proto.INT32,
        number=2,
    )
    filter: str = proto.Field(
        proto.STRING,
        number=3,
    )


class ReadTensorboardTimeSeriesDataResponse(proto.Message):
    r"""Response message for
    [TensorboardService.ReadTensorboardTimeSeriesData][google.cloud.aiplatform.v1.TensorboardService.ReadTensorboardTimeSeriesData].

    Attributes:
        time_series_data (google.cloud.aiplatform_v1.types.TimeSeriesData):
            The returned time series data.
    """

    time_series_data: tensorboard_data.TimeSeriesData = proto.Field(
        proto.MESSAGE,
        number=1,
        message=tensorboard_data.TimeSeriesData,
    )


class WriteTensorboardExperimentDataRequest(proto.Message):
    r"""Request message for
    [TensorboardService.WriteTensorboardExperimentData][google.cloud.aiplatform.v1.TensorboardService.WriteTensorboardExperimentData].

    Attributes:
        tensorboard_experiment (str):
            Required. The resource name of the TensorboardExperiment to
            write data to. Format:
            ``projects/{project}/locations/{location}/tensorboards/{tensorboard}/experiments/{experiment}``
        write_run_data_requests (MutableSequence[google.cloud.aiplatform_v1.types.WriteTensorboardRunDataRequest]):
            Required. Requests containing per-run
            TensorboardTimeSeries data to write.
    """

    tensorboard_experiment: str = proto.Field(
        proto.STRING,
        number=1,
    )
    write_run_data_requests: MutableSequence[
        "WriteTensorboardRunDataRequest"
    ] = proto.RepeatedField(
        proto.MESSAGE,
        number=2,
        message="WriteTensorboardRunDataRequest",
    )


class WriteTensorboardExperimentDataResponse(proto.Message):
    r"""Response message for
    [TensorboardService.WriteTensorboardExperimentData][google.cloud.aiplatform.v1.TensorboardService.WriteTensorboardExperimentData].

    """


class WriteTensorboardRunDataRequest(proto.Message):
    r"""Request message for
    [TensorboardService.WriteTensorboardRunData][google.cloud.aiplatform.v1.TensorboardService.WriteTensorboardRunData].

    Attributes:
        tensorboard_run (str):
            Required. The resource name of the TensorboardRun to write
            data to. Format:
            ``projects/{project}/locations/{location}/tensorboards/{tensorboard}/experiments/{experiment}/runs/{run}``
        time_series_data (MutableSequence[google.cloud.aiplatform_v1.types.TimeSeriesData]):
            Required. The TensorboardTimeSeries data to
            write. Values with in a time series are indexed
            by their step value. Repeated writes to the same
            step will overwrite the existing value for that
            step.
            The upper limit of data points per write request
            is 5000.
    """

    tensorboard_run: str = proto.Field(
        proto.STRING,
        number=1,
    )
    time_series_data: MutableSequence[
        tensorboard_data.TimeSeriesData
    ] = proto.RepeatedField(
        proto.MESSAGE,
        number=2,
        message=tensorboard_data.TimeSeriesData,
    )


class WriteTensorboardRunDataResponse(proto.Message):
    r"""Response message for
    [TensorboardService.WriteTensorboardRunData][google.cloud.aiplatform.v1.TensorboardService.WriteTensorboardRunData].

    """


class ExportTensorboardTimeSeriesDataRequest(proto.Message):
    r"""Request message for
    [TensorboardService.ExportTensorboardTimeSeriesData][google.cloud.aiplatform.v1.TensorboardService.ExportTensorboardTimeSeriesData].

    Attributes:
        tensorboard_time_series (str):
            Required. The resource name of the TensorboardTimeSeries to
            export data from. Format:
            ``projects/{project}/locations/{location}/tensorboards/{tensorboard}/experiments/{experiment}/runs/{run}/timeSeries/{time_series}``
        filter (str):
            Exports the TensorboardTimeSeries' data that
            match the filter expression.
        page_size (int):
            The maximum number of data points to return per page. The
            default page_size is 1000. Values must be between 1 and
            10000. Values above 10000 are coerced to 10000.
        page_token (str):
            A page token, received from a previous
            [ExportTensorboardTimeSeriesData][google.cloud.aiplatform.v1.TensorboardService.ExportTensorboardTimeSeriesData]
            call. Provide this to retrieve the subsequent page.

            When paginating, all other parameters provided to
            [ExportTensorboardTimeSeriesData][google.cloud.aiplatform.v1.TensorboardService.ExportTensorboardTimeSeriesData]
            must match the call that provided the page token.
        order_by (str):
            Field to use to sort the
            TensorboardTimeSeries' data. By default,
            TensorboardTimeSeries' data is returned in a
            pseudo random order.
    """

    tensorboard_time_series: str = proto.Field(
        proto.STRING,
        number=1,
    )
    filter: str = proto.Field(
        proto.STRING,
        number=2,
    )
    page_size: int = proto.Field(
        proto.INT32,
        number=3,
    )
    page_token: str = proto.Field(
        proto.STRING,
        number=4,
    )
    order_by: str = proto.Field(
        proto.STRING,
        number=5,
    )


class ExportTensorboardTimeSeriesDataResponse(proto.Message):
    r"""Response message for
    [TensorboardService.ExportTensorboardTimeSeriesData][google.cloud.aiplatform.v1.TensorboardService.ExportTensorboardTimeSeriesData].

    Attributes:
        time_series_data_points (MutableSequence[google.cloud.aiplatform_v1.types.TimeSeriesDataPoint]):
            The returned time series data points.
        next_page_token (str):
            A token, which can be sent as
            [page_token][google.cloud.aiplatform.v1.ExportTensorboardTimeSeriesDataRequest.page_token]
            to retrieve the next page. If this field is omitted, there
            are no subsequent pages.
    """

    @property
    def raw_page(self):
        return self

    time_series_data_points: MutableSequence[
        tensorboard_data.TimeSeriesDataPoint
    ] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=tensorboard_data.TimeSeriesDataPoint,
    )
    next_page_token: str = proto.Field(
        proto.STRING,
        number=2,
    )


class CreateTensorboardOperationMetadata(proto.Message):
    r"""Details of operations that perform create Tensorboard.

    Attributes:
        generic_metadata (google.cloud.aiplatform_v1.types.GenericOperationMetadata):
            Operation metadata for Tensorboard.
    """

    generic_metadata: operation.GenericOperationMetadata = proto.Field(
        proto.MESSAGE,
        number=1,
        message=operation.GenericOperationMetadata,
    )


class UpdateTensorboardOperationMetadata(proto.Message):
    r"""Details of operations that perform update Tensorboard.

    Attributes:
        generic_metadata (google.cloud.aiplatform_v1.types.GenericOperationMetadata):
            Operation metadata for Tensorboard.
    """

    generic_metadata: operation.GenericOperationMetadata = proto.Field(
        proto.MESSAGE,
        number=1,
        message=operation.GenericOperationMetadata,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
