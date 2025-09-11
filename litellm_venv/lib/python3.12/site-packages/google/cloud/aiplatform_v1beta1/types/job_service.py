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

from google.cloud.aiplatform_v1beta1.types import (
    batch_prediction_job as gca_batch_prediction_job,
)
from google.cloud.aiplatform_v1beta1.types import custom_job as gca_custom_job
from google.cloud.aiplatform_v1beta1.types import (
    data_labeling_job as gca_data_labeling_job,
)
from google.cloud.aiplatform_v1beta1.types import (
    hyperparameter_tuning_job as gca_hyperparameter_tuning_job,
)
from google.cloud.aiplatform_v1beta1.types import (
    model_deployment_monitoring_job as gca_model_deployment_monitoring_job,
)
from google.cloud.aiplatform_v1beta1.types import nas_job as gca_nas_job
from google.cloud.aiplatform_v1beta1.types import operation
from google.protobuf import field_mask_pb2  # type: ignore
from google.protobuf import timestamp_pb2  # type: ignore


__protobuf__ = proto.module(
    package="google.cloud.aiplatform.v1beta1",
    manifest={
        "CreateCustomJobRequest",
        "GetCustomJobRequest",
        "ListCustomJobsRequest",
        "ListCustomJobsResponse",
        "DeleteCustomJobRequest",
        "CancelCustomJobRequest",
        "CreateDataLabelingJobRequest",
        "GetDataLabelingJobRequest",
        "ListDataLabelingJobsRequest",
        "ListDataLabelingJobsResponse",
        "DeleteDataLabelingJobRequest",
        "CancelDataLabelingJobRequest",
        "CreateHyperparameterTuningJobRequest",
        "GetHyperparameterTuningJobRequest",
        "ListHyperparameterTuningJobsRequest",
        "ListHyperparameterTuningJobsResponse",
        "DeleteHyperparameterTuningJobRequest",
        "CancelHyperparameterTuningJobRequest",
        "CreateNasJobRequest",
        "GetNasJobRequest",
        "ListNasJobsRequest",
        "ListNasJobsResponse",
        "DeleteNasJobRequest",
        "CancelNasJobRequest",
        "GetNasTrialDetailRequest",
        "ListNasTrialDetailsRequest",
        "ListNasTrialDetailsResponse",
        "CreateBatchPredictionJobRequest",
        "GetBatchPredictionJobRequest",
        "ListBatchPredictionJobsRequest",
        "ListBatchPredictionJobsResponse",
        "DeleteBatchPredictionJobRequest",
        "CancelBatchPredictionJobRequest",
        "CreateModelDeploymentMonitoringJobRequest",
        "SearchModelDeploymentMonitoringStatsAnomaliesRequest",
        "SearchModelDeploymentMonitoringStatsAnomaliesResponse",
        "GetModelDeploymentMonitoringJobRequest",
        "ListModelDeploymentMonitoringJobsRequest",
        "ListModelDeploymentMonitoringJobsResponse",
        "UpdateModelDeploymentMonitoringJobRequest",
        "DeleteModelDeploymentMonitoringJobRequest",
        "PauseModelDeploymentMonitoringJobRequest",
        "ResumeModelDeploymentMonitoringJobRequest",
        "UpdateModelDeploymentMonitoringJobOperationMetadata",
    },
)


class CreateCustomJobRequest(proto.Message):
    r"""Request message for
    [JobService.CreateCustomJob][google.cloud.aiplatform.v1beta1.JobService.CreateCustomJob].

    Attributes:
        parent (str):
            Required. The resource name of the Location to create the
            CustomJob in. Format:
            ``projects/{project}/locations/{location}``
        custom_job (google.cloud.aiplatform_v1beta1.types.CustomJob):
            Required. The CustomJob to create.
    """

    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )
    custom_job: gca_custom_job.CustomJob = proto.Field(
        proto.MESSAGE,
        number=2,
        message=gca_custom_job.CustomJob,
    )


class GetCustomJobRequest(proto.Message):
    r"""Request message for
    [JobService.GetCustomJob][google.cloud.aiplatform.v1beta1.JobService.GetCustomJob].

    Attributes:
        name (str):
            Required. The name of the CustomJob resource. Format:
            ``projects/{project}/locations/{location}/customJobs/{custom_job}``
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class ListCustomJobsRequest(proto.Message):
    r"""Request message for
    [JobService.ListCustomJobs][google.cloud.aiplatform.v1beta1.JobService.ListCustomJobs].

    Attributes:
        parent (str):
            Required. The resource name of the Location to list the
            CustomJobs from. Format:
            ``projects/{project}/locations/{location}``
        filter (str):
            The standard list filter.

            Supported fields:

            -  ``display_name`` supports ``=``, ``!=`` comparisons, and
               ``:`` wildcard.
            -  ``state`` supports ``=``, ``!=`` comparisons.
            -  ``create_time`` supports ``=``, ``!=``,\ ``<``,
               ``<=``,\ ``>``, ``>=`` comparisons. ``create_time`` must
               be in RFC 3339 format.
            -  ``labels`` supports general map functions that is:
               ``labels.key=value`` - key:value equality \`labels.key:\*
               - key existence

            Some examples of using the filter are:

            -  ``state="JOB_STATE_SUCCEEDED" AND display_name:"my_job_*"``
            -  ``state!="JOB_STATE_FAILED" OR display_name="my_job"``
            -  ``NOT display_name="my_job"``
            -  ``create_time>"2021-05-18T00:00:00Z"``
            -  ``labels.keyA=valueA``
            -  ``labels.keyB:*``
        page_size (int):
            The standard list page size.
        page_token (str):
            The standard list page token. Typically obtained via
            [ListCustomJobsResponse.next_page_token][google.cloud.aiplatform.v1beta1.ListCustomJobsResponse.next_page_token]
            of the previous
            [JobService.ListCustomJobs][google.cloud.aiplatform.v1beta1.JobService.ListCustomJobs]
            call.
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
    read_mask: field_mask_pb2.FieldMask = proto.Field(
        proto.MESSAGE,
        number=5,
        message=field_mask_pb2.FieldMask,
    )


class ListCustomJobsResponse(proto.Message):
    r"""Response message for
    [JobService.ListCustomJobs][google.cloud.aiplatform.v1beta1.JobService.ListCustomJobs]

    Attributes:
        custom_jobs (MutableSequence[google.cloud.aiplatform_v1beta1.types.CustomJob]):
            List of CustomJobs in the requested page.
        next_page_token (str):
            A token to retrieve the next page of results. Pass to
            [ListCustomJobsRequest.page_token][google.cloud.aiplatform.v1beta1.ListCustomJobsRequest.page_token]
            to obtain that page.
    """

    @property
    def raw_page(self):
        return self

    custom_jobs: MutableSequence[gca_custom_job.CustomJob] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=gca_custom_job.CustomJob,
    )
    next_page_token: str = proto.Field(
        proto.STRING,
        number=2,
    )


class DeleteCustomJobRequest(proto.Message):
    r"""Request message for
    [JobService.DeleteCustomJob][google.cloud.aiplatform.v1beta1.JobService.DeleteCustomJob].

    Attributes:
        name (str):
            Required. The name of the CustomJob resource to be deleted.
            Format:
            ``projects/{project}/locations/{location}/customJobs/{custom_job}``
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class CancelCustomJobRequest(proto.Message):
    r"""Request message for
    [JobService.CancelCustomJob][google.cloud.aiplatform.v1beta1.JobService.CancelCustomJob].

    Attributes:
        name (str):
            Required. The name of the CustomJob to cancel. Format:
            ``projects/{project}/locations/{location}/customJobs/{custom_job}``
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class CreateDataLabelingJobRequest(proto.Message):
    r"""Request message for
    [JobService.CreateDataLabelingJob][google.cloud.aiplatform.v1beta1.JobService.CreateDataLabelingJob].

    Attributes:
        parent (str):
            Required. The parent of the DataLabelingJob. Format:
            ``projects/{project}/locations/{location}``
        data_labeling_job (google.cloud.aiplatform_v1beta1.types.DataLabelingJob):
            Required. The DataLabelingJob to create.
    """

    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )
    data_labeling_job: gca_data_labeling_job.DataLabelingJob = proto.Field(
        proto.MESSAGE,
        number=2,
        message=gca_data_labeling_job.DataLabelingJob,
    )


class GetDataLabelingJobRequest(proto.Message):
    r"""Request message for
    [JobService.GetDataLabelingJob][google.cloud.aiplatform.v1beta1.JobService.GetDataLabelingJob].

    Attributes:
        name (str):
            Required. The name of the DataLabelingJob. Format:
            ``projects/{project}/locations/{location}/dataLabelingJobs/{data_labeling_job}``
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class ListDataLabelingJobsRequest(proto.Message):
    r"""Request message for
    [JobService.ListDataLabelingJobs][google.cloud.aiplatform.v1beta1.JobService.ListDataLabelingJobs].

    Attributes:
        parent (str):
            Required. The parent of the DataLabelingJob. Format:
            ``projects/{project}/locations/{location}``
        filter (str):
            The standard list filter.

            Supported fields:

            -  ``display_name`` supports ``=``, ``!=`` comparisons, and
               ``:`` wildcard.
            -  ``state`` supports ``=``, ``!=`` comparisons.
            -  ``create_time`` supports ``=``, ``!=``,\ ``<``,
               ``<=``,\ ``>``, ``>=`` comparisons. ``create_time`` must
               be in RFC 3339 format.
            -  ``labels`` supports general map functions that is:
               ``labels.key=value`` - key:value equality \`labels.key:\*
               - key existence

            Some examples of using the filter are:

            -  ``state="JOB_STATE_SUCCEEDED" AND display_name:"my_job_*"``
            -  ``state!="JOB_STATE_FAILED" OR display_name="my_job"``
            -  ``NOT display_name="my_job"``
            -  ``create_time>"2021-05-18T00:00:00Z"``
            -  ``labels.keyA=valueA``
            -  ``labels.keyB:*``
        page_size (int):
            The standard list page size.
        page_token (str):
            The standard list page token.
        read_mask (google.protobuf.field_mask_pb2.FieldMask):
            Mask specifying which fields to read. FieldMask represents a
            set of symbolic field paths. For example, the mask can be
            ``paths: "name"``. The "name" here is a field in
            DataLabelingJob. If this field is not set, all fields of the
            DataLabelingJob are returned.
        order_by (str):
            A comma-separated list of fields to order by, sorted in
            ascending order by default. Use ``desc`` after a field name
            for descending.
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
    read_mask: field_mask_pb2.FieldMask = proto.Field(
        proto.MESSAGE,
        number=5,
        message=field_mask_pb2.FieldMask,
    )
    order_by: str = proto.Field(
        proto.STRING,
        number=6,
    )


class ListDataLabelingJobsResponse(proto.Message):
    r"""Response message for
    [JobService.ListDataLabelingJobs][google.cloud.aiplatform.v1beta1.JobService.ListDataLabelingJobs].

    Attributes:
        data_labeling_jobs (MutableSequence[google.cloud.aiplatform_v1beta1.types.DataLabelingJob]):
            A list of DataLabelingJobs that matches the
            specified filter in the request.
        next_page_token (str):
            The standard List next-page token.
    """

    @property
    def raw_page(self):
        return self

    data_labeling_jobs: MutableSequence[
        gca_data_labeling_job.DataLabelingJob
    ] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=gca_data_labeling_job.DataLabelingJob,
    )
    next_page_token: str = proto.Field(
        proto.STRING,
        number=2,
    )


class DeleteDataLabelingJobRequest(proto.Message):
    r"""Request message for
    [JobService.DeleteDataLabelingJob][google.cloud.aiplatform.v1beta1.JobService.DeleteDataLabelingJob].

    Attributes:
        name (str):
            Required. The name of the DataLabelingJob to be deleted.
            Format:
            ``projects/{project}/locations/{location}/dataLabelingJobs/{data_labeling_job}``
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class CancelDataLabelingJobRequest(proto.Message):
    r"""Request message for
    [JobService.CancelDataLabelingJob][google.cloud.aiplatform.v1beta1.JobService.CancelDataLabelingJob].

    Attributes:
        name (str):
            Required. The name of the DataLabelingJob. Format:
            ``projects/{project}/locations/{location}/dataLabelingJobs/{data_labeling_job}``
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class CreateHyperparameterTuningJobRequest(proto.Message):
    r"""Request message for
    [JobService.CreateHyperparameterTuningJob][google.cloud.aiplatform.v1beta1.JobService.CreateHyperparameterTuningJob].

    Attributes:
        parent (str):
            Required. The resource name of the Location to create the
            HyperparameterTuningJob in. Format:
            ``projects/{project}/locations/{location}``
        hyperparameter_tuning_job (google.cloud.aiplatform_v1beta1.types.HyperparameterTuningJob):
            Required. The HyperparameterTuningJob to
            create.
    """

    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )
    hyperparameter_tuning_job: gca_hyperparameter_tuning_job.HyperparameterTuningJob = (
        proto.Field(
            proto.MESSAGE,
            number=2,
            message=gca_hyperparameter_tuning_job.HyperparameterTuningJob,
        )
    )


class GetHyperparameterTuningJobRequest(proto.Message):
    r"""Request message for
    [JobService.GetHyperparameterTuningJob][google.cloud.aiplatform.v1beta1.JobService.GetHyperparameterTuningJob].

    Attributes:
        name (str):
            Required. The name of the HyperparameterTuningJob resource.
            Format:
            ``projects/{project}/locations/{location}/hyperparameterTuningJobs/{hyperparameter_tuning_job}``
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class ListHyperparameterTuningJobsRequest(proto.Message):
    r"""Request message for
    [JobService.ListHyperparameterTuningJobs][google.cloud.aiplatform.v1beta1.JobService.ListHyperparameterTuningJobs].

    Attributes:
        parent (str):
            Required. The resource name of the Location to list the
            HyperparameterTuningJobs from. Format:
            ``projects/{project}/locations/{location}``
        filter (str):
            The standard list filter.

            Supported fields:

            -  ``display_name`` supports ``=``, ``!=`` comparisons, and
               ``:`` wildcard.
            -  ``state`` supports ``=``, ``!=`` comparisons.
            -  ``create_time`` supports ``=``, ``!=``,\ ``<``,
               ``<=``,\ ``>``, ``>=`` comparisons. ``create_time`` must
               be in RFC 3339 format.
            -  ``labels`` supports general map functions that is:
               ``labels.key=value`` - key:value equality \`labels.key:\*
               - key existence

            Some examples of using the filter are:

            -  ``state="JOB_STATE_SUCCEEDED" AND display_name:"my_job_*"``
            -  ``state!="JOB_STATE_FAILED" OR display_name="my_job"``
            -  ``NOT display_name="my_job"``
            -  ``create_time>"2021-05-18T00:00:00Z"``
            -  ``labels.keyA=valueA``
            -  ``labels.keyB:*``
        page_size (int):
            The standard list page size.
        page_token (str):
            The standard list page token. Typically obtained via
            [ListHyperparameterTuningJobsResponse.next_page_token][google.cloud.aiplatform.v1beta1.ListHyperparameterTuningJobsResponse.next_page_token]
            of the previous
            [JobService.ListHyperparameterTuningJobs][google.cloud.aiplatform.v1beta1.JobService.ListHyperparameterTuningJobs]
            call.
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
    read_mask: field_mask_pb2.FieldMask = proto.Field(
        proto.MESSAGE,
        number=5,
        message=field_mask_pb2.FieldMask,
    )


class ListHyperparameterTuningJobsResponse(proto.Message):
    r"""Response message for
    [JobService.ListHyperparameterTuningJobs][google.cloud.aiplatform.v1beta1.JobService.ListHyperparameterTuningJobs]

    Attributes:
        hyperparameter_tuning_jobs (MutableSequence[google.cloud.aiplatform_v1beta1.types.HyperparameterTuningJob]):
            List of HyperparameterTuningJobs in the requested page.
            [HyperparameterTuningJob.trials][google.cloud.aiplatform.v1beta1.HyperparameterTuningJob.trials]
            of the jobs will be not be returned.
        next_page_token (str):
            A token to retrieve the next page of results. Pass to
            [ListHyperparameterTuningJobsRequest.page_token][google.cloud.aiplatform.v1beta1.ListHyperparameterTuningJobsRequest.page_token]
            to obtain that page.
    """

    @property
    def raw_page(self):
        return self

    hyperparameter_tuning_jobs: MutableSequence[
        gca_hyperparameter_tuning_job.HyperparameterTuningJob
    ] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=gca_hyperparameter_tuning_job.HyperparameterTuningJob,
    )
    next_page_token: str = proto.Field(
        proto.STRING,
        number=2,
    )


class DeleteHyperparameterTuningJobRequest(proto.Message):
    r"""Request message for
    [JobService.DeleteHyperparameterTuningJob][google.cloud.aiplatform.v1beta1.JobService.DeleteHyperparameterTuningJob].

    Attributes:
        name (str):
            Required. The name of the HyperparameterTuningJob resource
            to be deleted. Format:
            ``projects/{project}/locations/{location}/hyperparameterTuningJobs/{hyperparameter_tuning_job}``
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class CancelHyperparameterTuningJobRequest(proto.Message):
    r"""Request message for
    [JobService.CancelHyperparameterTuningJob][google.cloud.aiplatform.v1beta1.JobService.CancelHyperparameterTuningJob].

    Attributes:
        name (str):
            Required. The name of the HyperparameterTuningJob to cancel.
            Format:
            ``projects/{project}/locations/{location}/hyperparameterTuningJobs/{hyperparameter_tuning_job}``
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class CreateNasJobRequest(proto.Message):
    r"""Request message for
    [JobService.CreateNasJob][google.cloud.aiplatform.v1beta1.JobService.CreateNasJob].

    Attributes:
        parent (str):
            Required. The resource name of the Location to create the
            NasJob in. Format:
            ``projects/{project}/locations/{location}``
        nas_job (google.cloud.aiplatform_v1beta1.types.NasJob):
            Required. The NasJob to create.
    """

    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )
    nas_job: gca_nas_job.NasJob = proto.Field(
        proto.MESSAGE,
        number=2,
        message=gca_nas_job.NasJob,
    )


class GetNasJobRequest(proto.Message):
    r"""Request message for
    [JobService.GetNasJob][google.cloud.aiplatform.v1beta1.JobService.GetNasJob].

    Attributes:
        name (str):
            Required. The name of the NasJob resource. Format:
            ``projects/{project}/locations/{location}/nasJobs/{nas_job}``
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class ListNasJobsRequest(proto.Message):
    r"""Request message for
    [JobService.ListNasJobs][google.cloud.aiplatform.v1beta1.JobService.ListNasJobs].

    Attributes:
        parent (str):
            Required. The resource name of the Location to list the
            NasJobs from. Format:
            ``projects/{project}/locations/{location}``
        filter (str):
            The standard list filter.

            Supported fields:

            -  ``display_name`` supports ``=``, ``!=`` comparisons, and
               ``:`` wildcard.
            -  ``state`` supports ``=``, ``!=`` comparisons.
            -  ``create_time`` supports ``=``, ``!=``,\ ``<``,
               ``<=``,\ ``>``, ``>=`` comparisons. ``create_time`` must
               be in RFC 3339 format.
            -  ``labels`` supports general map functions that is:
               ``labels.key=value`` - key:value equality \`labels.key:\*
               - key existence

            Some examples of using the filter are:

            -  ``state="JOB_STATE_SUCCEEDED" AND display_name:"my_job_*"``
            -  ``state!="JOB_STATE_FAILED" OR display_name="my_job"``
            -  ``NOT display_name="my_job"``
            -  ``create_time>"2021-05-18T00:00:00Z"``
            -  ``labels.keyA=valueA``
            -  ``labels.keyB:*``
        page_size (int):
            The standard list page size.
        page_token (str):
            The standard list page token. Typically obtained via
            [ListNasJobsResponse.next_page_token][google.cloud.aiplatform.v1beta1.ListNasJobsResponse.next_page_token]
            of the previous
            [JobService.ListNasJobs][google.cloud.aiplatform.v1beta1.JobService.ListNasJobs]
            call.
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
    read_mask: field_mask_pb2.FieldMask = proto.Field(
        proto.MESSAGE,
        number=5,
        message=field_mask_pb2.FieldMask,
    )


class ListNasJobsResponse(proto.Message):
    r"""Response message for
    [JobService.ListNasJobs][google.cloud.aiplatform.v1beta1.JobService.ListNasJobs]

    Attributes:
        nas_jobs (MutableSequence[google.cloud.aiplatform_v1beta1.types.NasJob]):
            List of NasJobs in the requested page.
            [NasJob.nas_job_output][google.cloud.aiplatform.v1beta1.NasJob.nas_job_output]
            of the jobs will not be returned.
        next_page_token (str):
            A token to retrieve the next page of results. Pass to
            [ListNasJobsRequest.page_token][google.cloud.aiplatform.v1beta1.ListNasJobsRequest.page_token]
            to obtain that page.
    """

    @property
    def raw_page(self):
        return self

    nas_jobs: MutableSequence[gca_nas_job.NasJob] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=gca_nas_job.NasJob,
    )
    next_page_token: str = proto.Field(
        proto.STRING,
        number=2,
    )


class DeleteNasJobRequest(proto.Message):
    r"""Request message for
    [JobService.DeleteNasJob][google.cloud.aiplatform.v1beta1.JobService.DeleteNasJob].

    Attributes:
        name (str):
            Required. The name of the NasJob resource to be deleted.
            Format:
            ``projects/{project}/locations/{location}/nasJobs/{nas_job}``
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class CancelNasJobRequest(proto.Message):
    r"""Request message for
    [JobService.CancelNasJob][google.cloud.aiplatform.v1beta1.JobService.CancelNasJob].

    Attributes:
        name (str):
            Required. The name of the NasJob to cancel. Format:
            ``projects/{project}/locations/{location}/nasJobs/{nas_job}``
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class GetNasTrialDetailRequest(proto.Message):
    r"""Request message for
    [JobService.GetNasTrialDetail][google.cloud.aiplatform.v1beta1.JobService.GetNasTrialDetail].

    Attributes:
        name (str):
            Required. The name of the NasTrialDetail resource. Format:
            ``projects/{project}/locations/{location}/nasJobs/{nas_job}/nasTrialDetails/{nas_trial_detail}``
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class ListNasTrialDetailsRequest(proto.Message):
    r"""Request message for
    [JobService.ListNasTrialDetails][google.cloud.aiplatform.v1beta1.JobService.ListNasTrialDetails].

    Attributes:
        parent (str):
            Required. The name of the NasJob resource. Format:
            ``projects/{project}/locations/{location}/nasJobs/{nas_job}``
        page_size (int):
            The standard list page size.
        page_token (str):
            The standard list page token. Typically obtained via
            [ListNasTrialDetailsResponse.next_page_token][google.cloud.aiplatform.v1beta1.ListNasTrialDetailsResponse.next_page_token]
            of the previous
            [JobService.ListNasTrialDetails][google.cloud.aiplatform.v1beta1.JobService.ListNasTrialDetails]
            call.
    """

    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )
    page_size: int = proto.Field(
        proto.INT32,
        number=2,
    )
    page_token: str = proto.Field(
        proto.STRING,
        number=3,
    )


class ListNasTrialDetailsResponse(proto.Message):
    r"""Response message for
    [JobService.ListNasTrialDetails][google.cloud.aiplatform.v1beta1.JobService.ListNasTrialDetails]

    Attributes:
        nas_trial_details (MutableSequence[google.cloud.aiplatform_v1beta1.types.NasTrialDetail]):
            List of top NasTrials in the requested page.
        next_page_token (str):
            A token to retrieve the next page of results. Pass to
            [ListNasTrialDetailsRequest.page_token][google.cloud.aiplatform.v1beta1.ListNasTrialDetailsRequest.page_token]
            to obtain that page.
    """

    @property
    def raw_page(self):
        return self

    nas_trial_details: MutableSequence[
        gca_nas_job.NasTrialDetail
    ] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=gca_nas_job.NasTrialDetail,
    )
    next_page_token: str = proto.Field(
        proto.STRING,
        number=2,
    )


class CreateBatchPredictionJobRequest(proto.Message):
    r"""Request message for
    [JobService.CreateBatchPredictionJob][google.cloud.aiplatform.v1beta1.JobService.CreateBatchPredictionJob].

    Attributes:
        parent (str):
            Required. The resource name of the Location to create the
            BatchPredictionJob in. Format:
            ``projects/{project}/locations/{location}``
        batch_prediction_job (google.cloud.aiplatform_v1beta1.types.BatchPredictionJob):
            Required. The BatchPredictionJob to create.
    """

    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )
    batch_prediction_job: gca_batch_prediction_job.BatchPredictionJob = proto.Field(
        proto.MESSAGE,
        number=2,
        message=gca_batch_prediction_job.BatchPredictionJob,
    )


class GetBatchPredictionJobRequest(proto.Message):
    r"""Request message for
    [JobService.GetBatchPredictionJob][google.cloud.aiplatform.v1beta1.JobService.GetBatchPredictionJob].

    Attributes:
        name (str):
            Required. The name of the BatchPredictionJob resource.
            Format:
            ``projects/{project}/locations/{location}/batchPredictionJobs/{batch_prediction_job}``
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class ListBatchPredictionJobsRequest(proto.Message):
    r"""Request message for
    [JobService.ListBatchPredictionJobs][google.cloud.aiplatform.v1beta1.JobService.ListBatchPredictionJobs].

    Attributes:
        parent (str):
            Required. The resource name of the Location to list the
            BatchPredictionJobs from. Format:
            ``projects/{project}/locations/{location}``
        filter (str):
            The standard list filter.

            Supported fields:

            -  ``display_name`` supports ``=``, ``!=`` comparisons, and
               ``:`` wildcard.
            -  ``model_display_name`` supports ``=``, ``!=``
               comparisons.
            -  ``state`` supports ``=``, ``!=`` comparisons.
            -  ``create_time`` supports ``=``, ``!=``,\ ``<``,
               ``<=``,\ ``>``, ``>=`` comparisons. ``create_time`` must
               be in RFC 3339 format.
            -  ``labels`` supports general map functions that is:
               ``labels.key=value`` - key:value equality \`labels.key:\*
               - key existence

            Some examples of using the filter are:

            -  ``state="JOB_STATE_SUCCEEDED" AND display_name:"my_job_*"``
            -  ``state!="JOB_STATE_FAILED" OR display_name="my_job"``
            -  ``NOT display_name="my_job"``
            -  ``create_time>"2021-05-18T00:00:00Z"``
            -  ``labels.keyA=valueA``
            -  ``labels.keyB:*``
        page_size (int):
            The standard list page size.
        page_token (str):
            The standard list page token. Typically obtained via
            [ListBatchPredictionJobsResponse.next_page_token][google.cloud.aiplatform.v1beta1.ListBatchPredictionJobsResponse.next_page_token]
            of the previous
            [JobService.ListBatchPredictionJobs][google.cloud.aiplatform.v1beta1.JobService.ListBatchPredictionJobs]
            call.
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
    read_mask: field_mask_pb2.FieldMask = proto.Field(
        proto.MESSAGE,
        number=5,
        message=field_mask_pb2.FieldMask,
    )


class ListBatchPredictionJobsResponse(proto.Message):
    r"""Response message for
    [JobService.ListBatchPredictionJobs][google.cloud.aiplatform.v1beta1.JobService.ListBatchPredictionJobs]

    Attributes:
        batch_prediction_jobs (MutableSequence[google.cloud.aiplatform_v1beta1.types.BatchPredictionJob]):
            List of BatchPredictionJobs in the requested
            page.
        next_page_token (str):
            A token to retrieve the next page of results. Pass to
            [ListBatchPredictionJobsRequest.page_token][google.cloud.aiplatform.v1beta1.ListBatchPredictionJobsRequest.page_token]
            to obtain that page.
    """

    @property
    def raw_page(self):
        return self

    batch_prediction_jobs: MutableSequence[
        gca_batch_prediction_job.BatchPredictionJob
    ] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=gca_batch_prediction_job.BatchPredictionJob,
    )
    next_page_token: str = proto.Field(
        proto.STRING,
        number=2,
    )


class DeleteBatchPredictionJobRequest(proto.Message):
    r"""Request message for
    [JobService.DeleteBatchPredictionJob][google.cloud.aiplatform.v1beta1.JobService.DeleteBatchPredictionJob].

    Attributes:
        name (str):
            Required. The name of the BatchPredictionJob resource to be
            deleted. Format:
            ``projects/{project}/locations/{location}/batchPredictionJobs/{batch_prediction_job}``
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class CancelBatchPredictionJobRequest(proto.Message):
    r"""Request message for
    [JobService.CancelBatchPredictionJob][google.cloud.aiplatform.v1beta1.JobService.CancelBatchPredictionJob].

    Attributes:
        name (str):
            Required. The name of the BatchPredictionJob to cancel.
            Format:
            ``projects/{project}/locations/{location}/batchPredictionJobs/{batch_prediction_job}``
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class CreateModelDeploymentMonitoringJobRequest(proto.Message):
    r"""Request message for
    [JobService.CreateModelDeploymentMonitoringJob][google.cloud.aiplatform.v1beta1.JobService.CreateModelDeploymentMonitoringJob].

    Attributes:
        parent (str):
            Required. The parent of the ModelDeploymentMonitoringJob.
            Format: ``projects/{project}/locations/{location}``
        model_deployment_monitoring_job (google.cloud.aiplatform_v1beta1.types.ModelDeploymentMonitoringJob):
            Required. The ModelDeploymentMonitoringJob to
            create
    """

    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )
    model_deployment_monitoring_job: gca_model_deployment_monitoring_job.ModelDeploymentMonitoringJob = proto.Field(
        proto.MESSAGE,
        number=2,
        message=gca_model_deployment_monitoring_job.ModelDeploymentMonitoringJob,
    )


class SearchModelDeploymentMonitoringStatsAnomaliesRequest(proto.Message):
    r"""Request message for
    [JobService.SearchModelDeploymentMonitoringStatsAnomalies][google.cloud.aiplatform.v1beta1.JobService.SearchModelDeploymentMonitoringStatsAnomalies].

    Attributes:
        model_deployment_monitoring_job (str):
            Required. ModelDeploymentMonitoring Job resource name.
            Format:
            ``projects/{project}/locations/{location}/modelDeploymentMonitoringJobs/{model_deployment_monitoring_job}``
        deployed_model_id (str):
            Required. The DeployedModel ID of the
            [ModelDeploymentMonitoringObjectiveConfig.deployed_model_id].
        feature_display_name (str):
            The feature display name. If specified, only return the
            stats belonging to this feature. Format:
            [ModelMonitoringStatsAnomalies.FeatureHistoricStatsAnomalies.feature_display_name][google.cloud.aiplatform.v1beta1.ModelMonitoringStatsAnomalies.FeatureHistoricStatsAnomalies.feature_display_name],
            example: "user_destination".
        objectives (MutableSequence[google.cloud.aiplatform_v1beta1.types.SearchModelDeploymentMonitoringStatsAnomaliesRequest.StatsAnomaliesObjective]):
            Required. Objectives of the stats to
            retrieve.
        page_size (int):
            The standard list page size.
        page_token (str):
            A page token received from a previous
            [JobService.SearchModelDeploymentMonitoringStatsAnomalies][google.cloud.aiplatform.v1beta1.JobService.SearchModelDeploymentMonitoringStatsAnomalies]
            call.
        start_time (google.protobuf.timestamp_pb2.Timestamp):
            The earliest timestamp of stats being
            generated. If not set, indicates fetching stats
            till the earliest possible one.
        end_time (google.protobuf.timestamp_pb2.Timestamp):
            The latest timestamp of stats being
            generated. If not set, indicates feching stats
            till the latest possible one.
    """

    class StatsAnomaliesObjective(proto.Message):
        r"""Stats requested for specific objective.

        Attributes:
            type_ (google.cloud.aiplatform_v1beta1.types.ModelDeploymentMonitoringObjectiveType):

            top_feature_count (int):
                If set, all attribution scores between
                [SearchModelDeploymentMonitoringStatsAnomaliesRequest.start_time][google.cloud.aiplatform.v1beta1.SearchModelDeploymentMonitoringStatsAnomaliesRequest.start_time]
                and
                [SearchModelDeploymentMonitoringStatsAnomaliesRequest.end_time][google.cloud.aiplatform.v1beta1.SearchModelDeploymentMonitoringStatsAnomaliesRequest.end_time]
                are fetched, and page token doesn't take effect in this
                case. Only used to retrieve attribution score for the top
                Features which has the highest attribution score in the
                latest monitoring run.
        """

        type_: gca_model_deployment_monitoring_job.ModelDeploymentMonitoringObjectiveType = proto.Field(
            proto.ENUM,
            number=1,
            enum=gca_model_deployment_monitoring_job.ModelDeploymentMonitoringObjectiveType,
        )
        top_feature_count: int = proto.Field(
            proto.INT32,
            number=4,
        )

    model_deployment_monitoring_job: str = proto.Field(
        proto.STRING,
        number=1,
    )
    deployed_model_id: str = proto.Field(
        proto.STRING,
        number=2,
    )
    feature_display_name: str = proto.Field(
        proto.STRING,
        number=3,
    )
    objectives: MutableSequence[StatsAnomaliesObjective] = proto.RepeatedField(
        proto.MESSAGE,
        number=4,
        message=StatsAnomaliesObjective,
    )
    page_size: int = proto.Field(
        proto.INT32,
        number=5,
    )
    page_token: str = proto.Field(
        proto.STRING,
        number=6,
    )
    start_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=7,
        message=timestamp_pb2.Timestamp,
    )
    end_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=8,
        message=timestamp_pb2.Timestamp,
    )


class SearchModelDeploymentMonitoringStatsAnomaliesResponse(proto.Message):
    r"""Response message for
    [JobService.SearchModelDeploymentMonitoringStatsAnomalies][google.cloud.aiplatform.v1beta1.JobService.SearchModelDeploymentMonitoringStatsAnomalies].

    Attributes:
        monitoring_stats (MutableSequence[google.cloud.aiplatform_v1beta1.types.ModelMonitoringStatsAnomalies]):
            Stats retrieved for requested objectives. There are at most
            1000
            [ModelMonitoringStatsAnomalies.FeatureHistoricStatsAnomalies.prediction_stats][google.cloud.aiplatform.v1beta1.ModelMonitoringStatsAnomalies.FeatureHistoricStatsAnomalies.prediction_stats]
            in the response.
        next_page_token (str):
            The page token that can be used by the next
            [JobService.SearchModelDeploymentMonitoringStatsAnomalies][google.cloud.aiplatform.v1beta1.JobService.SearchModelDeploymentMonitoringStatsAnomalies]
            call.
    """

    @property
    def raw_page(self):
        return self

    monitoring_stats: MutableSequence[
        gca_model_deployment_monitoring_job.ModelMonitoringStatsAnomalies
    ] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=gca_model_deployment_monitoring_job.ModelMonitoringStatsAnomalies,
    )
    next_page_token: str = proto.Field(
        proto.STRING,
        number=2,
    )


class GetModelDeploymentMonitoringJobRequest(proto.Message):
    r"""Request message for
    [JobService.GetModelDeploymentMonitoringJob][google.cloud.aiplatform.v1beta1.JobService.GetModelDeploymentMonitoringJob].

    Attributes:
        name (str):
            Required. The resource name of the
            ModelDeploymentMonitoringJob. Format:
            ``projects/{project}/locations/{location}/modelDeploymentMonitoringJobs/{model_deployment_monitoring_job}``
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class ListModelDeploymentMonitoringJobsRequest(proto.Message):
    r"""Request message for
    [JobService.ListModelDeploymentMonitoringJobs][google.cloud.aiplatform.v1beta1.JobService.ListModelDeploymentMonitoringJobs].

    Attributes:
        parent (str):
            Required. The parent of the ModelDeploymentMonitoringJob.
            Format: ``projects/{project}/locations/{location}``
        filter (str):
            The standard list filter.

            Supported fields:

            -  ``display_name`` supports ``=``, ``!=`` comparisons, and
               ``:`` wildcard.
            -  ``state`` supports ``=``, ``!=`` comparisons.
            -  ``create_time`` supports ``=``, ``!=``,\ ``<``,
               ``<=``,\ ``>``, ``>=`` comparisons. ``create_time`` must
               be in RFC 3339 format.
            -  ``labels`` supports general map functions that is:
               ``labels.key=value`` - key:value equality \`labels.key:\*
               - key existence

            Some examples of using the filter are:

            -  ``state="JOB_STATE_SUCCEEDED" AND display_name:"my_job_*"``
            -  ``state!="JOB_STATE_FAILED" OR display_name="my_job"``
            -  ``NOT display_name="my_job"``
            -  ``create_time>"2021-05-18T00:00:00Z"``
            -  ``labels.keyA=valueA``
            -  ``labels.keyB:*``
        page_size (int):
            The standard list page size.
        page_token (str):
            The standard list page token.
        read_mask (google.protobuf.field_mask_pb2.FieldMask):
            Mask specifying which fields to read
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
    read_mask: field_mask_pb2.FieldMask = proto.Field(
        proto.MESSAGE,
        number=5,
        message=field_mask_pb2.FieldMask,
    )


class ListModelDeploymentMonitoringJobsResponse(proto.Message):
    r"""Response message for
    [JobService.ListModelDeploymentMonitoringJobs][google.cloud.aiplatform.v1beta1.JobService.ListModelDeploymentMonitoringJobs].

    Attributes:
        model_deployment_monitoring_jobs (MutableSequence[google.cloud.aiplatform_v1beta1.types.ModelDeploymentMonitoringJob]):
            A list of ModelDeploymentMonitoringJobs that
            matches the specified filter in the request.
        next_page_token (str):
            The standard List next-page token.
    """

    @property
    def raw_page(self):
        return self

    model_deployment_monitoring_jobs: MutableSequence[
        gca_model_deployment_monitoring_job.ModelDeploymentMonitoringJob
    ] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=gca_model_deployment_monitoring_job.ModelDeploymentMonitoringJob,
    )
    next_page_token: str = proto.Field(
        proto.STRING,
        number=2,
    )


class UpdateModelDeploymentMonitoringJobRequest(proto.Message):
    r"""Request message for
    [JobService.UpdateModelDeploymentMonitoringJob][google.cloud.aiplatform.v1beta1.JobService.UpdateModelDeploymentMonitoringJob].

    Attributes:
        model_deployment_monitoring_job (google.cloud.aiplatform_v1beta1.types.ModelDeploymentMonitoringJob):
            Required. The model monitoring configuration
            which replaces the resource on the server.
        update_mask (google.protobuf.field_mask_pb2.FieldMask):
            Required. The update mask is used to specify the fields to
            be overwritten in the ModelDeploymentMonitoringJob resource
            by the update. The fields specified in the update_mask are
            relative to the resource, not the full request. A field will
            be overwritten if it is in the mask. If the user does not
            provide a mask then only the non-empty fields present in the
            request will be overwritten. Set the update_mask to ``*`` to
            override all fields. For the objective config, the user can
            either provide the update mask for
            model_deployment_monitoring_objective_configs or any
            combination of its nested fields, such as:
            model_deployment_monitoring_objective_configs.objective_config.training_dataset.

            Updatable fields:

            -  ``display_name``
            -  ``model_deployment_monitoring_schedule_config``
            -  ``model_monitoring_alert_config``
            -  ``logging_sampling_strategy``
            -  ``labels``
            -  ``log_ttl``
            -  ``enable_monitoring_pipeline_logs`` . and
            -  ``model_deployment_monitoring_objective_configs`` . or
            -  ``model_deployment_monitoring_objective_configs.objective_config.training_dataset``
            -  ``model_deployment_monitoring_objective_configs.objective_config.training_prediction_skew_detection_config``
            -  ``model_deployment_monitoring_objective_configs.objective_config.prediction_drift_detection_config``
    """

    model_deployment_monitoring_job: gca_model_deployment_monitoring_job.ModelDeploymentMonitoringJob = proto.Field(
        proto.MESSAGE,
        number=1,
        message=gca_model_deployment_monitoring_job.ModelDeploymentMonitoringJob,
    )
    update_mask: field_mask_pb2.FieldMask = proto.Field(
        proto.MESSAGE,
        number=2,
        message=field_mask_pb2.FieldMask,
    )


class DeleteModelDeploymentMonitoringJobRequest(proto.Message):
    r"""Request message for
    [JobService.DeleteModelDeploymentMonitoringJob][google.cloud.aiplatform.v1beta1.JobService.DeleteModelDeploymentMonitoringJob].

    Attributes:
        name (str):
            Required. The resource name of the model monitoring job to
            delete. Format:
            ``projects/{project}/locations/{location}/modelDeploymentMonitoringJobs/{model_deployment_monitoring_job}``
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class PauseModelDeploymentMonitoringJobRequest(proto.Message):
    r"""Request message for
    [JobService.PauseModelDeploymentMonitoringJob][google.cloud.aiplatform.v1beta1.JobService.PauseModelDeploymentMonitoringJob].

    Attributes:
        name (str):
            Required. The resource name of the
            ModelDeploymentMonitoringJob to pause. Format:
            ``projects/{project}/locations/{location}/modelDeploymentMonitoringJobs/{model_deployment_monitoring_job}``
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class ResumeModelDeploymentMonitoringJobRequest(proto.Message):
    r"""Request message for
    [JobService.ResumeModelDeploymentMonitoringJob][google.cloud.aiplatform.v1beta1.JobService.ResumeModelDeploymentMonitoringJob].

    Attributes:
        name (str):
            Required. The resource name of the
            ModelDeploymentMonitoringJob to resume. Format:
            ``projects/{project}/locations/{location}/modelDeploymentMonitoringJobs/{model_deployment_monitoring_job}``
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class UpdateModelDeploymentMonitoringJobOperationMetadata(proto.Message):
    r"""Runtime operation information for
    [JobService.UpdateModelDeploymentMonitoringJob][google.cloud.aiplatform.v1beta1.JobService.UpdateModelDeploymentMonitoringJob].

    Attributes:
        generic_metadata (google.cloud.aiplatform_v1beta1.types.GenericOperationMetadata):
            The operation generic information.
    """

    generic_metadata: operation.GenericOperationMetadata = proto.Field(
        proto.MESSAGE,
        number=1,
        message=operation.GenericOperationMetadata,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
