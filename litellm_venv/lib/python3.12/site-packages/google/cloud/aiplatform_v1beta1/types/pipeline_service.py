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

from google.cloud.aiplatform_v1beta1.types import operation
from google.cloud.aiplatform_v1beta1.types import pipeline_job as gca_pipeline_job
from google.cloud.aiplatform_v1beta1.types import (
    training_pipeline as gca_training_pipeline,
)
from google.protobuf import field_mask_pb2  # type: ignore


__protobuf__ = proto.module(
    package="google.cloud.aiplatform.v1beta1",
    manifest={
        "BatchCancelPipelineJobsOperationMetadata",
        "CreateTrainingPipelineRequest",
        "GetTrainingPipelineRequest",
        "ListTrainingPipelinesRequest",
        "ListTrainingPipelinesResponse",
        "DeleteTrainingPipelineRequest",
        "CancelTrainingPipelineRequest",
        "CreatePipelineJobRequest",
        "GetPipelineJobRequest",
        "ListPipelineJobsRequest",
        "ListPipelineJobsResponse",
        "DeletePipelineJobRequest",
        "BatchDeletePipelineJobsRequest",
        "BatchDeletePipelineJobsResponse",
        "CancelPipelineJobRequest",
        "BatchCancelPipelineJobsRequest",
        "BatchCancelPipelineJobsResponse",
    },
)


class BatchCancelPipelineJobsOperationMetadata(proto.Message):
    r"""Runtime operation information for
    [PipelineService.BatchCancelPipelineJobs][google.cloud.aiplatform.v1beta1.PipelineService.BatchCancelPipelineJobs].

    Attributes:
        generic_metadata (google.cloud.aiplatform_v1beta1.types.GenericOperationMetadata):
            The common part of the operation metadata.
    """

    generic_metadata: operation.GenericOperationMetadata = proto.Field(
        proto.MESSAGE,
        number=1,
        message=operation.GenericOperationMetadata,
    )


class CreateTrainingPipelineRequest(proto.Message):
    r"""Request message for
    [PipelineService.CreateTrainingPipeline][google.cloud.aiplatform.v1beta1.PipelineService.CreateTrainingPipeline].

    Attributes:
        parent (str):
            Required. The resource name of the Location to create the
            TrainingPipeline in. Format:
            ``projects/{project}/locations/{location}``
        training_pipeline (google.cloud.aiplatform_v1beta1.types.TrainingPipeline):
            Required. The TrainingPipeline to create.
    """

    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )
    training_pipeline: gca_training_pipeline.TrainingPipeline = proto.Field(
        proto.MESSAGE,
        number=2,
        message=gca_training_pipeline.TrainingPipeline,
    )


class GetTrainingPipelineRequest(proto.Message):
    r"""Request message for
    [PipelineService.GetTrainingPipeline][google.cloud.aiplatform.v1beta1.PipelineService.GetTrainingPipeline].

    Attributes:
        name (str):
            Required. The name of the TrainingPipeline resource. Format:
            ``projects/{project}/locations/{location}/trainingPipelines/{training_pipeline}``
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class ListTrainingPipelinesRequest(proto.Message):
    r"""Request message for
    [PipelineService.ListTrainingPipelines][google.cloud.aiplatform.v1beta1.PipelineService.ListTrainingPipelines].

    Attributes:
        parent (str):
            Required. The resource name of the Location to list the
            TrainingPipelines from. Format:
            ``projects/{project}/locations/{location}``
        filter (str):
            The standard list filter.

            Supported fields:

            -  ``display_name`` supports ``=``, ``!=`` comparisons, and
               ``:`` wildcard.
            -  ``state`` supports ``=``, ``!=`` comparisons.
            -  ``training_task_definition`` ``=``, ``!=`` comparisons,
               and ``:`` wildcard.
            -  ``create_time`` supports ``=``, ``!=``,\ ``<``,
               ``<=``,\ ``>``, ``>=`` comparisons. ``create_time`` must
               be in RFC 3339 format.
            -  ``labels`` supports general map functions that is:
               ``labels.key=value`` - key:value equality \`labels.key:\*
               - key existence

            Some examples of using the filter are:

            -  ``state="PIPELINE_STATE_SUCCEEDED" AND display_name:"my_pipeline_*"``
            -  ``state!="PIPELINE_STATE_FAILED" OR display_name="my_pipeline"``
            -  ``NOT display_name="my_pipeline"``
            -  ``create_time>"2021-05-18T00:00:00Z"``
            -  ``training_task_definition:"*automl_text_classification*"``
        page_size (int):
            The standard list page size.
        page_token (str):
            The standard list page token. Typically obtained via
            [ListTrainingPipelinesResponse.next_page_token][google.cloud.aiplatform.v1beta1.ListTrainingPipelinesResponse.next_page_token]
            of the previous
            [PipelineService.ListTrainingPipelines][google.cloud.aiplatform.v1beta1.PipelineService.ListTrainingPipelines]
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


class ListTrainingPipelinesResponse(proto.Message):
    r"""Response message for
    [PipelineService.ListTrainingPipelines][google.cloud.aiplatform.v1beta1.PipelineService.ListTrainingPipelines]

    Attributes:
        training_pipelines (MutableSequence[google.cloud.aiplatform_v1beta1.types.TrainingPipeline]):
            List of TrainingPipelines in the requested
            page.
        next_page_token (str):
            A token to retrieve the next page of results. Pass to
            [ListTrainingPipelinesRequest.page_token][google.cloud.aiplatform.v1beta1.ListTrainingPipelinesRequest.page_token]
            to obtain that page.
    """

    @property
    def raw_page(self):
        return self

    training_pipelines: MutableSequence[
        gca_training_pipeline.TrainingPipeline
    ] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=gca_training_pipeline.TrainingPipeline,
    )
    next_page_token: str = proto.Field(
        proto.STRING,
        number=2,
    )


class DeleteTrainingPipelineRequest(proto.Message):
    r"""Request message for
    [PipelineService.DeleteTrainingPipeline][google.cloud.aiplatform.v1beta1.PipelineService.DeleteTrainingPipeline].

    Attributes:
        name (str):
            Required. The name of the TrainingPipeline resource to be
            deleted. Format:
            ``projects/{project}/locations/{location}/trainingPipelines/{training_pipeline}``
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class CancelTrainingPipelineRequest(proto.Message):
    r"""Request message for
    [PipelineService.CancelTrainingPipeline][google.cloud.aiplatform.v1beta1.PipelineService.CancelTrainingPipeline].

    Attributes:
        name (str):
            Required. The name of the TrainingPipeline to cancel.
            Format:
            ``projects/{project}/locations/{location}/trainingPipelines/{training_pipeline}``
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class CreatePipelineJobRequest(proto.Message):
    r"""Request message for
    [PipelineService.CreatePipelineJob][google.cloud.aiplatform.v1beta1.PipelineService.CreatePipelineJob].

    Attributes:
        parent (str):
            Required. The resource name of the Location to create the
            PipelineJob in. Format:
            ``projects/{project}/locations/{location}``
        pipeline_job (google.cloud.aiplatform_v1beta1.types.PipelineJob):
            Required. The PipelineJob to create.
        pipeline_job_id (str):
            The ID to use for the PipelineJob, which will become the
            final component of the PipelineJob name. If not provided, an
            ID will be automatically generated.

            This value should be less than 128 characters, and valid
            characters are ``/[a-z][0-9]-/``.
    """

    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )
    pipeline_job: gca_pipeline_job.PipelineJob = proto.Field(
        proto.MESSAGE,
        number=2,
        message=gca_pipeline_job.PipelineJob,
    )
    pipeline_job_id: str = proto.Field(
        proto.STRING,
        number=3,
    )


class GetPipelineJobRequest(proto.Message):
    r"""Request message for
    [PipelineService.GetPipelineJob][google.cloud.aiplatform.v1beta1.PipelineService.GetPipelineJob].

    Attributes:
        name (str):
            Required. The name of the PipelineJob resource. Format:
            ``projects/{project}/locations/{location}/pipelineJobs/{pipeline_job}``
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class ListPipelineJobsRequest(proto.Message):
    r"""Request message for
    [PipelineService.ListPipelineJobs][google.cloud.aiplatform.v1beta1.PipelineService.ListPipelineJobs].

    Attributes:
        parent (str):
            Required. The resource name of the Location to list the
            PipelineJobs from. Format:
            ``projects/{project}/locations/{location}``
        filter (str):
            Lists the PipelineJobs that match the filter expression. The
            following fields are supported:

            -  ``pipeline_name``: Supports ``=`` and ``!=`` comparisons.
            -  ``display_name``: Supports ``=``, ``!=`` comparisons, and
               ``:`` wildcard.
            -  ``pipeline_job_user_id``: Supports ``=``, ``!=``
               comparisons, and ``:`` wildcard. for example, can check
               if pipeline's display_name contains *step* by doing
               display_name:"*step*"
            -  ``state``: Supports ``=`` and ``!=`` comparisons.
            -  ``create_time``: Supports ``=``, ``!=``, ``<``, ``>``,
               ``<=``, and ``>=`` comparisons. Values must be in RFC
               3339 format.
            -  ``update_time``: Supports ``=``, ``!=``, ``<``, ``>``,
               ``<=``, and ``>=`` comparisons. Values must be in RFC
               3339 format.
            -  ``end_time``: Supports ``=``, ``!=``, ``<``, ``>``,
               ``<=``, and ``>=`` comparisons. Values must be in RFC
               3339 format.
            -  ``labels``: Supports key-value equality and key presence.
            -  ``template_uri``: Supports ``=``, ``!=`` comparisons, and
               ``:`` wildcard.
            -  ``template_metadata.version``: Supports ``=``, ``!=``
               comparisons, and ``:`` wildcard.

            Filter expressions can be combined together using logical
            operators (``AND`` & ``OR``). For example:
            ``pipeline_name="test" AND create_time>"2020-05-18T13:30:00Z"``.

            The syntax to define filter expression is based on
            https://google.aip.dev/160.

            Examples:

            -  ``create_time>"2021-05-18T00:00:00Z" OR update_time>"2020-05-18T00:00:00Z"``
               PipelineJobs created or updated after 2020-05-18 00:00:00
               UTC.
            -  ``labels.env = "prod"`` PipelineJobs with label "env" set
               to "prod".
        page_size (int):
            The standard list page size.
        page_token (str):
            The standard list page token. Typically obtained via
            [ListPipelineJobsResponse.next_page_token][google.cloud.aiplatform.v1beta1.ListPipelineJobsResponse.next_page_token]
            of the previous
            [PipelineService.ListPipelineJobs][google.cloud.aiplatform.v1beta1.PipelineService.ListPipelineJobs]
            call.
        order_by (str):
            A comma-separated list of fields to order by. The default
            sort order is in ascending order. Use "desc" after a field
            name for descending. You can have multiple order_by fields
            provided e.g. "create_time desc, end_time", "end_time,
            start_time, update_time" For example, using "create_time
            desc, end_time" will order results by create time in
            descending order, and if there are multiple jobs having the
            same create time, order them by the end time in ascending
            order. if order_by is not specified, it will order by
            default order is create time in descending order. Supported
            fields:

            -  ``create_time``
            -  ``update_time``
            -  ``end_time``
            -  ``start_time``
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
        number=6,
    )
    read_mask: field_mask_pb2.FieldMask = proto.Field(
        proto.MESSAGE,
        number=7,
        message=field_mask_pb2.FieldMask,
    )


class ListPipelineJobsResponse(proto.Message):
    r"""Response message for
    [PipelineService.ListPipelineJobs][google.cloud.aiplatform.v1beta1.PipelineService.ListPipelineJobs]

    Attributes:
        pipeline_jobs (MutableSequence[google.cloud.aiplatform_v1beta1.types.PipelineJob]):
            List of PipelineJobs in the requested page.
        next_page_token (str):
            A token to retrieve the next page of results. Pass to
            [ListPipelineJobsRequest.page_token][google.cloud.aiplatform.v1beta1.ListPipelineJobsRequest.page_token]
            to obtain that page.
    """

    @property
    def raw_page(self):
        return self

    pipeline_jobs: MutableSequence[gca_pipeline_job.PipelineJob] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=gca_pipeline_job.PipelineJob,
    )
    next_page_token: str = proto.Field(
        proto.STRING,
        number=2,
    )


class DeletePipelineJobRequest(proto.Message):
    r"""Request message for
    [PipelineService.DeletePipelineJob][google.cloud.aiplatform.v1beta1.PipelineService.DeletePipelineJob].

    Attributes:
        name (str):
            Required. The name of the PipelineJob resource to be
            deleted. Format:
            ``projects/{project}/locations/{location}/pipelineJobs/{pipeline_job}``
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class BatchDeletePipelineJobsRequest(proto.Message):
    r"""Request message for
    [PipelineService.BatchDeletePipelineJobs][google.cloud.aiplatform.v1beta1.PipelineService.BatchDeletePipelineJobs].

    Attributes:
        parent (str):
            Required. The name of the PipelineJobs' parent resource.
            Format: ``projects/{project}/locations/{location}``
        names (MutableSequence[str]):
            Required. The names of the PipelineJobs to delete. A maximum
            of 32 PipelineJobs can be deleted in a batch. Format:
            ``projects/{project}/locations/{location}/pipelineJobs/{pipelineJob}``
    """

    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )
    names: MutableSequence[str] = proto.RepeatedField(
        proto.STRING,
        number=2,
    )


class BatchDeletePipelineJobsResponse(proto.Message):
    r"""Response message for
    [PipelineService.BatchDeletePipelineJobs][google.cloud.aiplatform.v1beta1.PipelineService.BatchDeletePipelineJobs].

    Attributes:
        pipeline_jobs (MutableSequence[google.cloud.aiplatform_v1beta1.types.PipelineJob]):
            PipelineJobs deleted.
    """

    pipeline_jobs: MutableSequence[gca_pipeline_job.PipelineJob] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=gca_pipeline_job.PipelineJob,
    )


class CancelPipelineJobRequest(proto.Message):
    r"""Request message for
    [PipelineService.CancelPipelineJob][google.cloud.aiplatform.v1beta1.PipelineService.CancelPipelineJob].

    Attributes:
        name (str):
            Required. The name of the PipelineJob to cancel. Format:
            ``projects/{project}/locations/{location}/pipelineJobs/{pipeline_job}``
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class BatchCancelPipelineJobsRequest(proto.Message):
    r"""Request message for
    [PipelineService.BatchCancelPipelineJobs][google.cloud.aiplatform.v1beta1.PipelineService.BatchCancelPipelineJobs].

    Attributes:
        parent (str):
            Required. The name of the PipelineJobs' parent resource.
            Format: ``projects/{project}/locations/{location}``
        names (MutableSequence[str]):
            Required. The names of the PipelineJobs to cancel. A maximum
            of 32 PipelineJobs can be cancelled in a batch. Format:
            ``projects/{project}/locations/{location}/pipelineJobs/{pipelineJob}``
    """

    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )
    names: MutableSequence[str] = proto.RepeatedField(
        proto.STRING,
        number=2,
    )


class BatchCancelPipelineJobsResponse(proto.Message):
    r"""Response message for
    [PipelineService.BatchCancelPipelineJobs][google.cloud.aiplatform.v1beta1.PipelineService.BatchCancelPipelineJobs].

    Attributes:
        pipeline_jobs (MutableSequence[google.cloud.aiplatform_v1beta1.types.PipelineJob]):
            PipelineJobs cancelled.
    """

    pipeline_jobs: MutableSequence[gca_pipeline_job.PipelineJob] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=gca_pipeline_job.PipelineJob,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
