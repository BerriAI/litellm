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

from google.cloud.aiplatform_v1.types import tuning_job as gca_tuning_job


__protobuf__ = proto.module(
    package="google.cloud.aiplatform.v1",
    manifest={
        "CreateTuningJobRequest",
        "GetTuningJobRequest",
        "ListTuningJobsRequest",
        "ListTuningJobsResponse",
        "CancelTuningJobRequest",
    },
)


class CreateTuningJobRequest(proto.Message):
    r"""Request message for
    [GenAiTuningService.CreateTuningJob][google.cloud.aiplatform.v1.GenAiTuningService.CreateTuningJob].

    Attributes:
        parent (str):
            Required. The resource name of the Location to create the
            TuningJob in. Format:
            ``projects/{project}/locations/{location}``
        tuning_job (google.cloud.aiplatform_v1.types.TuningJob):
            Required. The TuningJob to create.
    """

    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )
    tuning_job: gca_tuning_job.TuningJob = proto.Field(
        proto.MESSAGE,
        number=2,
        message=gca_tuning_job.TuningJob,
    )


class GetTuningJobRequest(proto.Message):
    r"""Request message for
    [GenAiTuningService.GetTuningJob][google.cloud.aiplatform.v1.GenAiTuningService.GetTuningJob].

    Attributes:
        name (str):
            Required. The name of the TuningJob resource. Format:
            ``projects/{project}/locations/{location}/tuningJobs/{tuning_job}``
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class ListTuningJobsRequest(proto.Message):
    r"""Request message for
    [GenAiTuningService.ListTuningJobs][google.cloud.aiplatform.v1.GenAiTuningService.ListTuningJobs].

    Attributes:
        parent (str):
            Required. The resource name of the Location to list the
            TuningJobs from. Format:
            ``projects/{project}/locations/{location}``
        filter (str):
            Optional. The standard list filter.
        page_size (int):
            Optional. The standard list page size.
        page_token (str):
            Optional. The standard list page token. Typically obtained
            via [ListTuningJob.next_page_token][] of the previous
            GenAiTuningService.ListTuningJob][] call.
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


class ListTuningJobsResponse(proto.Message):
    r"""Response message for
    [GenAiTuningService.ListTuningJobs][google.cloud.aiplatform.v1.GenAiTuningService.ListTuningJobs]

    Attributes:
        tuning_jobs (MutableSequence[google.cloud.aiplatform_v1.types.TuningJob]):
            List of TuningJobs in the requested page.
        next_page_token (str):
            A token to retrieve the next page of results. Pass to
            [ListTuningJobsRequest.page_token][google.cloud.aiplatform.v1.ListTuningJobsRequest.page_token]
            to obtain that page.
    """

    @property
    def raw_page(self):
        return self

    tuning_jobs: MutableSequence[gca_tuning_job.TuningJob] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=gca_tuning_job.TuningJob,
    )
    next_page_token: str = proto.Field(
        proto.STRING,
        number=2,
    )


class CancelTuningJobRequest(proto.Message):
    r"""Request message for
    [GenAiTuningService.CancelTuningJob][google.cloud.aiplatform.v1.GenAiTuningService.CancelTuningJob].

    Attributes:
        name (str):
            Required. The name of the TuningJob to cancel. Format:
            ``projects/{project}/locations/{location}/tuningJobs/{tuning_job}``
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
