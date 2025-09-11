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
        "SpecialistPool",
    },
)


class SpecialistPool(proto.Message):
    r"""SpecialistPool represents customers' own workforce to work on
    their data labeling jobs. It includes a group of specialist
    managers and workers. Managers are responsible for managing the
    workers in this pool as well as customers' data labeling jobs
    associated with this pool. Customers create specialist pool as
    well as start data labeling jobs on Cloud, managers and workers
    handle the jobs using CrowdCompute console.

    Attributes:
        name (str):
            Required. The resource name of the
            SpecialistPool.
        display_name (str):
            Required. The user-defined name of the
            SpecialistPool. The name can be up to 128
            characters long and can consist of any UTF-8
            characters.
            This field should be unique on project-level.
        specialist_managers_count (int):
            Output only. The number of managers in this
            SpecialistPool.
        specialist_manager_emails (MutableSequence[str]):
            The email addresses of the managers in the
            SpecialistPool.
        pending_data_labeling_jobs (MutableSequence[str]):
            Output only. The resource name of the pending
            data labeling jobs.
        specialist_worker_emails (MutableSequence[str]):
            The email addresses of workers in the
            SpecialistPool.
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    display_name: str = proto.Field(
        proto.STRING,
        number=2,
    )
    specialist_managers_count: int = proto.Field(
        proto.INT32,
        number=3,
    )
    specialist_manager_emails: MutableSequence[str] = proto.RepeatedField(
        proto.STRING,
        number=4,
    )
    pending_data_labeling_jobs: MutableSequence[str] = proto.RepeatedField(
        proto.STRING,
        number=5,
    )
    specialist_worker_emails: MutableSequence[str] = proto.RepeatedField(
        proto.STRING,
        number=7,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
