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
        "JobState",
    },
)


class JobState(proto.Enum):
    r"""Describes the state of a job.

    Values:
        JOB_STATE_UNSPECIFIED (0):
            The job state is unspecified.
        JOB_STATE_QUEUED (1):
            The job has been just created or resumed and
            processing has not yet begun.
        JOB_STATE_PENDING (2):
            The service is preparing to run the job.
        JOB_STATE_RUNNING (3):
            The job is in progress.
        JOB_STATE_SUCCEEDED (4):
            The job completed successfully.
        JOB_STATE_FAILED (5):
            The job failed.
        JOB_STATE_CANCELLING (6):
            The job is being cancelled. From this state the job may only
            go to either ``JOB_STATE_SUCCEEDED``, ``JOB_STATE_FAILED``
            or ``JOB_STATE_CANCELLED``.
        JOB_STATE_CANCELLED (7):
            The job has been cancelled.
        JOB_STATE_PAUSED (8):
            The job has been stopped, and can be resumed.
        JOB_STATE_EXPIRED (9):
            The job has expired.
        JOB_STATE_UPDATING (10):
            The job is being updated. Only jobs in the ``RUNNING`` state
            can be updated. After updating, the job goes back to the
            ``RUNNING`` state.
        JOB_STATE_PARTIALLY_SUCCEEDED (11):
            The job is partially succeeded, some results
            may be missing due to errors.
    """
    JOB_STATE_UNSPECIFIED = 0
    JOB_STATE_QUEUED = 1
    JOB_STATE_PENDING = 2
    JOB_STATE_RUNNING = 3
    JOB_STATE_SUCCEEDED = 4
    JOB_STATE_FAILED = 5
    JOB_STATE_CANCELLING = 6
    JOB_STATE_CANCELLED = 7
    JOB_STATE_PAUSED = 8
    JOB_STATE_EXPIRED = 9
    JOB_STATE_UPDATING = 10
    JOB_STATE_PARTIALLY_SUCCEEDED = 11


__all__ = tuple(sorted(__protobuf__.manifest))
