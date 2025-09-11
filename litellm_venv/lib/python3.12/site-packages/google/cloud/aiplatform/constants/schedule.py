# -*- coding: utf-8 -*-

# Copyright 2023 Google LLC
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

from google.cloud.aiplatform.compat.types import (
    schedule as gca_schedule,
)
from google.cloud.aiplatform.constants import pipeline as pipeline_constants

_SCHEDULE_COMPLETE_STATES = set(
    [
        gca_schedule.Schedule.State.PAUSED,
        gca_schedule.Schedule.State.COMPLETED,
    ]
)

_SCHEDULE_ERROR_STATES = set(
    [
        gca_schedule.Schedule.State.STATE_UNSPECIFIED,
    ]
)

# Pattern for valid names used as a Vertex resource name.
_VALID_NAME_PATTERN = pipeline_constants._VALID_NAME_PATTERN

# Pattern for an Artifact Registry URL.
_VALID_AR_URL = pipeline_constants._VALID_AR_URL

# Pattern for any JSON or YAML file over HTTPS.
_VALID_HTTPS_URL = pipeline_constants._VALID_HTTPS_URL
