# -*- coding: utf-8 -*-

# Copyright 2022 Google LLC
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

import re

from google.cloud.aiplatform.compat.types import (
    pipeline_state as gca_pipeline_state,
)

_PIPELINE_COMPLETE_STATES = set(
    [
        gca_pipeline_state.PipelineState.PIPELINE_STATE_SUCCEEDED,
        gca_pipeline_state.PipelineState.PIPELINE_STATE_FAILED,
        gca_pipeline_state.PipelineState.PIPELINE_STATE_CANCELLED,
        gca_pipeline_state.PipelineState.PIPELINE_STATE_PAUSED,
    ]
)

_PIPELINE_ERROR_STATES = set([gca_pipeline_state.PipelineState.PIPELINE_STATE_FAILED])

# Pattern for valid names used as a Vertex resource name.
_VALID_NAME_PATTERN = re.compile("^[a-z][-a-z0-9]{0,127}$", re.IGNORECASE)

# Pattern for an Artifact Registry URL.
_VALID_AR_URL = re.compile(r"^https:\/\/([\w-]+)-kfp\.pkg\.dev\/.*", re.IGNORECASE)

# Pattern for any JSON or YAML file over HTTPS.
_VALID_HTTPS_URL = re.compile(r"^https:\/\/([\.\/\w-]+)\/.*(json|yaml|yml)$")

# Fields to include in returned PipelineJob when enable_simple_view=True in PipelineJob.list()
_READ_MASK_FIELDS = [
    "name",
    "state",
    "display_name",
    "pipeline_spec.pipeline_info",
    "create_time",
    "start_time",
    "end_time",
    "update_time",
    "labels",
    "template_uri",
    "template_metadata.version",
    "job_detail.pipeline_run_context",
    "job_detail.pipeline_context",
]
