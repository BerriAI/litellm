# -*- coding: utf-8 -*-

# Copyright 2021 Google LLC
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

from google.cloud.aiplatform import jobs
from google.cloud.aiplatform import tensorboard


def custom_job_console_uri(custom_job_resource_name: str) -> str:
    """Helper method to create console uri from custom job resource name."""
    fields = jobs.CustomJob._parse_resource_name(custom_job_resource_name)
    return f"https://console.cloud.google.com/ai/platform/locations/{fields['location']}/training/{fields['custom_job']}?project={fields['project']}"


def custom_job_tensorboard_console_uri(
    tensorboard_resource_name: str, custom_job_resource_name: str
) -> str:
    """Helper method to create console uri to tensorboard from custom job resource."""
    # projects+40556267596+locations+us-central1+tensorboards+740208820004847616+experiments+2214368039829241856
    fields = tensorboard.Tensorboard._parse_resource_name(tensorboard_resource_name)
    experiment_resource_name = f"{tensorboard_resource_name}/experiments/{custom_job_resource_name.split('/')[-1]}"
    uri_experiment_resource_name = experiment_resource_name.replace("/", "+")
    return f"https://{fields['location']}.tensorboard.googleusercontent.com/experiment/{uri_experiment_resource_name}"
