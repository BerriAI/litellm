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


from typing import Optional

from google.auth import credentials as auth_credentials
from google.cloud import resourcemanager

from google.cloud.aiplatform import initializer


def get_project_id(
    project_number: str,
    credentials: Optional[auth_credentials.Credentials] = None,
) -> str:
    """Gets project ID given the project number

    Args:
        project_number (str):
            Required. The automatically generated unique identifier for your GCP project.
        credentials: The custom credentials to use when making API calls.
            Optional. If not provided, default credentials will be used.

    Returns:
        str - The unique string used to differentiate your GCP project from all others in Google Cloud.

    """

    credentials = credentials or initializer.global_config.credentials

    projects_client = resourcemanager.ProjectsClient(credentials=credentials)

    project = projects_client.get_project(name=f"projects/{project_number}")

    return project.project_id


def get_project_number(
    project_id: str,
    credentials: Optional[auth_credentials.Credentials] = None,
) -> str:
    """Gets project ID given the project number

    Args:
        project_id (str):
            Required. Google Cloud project unique ID.
        credentials: The custom credentials to use when making API calls.
            Optional. If not provided, default credentials will be used.

    Returns:
        str - The automatically generated unique numerical identifier for your GCP project.

    """

    credentials = credentials or initializer.global_config.credentials

    projects_client = resourcemanager.ProjectsClient(credentials=credentials)

    project = projects_client.get_project(name=f"projects/{project_id}")
    project_number = project.name.split("/", 1)[1]

    return project_number
