"""Ray on Vertex AI."""

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
import sys

from google.cloud.aiplatform.preview.vertex_ray.bigquery_datasource import (
    BigQueryDatasource,
)
from google.cloud.aiplatform.preview.vertex_ray.client_builder import (
    VertexRayClientBuilder as ClientBuilder,
)

from google.cloud.aiplatform.preview.vertex_ray.cluster_init import (
    create_ray_cluster,
    delete_ray_cluster,
    get_ray_cluster,
    list_ray_clusters,
    update_ray_cluster,
)

from google.cloud.aiplatform.preview.vertex_ray import data

from google.cloud.aiplatform.preview.vertex_ray.util.resources import (
    Resources,
    NodeImages,
)

from google.cloud.aiplatform.preview.vertex_ray.dashboard_sdk import (
    get_job_submission_client_cluster_info,
)

if sys.version_info[1] != 10:
    print(
        "[Ray on Vertex]: The client environment with Python version 3.10 is required."
    )

__all__ = (
    "BigQueryDatasource",
    "data",
    "ClientBuilder",
    "get_job_submission_client_cluster_info",
    "create_ray_cluster",
    "delete_ray_cluster",
    "get_ray_cluster",
    "list_ray_clusters",
    "update_ray_cluster",
    "Resources",
    "NodeImages",
)
