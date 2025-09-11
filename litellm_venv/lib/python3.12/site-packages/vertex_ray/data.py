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

import ray.data
from ray.data.dataset import Dataset
from typing import Any, Dict, Optional

from google.cloud.aiplatform.preview.vertex_ray.bigquery_datasource import (
    BigQueryDatasource,
)

try:
    from google.cloud.aiplatform.preview.vertex_ray.bigquery_datasink import (
        _BigQueryDatasink,
    )
except ImportError:
    _BigQueryDatasink = None


def read_bigquery(
    project_id: Optional[str] = None,
    dataset: Optional[str] = None,
    query: Optional[str] = None,
    *,
    parallelism: int = -1,
) -> Dataset:
    # The read is identical in Ray 2.4 and 2.9
    return ray.data.read_datasource(
        BigQueryDatasource(),
        project_id=project_id,
        dataset=dataset,
        query=query,
        parallelism=parallelism,
    )


def write_bigquery(
    ds: Dataset,
    project_id: Optional[str] = None,
    dataset: Optional[str] = None,
    max_retry_cnt: int = 10,
    ray_remote_args: Dict[str, Any] = None,
) -> Any:
    if ray.__version__ == "2.4.0":
        return ds.write_datasource(
            BigQueryDatasource(),
            project_id=project_id,
            dataset=dataset,
            max_retry_cnt=max_retry_cnt,
        )
    elif ray.__version__ == "2.9.3":
        if ray_remote_args is None:
            ray_remote_args = {}

        # Each write task will launch individual remote tasks to write each block
        # To avoid duplicate block writes, the write task should not be retried
        if ray_remote_args.get("max_retries", 0) != 0:
            print(
                "[Ray on Vertex AI]: The max_retries of a BigQuery Write "
                "Task should be set to 0 to avoid duplicate writes."
            )
        else:
            ray_remote_args["max_retries"] = 0

        datasink = _BigQueryDatasink(
            project_id=project_id, dataset=dataset, max_retry_cnt=max_retry_cnt
        )
        return ds.write_datasink(datasink, ray_remote_args=ray_remote_args)
    else:
        raise ImportError(
            f"[Ray on Vertex AI]: Unsupported version {ray.__version__}."
            + "Only 2.4.0 and 2.9.3 are supported."
        )
