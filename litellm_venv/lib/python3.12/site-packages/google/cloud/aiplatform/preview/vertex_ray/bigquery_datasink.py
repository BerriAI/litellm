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


import logging
import os
import tempfile
import time
import uuid
from typing import Any, Iterable, Optional

import pyarrow.parquet as pq

from google.api_core import client_info
from google.api_core import exceptions
from google.cloud import bigquery
from google.cloud.aiplatform import initializer

import ray
from ray.data._internal.execution.interfaces import TaskContext
from ray.data._internal.remote_fn import cached_remote_fn
from ray.data.block import Block, BlockAccessor

try:
    from ray.data.datasource.datasink import Datasink
except ImportError:
    # If datasink cannot be imported, Ray 2.9.3 is not installed
    Datasink = None

DEFAULT_MAX_RETRY_CNT = 10
RATE_LIMIT_EXCEEDED_SLEEP_TIME = 11

_BQ_GAPIC_VERSION = bigquery.__version__ + "+vertex_ray"
bq_info = client_info.ClientInfo(
    gapic_version=_BQ_GAPIC_VERSION, user_agent=f"ray-on-vertex/{_BQ_GAPIC_VERSION}"
)

if Datasink is None:
    _BigQueryDatasink = None
else:
    # BigQuery write for Ray 2.9.3
    class _BigQueryDatasink(Datasink):
        def __init__(
            self,
            dataset: str,
            project_id: str = None,
            max_retry_cnt: int = DEFAULT_MAX_RETRY_CNT,
            overwrite_table: Optional[bool] = True,
        ) -> None:
            self.dataset = dataset
            self.project_id = project_id or initializer.global_config.project
            self.max_retry_cnt = max_retry_cnt
            self.overwrite_table = overwrite_table

        def on_write_start(self) -> None:
            # Set up datasets to write
            client = bigquery.Client(project=self.project_id, client_info=bq_info)
            dataset_id = self.dataset.split(".", 1)[0]
            try:
                client.get_dataset(dataset_id)
            except exceptions.NotFound:
                client.create_dataset(f"{self.project_id}.{dataset_id}", timeout=30)
                print("[Ray on Vertex AI]: Created dataset " + dataset_id)

            # Delete table if overwrite_table is True
            if self.overwrite_table:
                print(
                    f"[Ray on Vertex AI]: Attempting to delete table {self.dataset}"
                    + " if it already exists since kwarg overwrite_table = True."
                )
                client.delete_table(
                    f"{self.project_id}.{self.dataset}", not_found_ok=True
                )
            else:
                print(
                    "[Ray on Vertex AI]: The write will append to table "
                    + f"{self.dataset} if it already exists "
                    + "since kwarg overwrite_table = False."
                )

        def write(
            self,
            blocks: Iterable[Block],
            ctx: TaskContext,
        ) -> Any:
            def _write_single_block(
                block: Block, project_id: str, dataset: str
            ) -> None:
                block = BlockAccessor.for_block(block).to_arrow()

                client = bigquery.Client(project=project_id, client_info=bq_info)
                job_config = bigquery.LoadJobConfig(autodetect=True)
                job_config.source_format = bigquery.SourceFormat.PARQUET
                job_config.write_disposition = bigquery.WriteDisposition.WRITE_APPEND

                with tempfile.TemporaryDirectory() as temp_dir:
                    fp = os.path.join(temp_dir, f"block_{uuid.uuid4()}.parquet")
                    pq.write_table(block, fp, compression="SNAPPY")

                    retry_cnt = 0
                    while retry_cnt <= self.max_retry_cnt:
                        with open(fp, "rb") as source_file:
                            job = client.load_table_from_file(
                                source_file, dataset, job_config=job_config
                            )
                        try:
                            logging.info(job.result())
                            break
                        except exceptions.Forbidden as e:
                            retry_cnt += 1
                            if retry_cnt > self.max_retry_cnt:
                                break
                            print(
                                "[Ray on Vertex AI]: A block write encountered"
                                + f" a rate limit exceeded error {retry_cnt} time(s)."
                                + " Sleeping to try again."
                            )
                            logging.debug(e)
                            time.sleep(RATE_LIMIT_EXCEEDED_SLEEP_TIME)

                    # Raise exception if retry_cnt exceeds max_retry_cnt
                    if retry_cnt > self.max_retry_cnt:
                        print(
                            f"[Ray on Vertex AI]: Maximum ({self.max_retry_cnt}) retry count exceeded."
                            + " Ray will attempt to retry the block write via fault tolerance."
                            + " For more information, see https://docs.ray.io/en/latest/ray-core/fault_tolerance/tasks.html"
                        )
                        raise RuntimeError(
                            f"[Ray on Vertex AI]: Write failed due to {retry_cnt}"
                            + " repeated API rate limit exceeded responses. Consider"
                            + " specifiying the max_retry_cnt kwarg with a higher value."
                        )

            _write_single_block = cached_remote_fn(_write_single_block)

            # Launch a remote task for each block within this write task
            ray.get(
                [
                    _write_single_block.remote(block, self.project_id, self.dataset)
                    for block in blocks
                ]
            )

            return "ok"
