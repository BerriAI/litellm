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

import logging
import os
import tempfile
import time
from typing import Any, Dict, List, Optional
import uuid

from google.api_core import client_info
from google.api_core import exceptions
from google.api_core.gapic_v1 import client_info as v1_client_info
from google.cloud import bigquery
from google.cloud import bigquery_storage
from google.cloud.aiplatform import initializer
from google.cloud.bigquery_storage import types
import pyarrow.parquet as pq
from ray.data._internal.remote_fn import cached_remote_fn
from ray.data.block import Block
from ray.data.block import BlockAccessor
from ray.data.block import BlockMetadata
from ray.data.datasource.datasource import Datasource
from ray.data.datasource.datasource import Reader
from ray.data.datasource.datasource import ReadTask
from ray.data.datasource.datasource import WriteResult
from ray.types import ObjectRef


_BQ_GAPIC_VERSION = bigquery.__version__ + "+vertex_ray"
_BQS_GAPIC_VERSION = bigquery_storage.__version__ + "+vertex_ray"
bq_info = client_info.ClientInfo(
    gapic_version=_BQ_GAPIC_VERSION, user_agent=f"ray-on-vertex/{_BQ_GAPIC_VERSION}"
)
bqstorage_info = v1_client_info.ClientInfo(
    gapic_version=_BQS_GAPIC_VERSION, user_agent=f"ray-on-vertex/{_BQS_GAPIC_VERSION}"
)

DEFAULT_MAX_RETRY_CNT = 10
RATE_LIMIT_EXCEEDED_SLEEP_TIME = 11


class _BigQueryDatasourceReader(Reader):
    def __init__(
        self,
        project_id: Optional[str] = None,
        dataset: Optional[str] = None,
        query: Optional[str] = None,
        parallelism: Optional[int] = -1,
        **kwargs: Optional[Dict[str, Any]],
    ):
        self._project_id = project_id or initializer.global_config.project
        self._dataset = dataset
        self._query = query
        self._kwargs = kwargs

        if query is not None and dataset is not None:
            raise ValueError(
                "[Ray on Vertex AI]: Query and dataset kwargs cannot both be provided (must be mutually exclusive)."
            )

    def get_read_tasks(self, parallelism: int) -> List[ReadTask]:
        # Executed by a worker node
        def _read_single_partition(stream, kwargs) -> Block:
            client = bigquery_storage.BigQueryReadClient(client_info=bqstorage_info)
            reader = client.read_rows(stream.name)
            return reader.to_arrow()

        if self._query:
            query_client = bigquery.Client(
                project=self._project_id, client_info=bq_info
            )
            query_job = query_client.query(self._query)
            query_job.result()
            destination = str(query_job.destination)
            dataset_id = destination.split(".")[-2]
            table_id = destination.split(".")[-1]
        else:
            self._validate_dataset_table_exist(self._project_id, self._dataset)
            dataset_id = self._dataset.split(".")[0]
            table_id = self._dataset.split(".")[1]

        bqs_client = bigquery_storage.BigQueryReadClient(client_info=bqstorage_info)
        table = f"projects/{self._project_id}/datasets/{dataset_id}/tables/{table_id}"

        if parallelism == -1:
            parallelism = None
        requested_session = types.ReadSession(
            table=table,
            data_format=types.DataFormat.ARROW,
        )
        read_session = bqs_client.create_read_session(
            parent=f"projects/{self._project_id}",
            read_session=requested_session,
            max_stream_count=parallelism,
        )

        read_tasks = []
        print("[Ray on Vertex AI]: Created streams:", len(read_session.streams))
        if len(read_session.streams) < parallelism:
            print(
                "[Ray on Vertex AI]: The number of streams created by the "
                + "BigQuery Storage Read API is less than the requested "
                + "parallelism due to the size of the dataset."
            )

        for stream in read_session.streams:
            # Create a metadata block object to store schema, etc.
            metadata = BlockMetadata(
                num_rows=None,
                size_bytes=None,
                schema=None,
                input_files=None,
                exec_stats=None,
            )

            # Create a no-arg wrapper read function which returns a block
            read_single_partition = (
                lambda stream=stream, kwargs=self._kwargs: [  # noqa: F731
                    _read_single_partition(stream, kwargs)
                ]
            )

            # Create the read task and pass the wrapper and metadata in
            read_task = ReadTask(read_single_partition, metadata)
            read_tasks.append(read_task)

        return read_tasks

    def estimate_inmemory_data_size(self) -> Optional[int]:
        # TODO(b/281891467): Implement this method
        return None

    def _validate_dataset_table_exist(self, project_id: str, dataset: str) -> None:
        client = bigquery.Client(project=project_id, client_info=bq_info)
        dataset_id = dataset.split(".")[0]
        try:
            client.get_dataset(dataset_id)
        except exceptions.NotFound:
            raise ValueError(
                "[Ray on Vertex AI]: Dataset {} is not found. Please ensure that it exists.".format(
                    dataset_id
                )
            )

        try:
            client.get_table(dataset)
        except exceptions.NotFound:
            raise ValueError(
                "[Ray on Vertex AI]: Table {} is not found. Please ensure that it exists.".format(
                    dataset
                )
            )


class BigQueryDatasource(Datasource):
    def create_reader(self, **kwargs) -> Reader:
        return _BigQueryDatasourceReader(**kwargs)

    # BigQuery write for Ray 2.4.0
    def do_write(
        self,
        blocks: List[ObjectRef[Block]],
        metadata: List[BlockMetadata],
        ray_remote_args: Optional[Dict[str, Any]],
        project_id: Optional[str] = None,
        dataset: Optional[str] = None,
        max_retry_cnt: Optional[int] = DEFAULT_MAX_RETRY_CNT,
        overwrite_table: Optional[bool] = True,
    ) -> List[ObjectRef[WriteResult]]:
        def _write_single_block(
            block: Block, metadata: BlockMetadata, project_id: str, dataset: str
        ):
            print("[Ray on Vertex AI]: Starting to write", metadata.num_rows, "rows")
            block = BlockAccessor.for_block(block).to_arrow()

            client = bigquery.Client(project=project_id, client_info=bq_info)
            job_config = bigquery.LoadJobConfig(autodetect=True)
            job_config.source_format = bigquery.SourceFormat.PARQUET
            job_config.write_disposition = bigquery.WriteDisposition.WRITE_APPEND

            with tempfile.TemporaryDirectory() as temp_dir:
                fp = os.path.join(temp_dir, f"block_{uuid.uuid4()}.parquet")
                pq.write_table(block, fp, compression="SNAPPY")

                retry_cnt = 0
                while retry_cnt <= max_retry_cnt:
                    with open(fp, "rb") as source_file:
                        job = client.load_table_from_file(
                            source_file, dataset, job_config=job_config
                        )
                    try:
                        logging.info(job.result())
                        break
                    except exceptions.Forbidden as e:
                        retry_cnt += 1
                        if retry_cnt > max_retry_cnt:
                            break
                        print(
                            "[Ray on Vertex AI]: A block write encountered"
                            + f" a rate limit exceeded error {retry_cnt} time(s)."
                            + " Sleeping to try again."
                        )
                        logging.debug(e)
                        time.sleep(RATE_LIMIT_EXCEEDED_SLEEP_TIME)

                # Raise exception if retry_cnt exceeds MAX_RETRY_CNT
                if retry_cnt > max_retry_cnt:
                    print(
                        f"[Ray on Vertex AI]: Maximum ({max_retry_cnt}) retry count exceeded."
                        + " Ray will attempt to retry the block write via fault tolerance."
                        + " For more information, see https://docs.ray.io/en/latest/ray-core/fault_tolerance/tasks.html"
                    )
                    raise RuntimeError(
                        f"[Ray on Vertex AI]: Write failed due to {retry_cnt}"
                        + " repeated API rate limit exceeded responses. Consider"
                        + " specifiying the max_retry_cnt kwarg with a higher value."
                    )

            print("[Ray on Vertex AI]: Finished writing", metadata.num_rows, "rows")

        project_id = project_id or initializer.global_config.project

        if dataset is None:
            raise ValueError(
                "[Ray on Vertex AI]: Dataset is required when writing to BigQuery."
            )

        if ray_remote_args is None:
            ray_remote_args = {}

        _write_single_block = cached_remote_fn(_write_single_block).options(
            **ray_remote_args
        )
        write_tasks = []

        # Set up datasets to write
        client = bigquery.Client(project=project_id, client_info=bq_info)
        dataset_id = dataset.split(".", 1)[0]
        try:
            client.get_dataset(dataset_id)
        except exceptions.NotFound:
            client.create_dataset(f"{project_id}.{dataset_id}", timeout=30)
            print(f"[Ray on Vertex AI]: Created dataset {dataset_id}")

        # Delete table if overwrite_table is True
        if overwrite_table:
            print(
                f"[Ray on Vertex AI]: Attempting to delete table {dataset}"
                + " if it already exists since kwarg overwrite_table = True."
            )
            client.delete_table(f"{project_id}.{dataset}", not_found_ok=True)
        else:
            print(
                f"[Ray on Vertex AI]: The write will append to table {dataset}"
                + " if it already exists since kwarg overwrite_table = False."
            )

        print("[Ray on Vertex AI]: Writing", len(blocks), "blocks")
        for i in range(len(blocks)):
            write_task = _write_single_block.remote(
                blocks[i], metadata[i], project_id, dataset
            )
            write_tasks.append(write_task)
        return write_tasks
