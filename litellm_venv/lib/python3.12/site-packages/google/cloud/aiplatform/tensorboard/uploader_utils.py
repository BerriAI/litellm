# -*- coding: utf-8 -*-

# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""Shared utils for tensorboard log uploader."""
import abc
import contextlib
import json
import logging
import re
import time
from typing import Callable, Dict, Generator, List, Optional, Tuple
import uuid

from absl import app
from google.api_core import exceptions
from google.cloud import storage
from google.cloud.aiplatform.compat.services import (
    tensorboard_service_client,
)
from google.cloud.aiplatform.compat.types import tensorboard_run
from google.cloud.aiplatform.compat.types import tensorboard_service
from google.cloud.aiplatform.compat.types import tensorboard_time_series
import grpc

from tensorboard.util import tb_logging

TensorboardServiceClient = tensorboard_service_client.TensorboardServiceClient

logger = tb_logging.get_logger()
logger.setLevel(logging.WARNING)


class ExistingResourceNotFoundError(RuntimeError):
    """Resource could not be created or retrieved."""


class RequestSender(object):
    """A base class for additional request sender objects.

    Currently just used for typing.
    """

    @abc.abstractmethod
    def send_requests(run_name: str):
        """Sends any request for the run."""
        pass


class OnePlatformResourceManager(object):
    """Helper class managing One Platform resources."""

    CREATE_RUN_BATCH_SIZE = 1000
    CREATE_TIME_SERIES_BATCH_SIZE = 1000

    def __init__(self, experiment_resource_name: str, api: TensorboardServiceClient):
        """Constructor for OnePlatformResourceManager.

        Args:
            experiment_resource_name (str):
                Required. The resource id for the run with the following format
                projects/{project}/locations/{location}/tensorboards/{tensorboard}/experiments/{experiment}
            api (TensorboardServiceClient):
                Required. TensorboardServiceStub for calling various tensorboard services.
        """
        self._experiment_resource_name = experiment_resource_name
        self._api = api
        self._run_name_to_run_resource_name: Dict[str, str] = {}
        self._run_tag_name_to_time_series_name: Dict[(str, str), str] = {}

    def batch_create_runs(
        self, run_names: List[str]
    ) -> List[tensorboard_run.TensorboardRun]:
        """Batch creates TensorboardRuns.

        Args:
            run_names: a list of run_names for creating the TensorboardRuns.
        Returns:
            the created TensorboardRuns
        """
        batch_size = OnePlatformResourceManager.CREATE_RUN_BATCH_SIZE
        created_runs = []
        for i in range(0, len(run_names), batch_size):
            one_batch_run_names = run_names[i : i + batch_size]
            tb_run_requests = [
                tensorboard_service.CreateTensorboardRunRequest(
                    parent=self._experiment_resource_name,
                    tensorboard_run=tensorboard_run.TensorboardRun(
                        display_name=run_name
                    ),
                    tensorboard_run_id=str(uuid.uuid4()),
                )
                for run_name in one_batch_run_names
            ]

            tb_runs = self._api.batch_create_tensorboard_runs(
                parent=self._experiment_resource_name,
                requests=tb_run_requests,
            ).tensorboard_runs

            self._run_name_to_run_resource_name.update(
                {run.display_name: run.name for run in tb_runs}
            )

            created_runs.extend(tb_runs)

        return created_runs

    def batch_create_time_series(
        self,
        run_tag_name_to_time_series: Dict[
            Tuple[str, str], tensorboard_time_series.TensorboardTimeSeries
        ],
    ) -> List[tensorboard_time_series.TensorboardTimeSeries]:
        """Batch creates TensorboardTimeSeries.

        Args:
            run_tag_name_to_time_series: a dictionary of
            (run_name, tag_name) to TensorboardTimeSeries proto, containing
            the TensorboardTimeSeries to create.
        Returns:
            the created TensorboardTimeSeries
        """
        batch_size = OnePlatformResourceManager.CREATE_TIME_SERIES_BATCH_SIZE
        run_tag_name_to_time_series_entries = list(run_tag_name_to_time_series.items())
        run_resource_name_to_run_name = {
            v: k for k, v in self._run_name_to_run_resource_name.items()
        }
        created_time_series = []
        for i in range(0, len(run_tag_name_to_time_series_entries), batch_size):
            requests = [
                tensorboard_service.CreateTensorboardTimeSeriesRequest(
                    parent=self._run_name_to_run_resource_name[run_name],
                    tensorboard_time_series=time_series,
                )
                for (
                    (run_name, tag_name),
                    time_series,
                ) in run_tag_name_to_time_series_entries[i : i + batch_size]
            ]

            time_series = self._api.batch_create_tensorboard_time_series(
                parent=self._experiment_resource_name,
                requests=requests,
            ).tensorboard_time_series

            self._run_tag_name_to_time_series_name.update(
                {
                    (
                        run_resource_name_to_run_name[
                            ts.name[: ts.name.index("/timeSeries")]
                        ],
                        ts.display_name,
                    ): ts.name
                    for ts in time_series
                }
            )

            created_time_series.extend(time_series)

        return created_time_series

    def get_run_resource_name(self, run_name: str) -> str:
        """
        Get the resource name of the run if it exists, otherwise creates the run
        on One Platform before returning its resource name.

        Args:
            run_name (str):
                Required. The name of the run.

        Returns:
            run_resource (str):
                Resource name of the run.
        """
        if run_name not in self._run_name_to_run_resource_name:
            tb_run = self._create_or_get_run_resource(run_name)
            self._run_name_to_run_resource_name[run_name] = tb_run.name
        return self._run_name_to_run_resource_name[run_name]

    def _create_or_get_run_resource(
        self, run_name: str
    ) -> tensorboard_run.TensorboardRun:
        """Creates a new run resource in current tensorboard experiment resource.

        Args:
            run_name (str):
                Required. The display name of this run.

        Returns:
            tb_run (tensorboard_run.TensorboardRun):
                The TensorboardRun given the run_name.

        Raises:
            ExistingResourceNotFoundError:
                Run name could not be found in resource list.
            exceptions.InvalidArgument:
                run_name argument is invalid.
        """
        tb_run = tensorboard_run.TensorboardRun()
        tb_run.display_name = run_name
        try:
            tb_run = self._api.create_tensorboard_run(
                parent=self._experiment_resource_name,
                tensorboard_run=tb_run,
                tensorboard_run_id=str(uuid.uuid4()),
            )
        except exceptions.InvalidArgument as e:
            # If the run name already exists then retrieve it
            if "already exist" in e.message:
                runs_pages = self._api.list_tensorboard_runs(
                    parent=self._experiment_resource_name
                )
                for tb_run in runs_pages:
                    if tb_run.display_name == run_name:
                        break

                if tb_run.display_name != run_name:
                    raise ExistingResourceNotFoundError(
                        "Run with name %s already exists but is not resource list."
                        % run_name
                    )
            else:
                raise
        return tb_run

    def get_time_series_resource_name(
        self,
        run_name: str,
        tag_name: str,
        time_series_resource_creator: Callable[
            [], tensorboard_time_series.TensorboardTimeSeries
        ],
    ) -> str:
        """
        Get the resource name of the time series corresponding to the tag, if it
        exists, otherwise creates the time series on One Platform before
        returning its resource name.

        Args:
            run_name (str):
                Required. The name of the run.
            tag_name (str):
                Required. The name of the tag.
            time_series_resource_creator (tensorboard_time_series.TensorboardTimeSeries):
                Required. A constructor used for creating the time series on One Platform.

        Returns:
            time_series_name (str):
                Resource name of the time series
        """
        if (run_name, tag_name) not in self._run_tag_name_to_time_series_name:
            time_series = self._create_or_get_time_series(
                self.get_run_resource_name(run_name),
                tag_name,
                time_series_resource_creator,
            )
            self._run_tag_name_to_time_series_name[
                (run_name, tag_name)
            ] = time_series.name
        return self._run_tag_name_to_time_series_name[(run_name, tag_name)]

    def _create_or_get_time_series(
        self,
        run_resource_name: str,
        tag_name: str,
        time_series_resource_creator: Callable[
            [], tensorboard_time_series.TensorboardTimeSeries
        ],
    ) -> tensorboard_time_series.TensorboardTimeSeries:
        """
        Get a time series resource with given tag_name, and create a new one on
        OnePlatform if not present.

        Args:
            tag_name (str):
                Required. The tag name of the time series in the Tensorboard log dir.
            time_series_resource_creator (Callable[[], tensorboard_time_series.TensorboardTimeSeries):
                Required. A callable that produces a TimeSeries for creation.

        Returns:
            time_series (tensorboard_time_series.TensorboardTimeSeries):
                A created or existing tensorboard_time_series.TensorboardTimeSeries.

        Raises:
            exceptions.InvalidArgument:
                Invalid run_resource_name, tag_name, or time_series_resource_creator.
            ExistingResourceNotFoundError:
                Could not find the resource given the tag name.
            ValueError:
                More than one time series with the resource name was found.
        """
        time_series = time_series_resource_creator()
        time_series.display_name = tag_name
        try:
            time_series = self._api.create_tensorboard_time_series(
                parent=run_resource_name, tensorboard_time_series=time_series
            )
        except exceptions.InvalidArgument as e:
            # If the time series display name already exists then retrieve it
            if "already exist" in e.message:
                list_of_time_series = self._api.list_tensorboard_time_series(
                    request=tensorboard_service.ListTensorboardTimeSeriesRequest(
                        parent=run_resource_name,
                        filter="display_name = {}".format(json.dumps(str(tag_name))),
                    )
                )
                num = 0
                time_series = None

                for ts in list_of_time_series:
                    num += 1
                    if num > 1:
                        break
                    time_series = ts

                if not time_series:
                    raise ExistingResourceNotFoundError(
                        "Could not find time series resource with display name: {}".format(
                            tag_name
                        )
                    )

                if num != 1:
                    raise ValueError(
                        "More than one time series resource found with display_name: {}".format(
                            tag_name
                        )
                    )
            else:
                raise
        return time_series


class TimeSeriesResourceManager(object):
    """Helper class managing Time Series resources."""

    def __init__(self, run_resource_id: str, api: TensorboardServiceClient):
        """Constructor for TimeSeriesResourceManager.

        Args:
            run_resource_id (str):
                Required. The resource id for the run with the following format.
                    projects/{project}/locations/{location}/tensorboards/{tensorboard}/experiments/{experiment}/runs/{run}
            api (TensorboardServiceClient):
                Required. A TensorboardServiceStub.
        """
        self._run_resource_id = run_resource_id
        self._api = api
        self._tag_to_time_series_proto: Dict[
            str, tensorboard_time_series.TensorboardTimeSeries
        ] = {}

    def get_or_create(
        self,
        tag_name: str,
        time_series_resource_creator: Callable[
            [], tensorboard_time_series.TensorboardTimeSeries
        ],
    ) -> tensorboard_time_series.TensorboardTimeSeries:
        """
        Get a time series resource with given tag_name, and create a new one on
        OnePlatform if not present.

        Args:
            tag_name (str):
                Required. The tag name of the time series in the Tensorboard log dir.
            time_series_resource_creator (Callable[[], tensorboard_time_series.TensorboardTimeSeries]):
                Required. A callable that produces a TimeSeries for creation.

        Returns:
            time_series (tensorboard_time_series.TensorboardTimeSeries):
                A new or existing tensorboard_time_series.TensorboardTimeSeries.

        Raises:
            exceptions.InvalidArgument:
                The tag_name or time_series_resource_creator is an invalid argument
                to create_tensorboard_time_series api call.
            ExistingResourceNotFoundError:
                Could not find the resource given the tag name.
            ValueError:
                More than one time series with the resource name was found.
        """
        if tag_name in self._tag_to_time_series_proto:
            return self._tag_to_time_series_proto[tag_name]

        time_series = time_series_resource_creator()
        time_series.display_name = tag_name
        try:
            time_series = self._api.create_tensorboard_time_series(
                parent=self._run_resource_id, tensorboard_time_series=time_series
            )
        except exceptions.InvalidArgument as e:
            # If the time series display name already exists then retrieve it
            if "already exist" in e.message:
                list_of_time_series = self._api.list_tensorboard_time_series(
                    request=tensorboard_service.ListTensorboardTimeSeriesRequest(
                        parent=self._run_resource_id,
                        filter="display_name = {}".format(json.dumps(str(tag_name))),
                    )
                )
                num = 0
                time_series = None

                for ts in list_of_time_series:
                    num += 1
                    if num > 1:
                        break
                    time_series = ts

                if not time_series:
                    raise ExistingResourceNotFoundError(
                        "Could not find time series resource with display name: {}".format(
                            tag_name
                        )
                    )

                if num != 1:
                    raise ValueError(
                        "More than one time series resource found with display_name: {}".format(
                            tag_name
                        )
                    )
            else:
                raise

        self._tag_to_time_series_proto[tag_name] = time_series
        return time_series


def get_source_bucket(logdir: str) -> Optional[storage.Bucket]:
    """Returns a storage bucket object given a log directory.

    Args:
        logdir (str):
            Required. Path of the log directory.

    Returns:
        bucket (Optional[storage.Bucket]):
            A bucket if the path is a gs bucket, None otherwise.
    """
    m = re.match(r"gs:\/\/(.*?)(?=\/|$)", logdir)
    if not m:
        return None
    bucket = storage.Client().bucket(m[1])
    return bucket


@contextlib.contextmanager
def request_logger(
    request: tensorboard_service.WriteTensorboardRunDataRequest,
) -> Generator[None, None, None]:
    """Context manager to log request size and duration.

    Args:
        request (tensorboard_service.WriteTensorboardRunDataRequest):
            Required. A request object that provides the size of the request.

    Yields:
        An empty response when the request logger has started.
    """
    upload_start_time = time.time()
    request_bytes = request._pb.ByteSize()  # pylint: disable=protected-access
    logger.info("Trying request of %d bytes", request_bytes)
    yield
    upload_duration_secs = time.time() - upload_start_time
    logger.info(
        "Upload of (%d bytes) took %.3f seconds",
        request_bytes,
        upload_duration_secs,
    )


def get_blob_storage_bucket_and_folder(
    api_client: TensorboardServiceClient,
    tensorboard_resource_name: str,
    project_id: str,
) -> Optional[Tuple[storage.Bucket, str]]:
    """Get the blob storage bucket and blob storage folder for a given TensorBoard.

    Args:
        api_client (TensorboardServiceClient): Required. TensorBoard service client
        tensorboard_resource_name (str): Required. Full TensorBoard name.
        project_id (str): Required. Project ID.

    Returns:
        Blob storage bucket and blob storage folder.

    Raise:
        NotFound exception if the TensorBoard does not exist.
    """
    try:
        tensorboard = api_client.get_tensorboard(name=tensorboard_resource_name)
    except grpc.RpcError as rpc_error:
        if rpc_error.code() == grpc.StatusCode.NOT_FOUND:
            raise app.UsageError(
                "Tensorboard resource %s not found" % tensorboard_resource_name,
                exitcode=0,
            ) from rpc_error
        raise

    if tensorboard.blob_storage_path_prefix:
        path_prefix = tensorboard.blob_storage_path_prefix + "/"
        first_slash_index = path_prefix.find("/")
        bucket_name = path_prefix[:first_slash_index]
        blob_storage_bucket = storage.Client(project=project_id).bucket(bucket_name)
        blob_storage_folder = path_prefix[first_slash_index + 1 :]
        return blob_storage_bucket, blob_storage_folder

    raise app.UsageError(
        "Tensorboard resource {} is obsolete. Please create a new one.".format(
            tensorboard_resource_name
        ),
        exitcode=0,
    )
