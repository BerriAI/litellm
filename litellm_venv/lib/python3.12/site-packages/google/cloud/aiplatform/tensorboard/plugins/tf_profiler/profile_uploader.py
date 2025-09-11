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
"""Upload profile sessions to Vertex AI Tensorboard."""
from collections import defaultdict
import datetime
import functools
import os
import re
from typing import (
    DefaultDict,
    Dict,
    Generator,
    List,
    Optional,
    Set,
    Tuple,
)

import grpc
from tensorboard.uploader import upload_tracker
from tensorboard.uploader import util
from tensorboard.uploader.proto import server_info_pb2
from tensorboard.util import tb_logging
import tensorflow as tf

from google.cloud import storage
from google.cloud.aiplatform.compat.services import tensorboard_service_client
from google.cloud.aiplatform.compat.types import tensorboard_data
from google.cloud.aiplatform.compat.types import tensorboard_service
from google.cloud.aiplatform.compat.types import tensorboard_time_series
from google.cloud.aiplatform.tensorboard import uploader_utils
from google.protobuf import timestamp_pb2 as timestamp

TensorboardServiceClient = tensorboard_service_client.TensorboardServiceClient

logger = tb_logging.get_logger()


class ProfileRequestSender(uploader_utils.RequestSender):
    """Helper class for building requests for the profiler plugin.

    While the profile plugin does create event files when a profile run is performed
    for a new training run, these event files do not contain any values
    like other events do. Instead, the plugin will create subdirectories and profiling
    files within these subdirectories.

    To verify the plugin, subdirectories need to be searched to confirm valid
    profile directories and files.

    This class is not threadsafe. Use external synchronization if
    calling its methods concurrently.
    """

    PLUGIN_NAME = "profile"
    PROFILE_PATH = "plugins/profile"

    def __init__(
        self,
        experiment_resource_name: str,
        api: TensorboardServiceClient,
        upload_limits: server_info_pb2.UploadLimits,
        blob_rpc_rate_limiter: util.RateLimiter,
        blob_storage_bucket: storage.Bucket,
        blob_storage_folder: str,
        tracker: upload_tracker.UploadTracker,
        logdir: str,
        source_bucket: Optional[storage.Bucket],
    ):
        """Constructs ProfileRequestSender for the given experiment resource.

        Args:
            experiment_resource_name (str):
                Required. Name of the experiment resource of the form:
                    projects/{project}/locations/{location}/tensorboards/{tensorboard}/experiments/{experiment}
            api (TensorboardServiceClient):
                Required. Tensorboard service stub used to interact with experiment resource.
            upload_limits (server_info_pb2.UploadLimits):
                Required. Upload limits for for api calls.
            blob_rpc_rate_limiter (util.RateLimiter):
                Required. A `RateLimiter` to use to limit write RPC frequency.
                Note this limit applies at the level of single RPCs in the Scalar and
                Tensor case, but at the level of an entire blob upload in the Blob
                case-- which may require a few preparatory RPCs and a stream of chunks.
                Note the chunk stream is internally rate-limited by backpressure from
                the server, so it is not a concern that we do not explicitly rate-limit
                within the stream here.
            blob_storage_bucket (storage.Bucket):
                Required. A `storage.Bucket` to send all blob files to.
            blob_storage_folder (str):
                Required. Name of the folder to save blob files to within the blob_storage_bucket.
            tracker (upload_tracker.UploadTracker):
                Required. Upload tracker to track information about uploads.
            logdir (str).
                Required. The log directory for the request sender to search.
            source_bucket (Optional[storage.Bucket]):
                Optional. The user's specified `storage.Bucket` to save events to. If a user is uploading from
                a local directory, this can be None.
        """
        self._experiment_resource_name = experiment_resource_name
        self._api = api
        self._logdir = logdir
        self._tag_metadata = {}
        self._tracker = tracker
        self._one_platform_resource_manager = uploader_utils.OnePlatformResourceManager(
            experiment_resource_name=experiment_resource_name, api=api
        )

        self._run_to_file_request_sender: Dict[str, _FileRequestSender] = {}
        self._run_to_profile_loaders: Dict[str, _ProfileSessionLoader] = {}

        self._file_request_sender_factory = functools.partial(
            _FileRequestSender,
            api=api,
            rpc_rate_limiter=blob_rpc_rate_limiter,
            max_blob_request_size=upload_limits.max_blob_request_size,
            max_blob_size=upload_limits.max_blob_size,
            blob_storage_bucket=blob_storage_bucket,
            source_bucket=source_bucket,
            blob_storage_folder=blob_storage_folder,
            tracker=self._tracker,
        )

    def _is_valid_event(self, run_name: str) -> bool:
        """Determines whether a valid profile session has occurred.

        Profile events are determined by whether a corresponding directory has
        been created for the profile plugin.

        Args:
            run_name (str):
                Required. String representing the run name.

        Returns:
            True if is a valid profile plugin event, False otherwise.
        """

        return tf.io.gfile.isdir(self._profile_dir(run_name))

    def _profile_dir(self, run_name: str) -> str:
        """Converts run name to full profile path.

        Args:
            run_name (str):
                Required. Name of training run.

        Returns:
            Full path for run name.
        """
        return os.path.join(self._logdir, run_name, self.PROFILE_PATH)

    def send_request(self, run_name: str):
        """Accepts run_name and sends an RPC request if an event is detected.

        Args:
            run_name (str):
                Required. Name of the training run.
        """

        if not self._is_valid_event(run_name):
            logger.warning("No such profile run for %s", run_name)
            return

        # Create a profiler loader if one is not created.
        # This will store any new runs that occur within the training.
        if run_name not in self._run_to_profile_loaders:
            self._run_to_profile_loaders[run_name] = _ProfileSessionLoader(
                self._profile_dir(run_name)
            )

        tb_run = self._one_platform_resource_manager.get_run_resource_name(run_name)

        if run_name not in self._run_to_file_request_sender:
            self._run_to_file_request_sender[
                run_name
            ] = self._file_request_sender_factory(tb_run)

        # Loop through any of the profiling sessions within this training run.
        # A training run can have multiple profile sessions.
        for prof_session, files in self._run_to_profile_loaders[
            run_name
        ].prof_sessions_to_files():
            event_time = datetime.datetime.strptime(prof_session, "%Y_%m_%d_%H_%M_%S")
            event_timestamp = timestamp.Timestamp().FromDatetime(event_time)

            # Implicit flush to any files after they are uploaded.
            self._run_to_file_request_sender[run_name].add_files(
                files=files,
                tag=prof_session,
                plugin=self.PLUGIN_NAME,
                event_timestamp=event_timestamp,
            )


class _ProfileSessionLoader(object):
    """Loader for a profile session within a training run.

    The term 'session' refers to an instance of a profile, where
    one may have multiple profile sessions under a training run.
    """

    # A regular expression for the naming of a profiling path.
    PROF_PATH_REGEX = r".*\/plugins\/profile\/[0-9]{4}_[0-9]{2}_[0-9]{2}_[0-9]{2}_[0-9]{2}_[0-9]{2}\/?$"

    def __init__(
        self,
        path: str,
    ):
        """Create a loader for profiling sessions with a training run.

        Args:
            path (str):
                Required. Path to the training run, which contains one or more profiling
                sessions. Path should end with '/profile/plugin'.
        """
        self._path = path
        self._prof_session_to_files: DefaultDict[str, Set[str]] = defaultdict(set)

    def _path_filter(self, path: str) -> bool:
        """Determine which paths we should upload.

        Paths written by profiler should be of form:
        /some/path/to/dir/plugins/profile/%Y_%m_%d_%H_%M_%S

        Args:
            path (str):
                Required. String representing a full directory path.

        Returns:
            True if valid path and path matches the filter, False otherwise.
        """
        return tf.io.gfile.isdir(path) and re.match(self.PROF_PATH_REGEX, path)

    def _path_to_files(self, prof_session: str, path: str) -> List[str]:
        """Generates files that have not yet been tracked.

        Files are generated by the profiler and are added to an internal
        dictionary. For files that have not yet been uploaded, we return these
        files.

        Args:
            prof_session (str):
                Required. The profiling session name.
            path (str):
                Required. Directory of the profiling session.

        Returns:
            files (List[str]):
                Files that have not been tracked yet.
        """

        files = []
        for prof_file in tf.io.gfile.listdir(path):
            full_file_path = os.path.join(path, prof_file)
            if full_file_path not in self._prof_session_to_files[prof_session]:
                files.append(full_file_path)

        self._prof_session_to_files[prof_session].update(files)
        return files

    def prof_sessions_to_files(self) -> Generator[Tuple[str, List[str]], None, None]:
        """Map files to a profile session.

        Yields:
            A tuple containing the profiling session name and a list of files
                that have not yet been tracked.
        """

        prof_sessions = tf.io.gfile.listdir(self._path)

        for prof_session in prof_sessions:
            # Remove trailing slashes in path names
            prof_session = (
                prof_session if not prof_session.endswith("/") else prof_session[:-1]
            )

            full_path = os.path.join(self._path, prof_session)
            if not self._path_filter(full_path):
                continue

            files = self._path_to_files(prof_session, full_path)

            if files:
                yield (prof_session, files)


class _FileRequestSender(object):
    """Uploader for file based items.

    This sender is closely related to the `_BlobRequestSender`, however it expects
    file paths instead of blob files, so that data is not directly read in and instead
    files are moved between buckets. Additionally, this sender does not take event files
    as the other request sender objects do. The sender takes files from either local storage
    or a gcs bucket and uploads to the tensorboard bucket.

    This class is not threadsafe. Use external synchronization if calling its
    methods concurrently.
    """

    def __init__(
        self,
        run_resource_id: str,
        api: TensorboardServiceClient,
        rpc_rate_limiter: util.RateLimiter,
        max_blob_request_size: int,
        max_blob_size: int,
        blob_storage_bucket: storage.Bucket,
        blob_storage_folder: str,
        tracker: upload_tracker.UploadTracker,
        source_bucket: Optional[storage.Bucket] = None,
    ):
        """Creates a _FileRequestSender object.

        Args:
            run_resource_id (str):
                Required. Name of the run resource of the form:
                    projects/{project}/locations/{location}/tensorboards/{tensorboard}/experiments/{experiment}/runs/{run}
            api (TensorboardServiceClient):
                Required. TensorboardServiceStub for calling various tensorboard services.
            rpc_rate_limiter (util.RateLimiter):
                Required. A `RateLimiter` to use to limit write RPC frequency.
                Note this limit applies at the level of single RPCs in the Scalar and
                Tensor case, but at the level of an entire blob upload in the Blob
                case-- which may require a few preparatory RPCs and a stream of chunks.
                Note the chunk stream is internally rate-limited by backpressure from
                the server, so it is not a concern that we do not explicitly rate-limit
                within the stream here.
            max_blob_request_size (int):
                Required. Maximum request size to send.
            max_blob_size (int):
                Required. Maximum size in bytes of the blobs to send.
            blob_storage_bucket (storage.Bucket):
                Required. Bucket to send event files to.
            blob_storage_folder (str):
                Required. The folder to save blob files to.
            tracker (upload_tracker.UploadTracker):
                Required. Track any uploads to backend.
            source_bucket (storage.Bucket):
                Optional. The source bucket to upload from. If not set, use local filesystem instead.
        """
        self._run_resource_id = run_resource_id
        self._api = api
        self._rpc_rate_limiter = rpc_rate_limiter
        self._max_blob_request_size = max_blob_request_size
        self._max_blob_size = max_blob_size
        self._tracker = tracker
        self._time_series_resource_manager = uploader_utils.TimeSeriesResourceManager(
            run_resource_id, api
        )

        self._bucket = blob_storage_bucket
        self._folder = blob_storage_folder
        self._source_bucket = source_bucket

        self._new_request()

    def _new_request(self):
        """Declares the previous event complete."""
        self._files = []
        self._tag = None
        self._plugin = None
        self._event_timestamp = None

    def add_files(
        self,
        files: List[str],
        tag: str,
        plugin: str,
        event_timestamp: timestamp.Timestamp,
    ):
        """Attempts to add the given file to the current request.

        If a file does not exist, the file is ignored and the rest of the
        files are checked to ensure the remaining files exist. After checking
        the files, an rpc is immediately sent.

        Files are flushed immediately, opposed to some of the other request senders.

        Args:
            files (List[str]):
                Required. The paths of the files to upload.
            tag (str):
                Required. A unique identifier for the blob sequence.
            plugin (str):
                Required. Name of the plugin making the request.
            event_timestamp (timestamp.Timestamp):
                Required. The time the event is created.
        """

        for prof_file in files:
            if not tf.io.gfile.exists(prof_file):
                logger.warning(
                    "The file provided does not exist. "
                    "Will not be uploading file %s.",
                    prof_file,
                )
            else:
                self._files.append(prof_file)

        self._tag = tag
        self._plugin = plugin
        self._event_timestamp = event_timestamp
        self.flush()
        self._new_request()

    def flush(self):
        """Sends the current file fully, and clears it to make way for the next."""
        if not self._files:
            return

        time_series_proto = self._time_series_resource_manager.get_or_create(
            self._tag,
            lambda: tensorboard_time_series.TensorboardTimeSeries(
                display_name=self._tag,
                value_type=tensorboard_time_series.TensorboardTimeSeries.ValueType.BLOB_SEQUENCE,
                plugin_name=self._plugin,
            ),
        )
        m = re.match(
            ".*/tensorboards/(.*)/experiments/(.*)/runs/(.*)/timeSeries/(.*)",
            time_series_proto.name,
        )
        blob_path_prefix = "tensorboard-{}/{}/{}/{}".format(m[1], m[2], m[3], m[4])
        blob_path_prefix = (
            "{}/{}".format(self._folder, blob_path_prefix)
            if self._folder
            else blob_path_prefix
        )
        sent_blob_ids = []

        for prof_file in self._files:
            self._rpc_rate_limiter.tick()
            file_size = tf.io.gfile.stat(prof_file).length
            with self._tracker.blob_tracker(file_size) as blob_tracker:
                if not self._file_too_large(prof_file):
                    blob_id = self._upload(prof_file, blob_path_prefix)
                    sent_blob_ids.append(str(blob_id))
                    blob_tracker.mark_uploaded(blob_id is not None)

        data_point = tensorboard_data.TimeSeriesDataPoint(
            blobs=tensorboard_data.TensorboardBlobSequence(
                values=[
                    tensorboard_data.TensorboardBlob(id=blob_id)
                    for blob_id in sent_blob_ids
                ]
            ),
            wall_time=self._event_timestamp,
        )

        time_series_data_proto = tensorboard_data.TimeSeriesData(
            tensorboard_time_series_id=time_series_proto.name.split("/")[-1],
            value_type=tensorboard_time_series.TensorboardTimeSeries.ValueType.BLOB_SEQUENCE,
            values=[data_point],
        )
        request = tensorboard_service.WriteTensorboardRunDataRequest(
            time_series_data=[time_series_data_proto]
        )

        _prune_empty_time_series_from_blob(request)
        if not request.time_series_data:
            return

        with uploader_utils.request_logger(request):
            try:
                self._api.write_tensorboard_run_data(
                    tensorboard_run=self._run_resource_id,
                    time_series_data=request.time_series_data,
                )
            except grpc.RpcError as e:
                logger.error("Upload call failed with error %s", e)

    def _file_too_large(self, filename: str) -> bool:
        """Determines if a file is too large to upload.

        Args:
            filename (str):
                Required. The filename to check.

        Returns:
            True if too large, False otherwise.
        """

        file_size = tf.io.gfile.stat(filename).length
        if file_size > self._max_blob_size:
            logger.warning(
                "Blob too large; skipping.  Size %d exceeds limit of %d bytes.",
                file_size,
                self._max_blob_size,
            )
            return True
        return False

    def _upload(self, filename: str, blob_path_prefix: Optional[str] = None) -> str:
        """Copies files between either a local directory or a bucket and the tenant bucket.

        Args:
            filename (str):
                Required. The full path of the file to upload.
            blob_path_prefix (str):
                Optional. Path prefix for the location to store the file.

        Returns:
            blob_id (str):
                The base path of the file.
        """
        blob_id = os.path.basename(filename)
        blob_path = (
            "{}/{}".format(blob_path_prefix, blob_id) if blob_path_prefix else blob_id
        )

        # Source bucket indicates files are storage on cloud storage
        if self._source_bucket:
            self._copy_between_buckets(filename, blob_path)
        else:
            self._upload_from_local(filename, blob_path)

        return blob_id

    def _copy_between_buckets(self, filename: str, blob_path: str):
        """Move files between the user's bucket and the tenant bucket.

        Args:
            filename (str):
                Required. Full path of the file to upload.
            blob_path (str):
                Required. A bucket path to upload the file to.

        """
        blob_name = _get_blob_from_file(filename)

        source_blob = self._source_bucket.blob(blob_name)

        self._source_bucket.copy_blob(
            source_blob,
            self._bucket,
            blob_path,
        )

    def _upload_from_local(self, filename: str, blob_path: str):
        """Uploads a local file to the tenant bucket.

        Args:
            filename (str):
                Required. Full path of the file to upload.
            blob_path (str):
                Required. A bucket path to upload the file to.a
        """
        blob = self._bucket.blob(blob_path)
        blob.upload_from_filename(filename)


def _get_blob_from_file(fp: str) -> Optional[str]:
    """Gets blob name from a storage bucket.

    Args:
        fp (str):
            Required. A file path.

    Returns:
        blob_name (str):
            Optional. Base blob file name if it exists, else None
    """
    m = re.match(r"gs:\/\/.*?\/(.*)", fp)
    if not m:
        logger.warning("Could not get the blob name from file %s", fp)
        return None
    return m[1]


def _prune_empty_time_series_from_blob(
    request: tensorboard_service.WriteTensorboardRunDataRequest,
):
    """Removes empty time_series from request if there are no blob files.'

    Args:
        request (tensorboard_service.WriteTensorboardRunDataRequest):
            Required. A write request for blob files.
    """
    for time_series_idx, time_series_data in reversed(
        list(enumerate(request.time_series_data))
    ):
        if not any(x.blobs for x in time_series_data.values):
            del request.time_series_data[time_series_idx]
