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

"""Helpers for creating a profile request sender for tf profiler plugin."""

import os
import re
from typing import Tuple

from tensorboard.uploader import upload_tracker
from tensorboard.uploader import util
from tensorboard.uploader.proto import server_info_pb2
from tensorboard.util import tb_logging

from google.api_core import exceptions
from google.cloud import aiplatform
from google.cloud import storage
from google.cloud.aiplatform.utils import TensorboardClientWithOverride
from google.cloud.aiplatform.tensorboard import uploader_utils
from google.cloud.aiplatform.compat.types import tensorboard_experiment
from google.cloud.aiplatform.tensorboard.plugins.tf_profiler import profile_uploader
from google.cloud.aiplatform import training_utils

logger = tb_logging.get_logger()


def _get_api_client() -> TensorboardClientWithOverride:
    """Creates an Tensorboard API client."""
    m = re.match(
        "projects/.*/locations/(.*)/tensorboards/.*",
        training_utils.environment_variables.tensorboard_resource_name,
    )
    region = m[1]

    api_client = aiplatform.initializer.global_config.create_client(
        client_class=TensorboardClientWithOverride,
        location_override=region,
        api_base_path_override=training_utils.environment_variables.tensorboard_api_uri,
    )

    return api_client


def _get_project_id() -> str:
    """Gets the project id from the tensorboard resource name.

    Returns:
        Project ID for current project.

    Raises:
        ValueError: Cannot parse the tensorboard resource name.
    """
    m = re.match(
        "projects/(.*)/locations/.*/tensorboards/.*",
        training_utils.environment_variables.tensorboard_resource_name,
    )
    if not m:
        raise ValueError(
            "Incorrect format for tensorboard resource name: %s",
            training_utils.environment_variables.tensorboard_resource_name,
        )
    return m[1]


def _make_upload_limits() -> server_info_pb2.UploadLimits:
    """Creates the upload limits for tensorboard.

    Returns:
        An UploadLimits object.
    """
    upload_limits = server_info_pb2.UploadLimits()
    upload_limits.min_blob_request_interval = 10
    upload_limits.max_blob_request_size = 4 * (2**20) - 256 * (2**10)
    upload_limits.max_blob_size = 10 * (2**30)  # 10GiB

    return upload_limits


def _get_blob_items(
    api_client: TensorboardClientWithOverride,
) -> Tuple[storage.bucket.Bucket, str]:
    """Gets the blob storage items for the tensorboard resource.

    Args:
        api_client ():
            Required. Client go get information about the tensorboard instance.

    Returns:
        A tuple of storage buckets and the blob storage folder name.
    """
    project_id = _get_project_id()
    tensorboard = api_client.get_tensorboard(
        name=training_utils.environment_variables.tensorboard_resource_name
    )

    path_prefix = tensorboard.blob_storage_path_prefix + "/"
    first_slash_index = path_prefix.find("/")
    bucket_name = path_prefix[:first_slash_index]
    blob_storage_bucket = storage.Client(project=project_id).bucket(bucket_name)
    blob_storage_folder = path_prefix[first_slash_index + 1 :]

    return blob_storage_bucket, blob_storage_folder


def _get_or_create_experiment(
    api: TensorboardClientWithOverride, experiment_name: str
) -> str:
    """Creates a tensorboard experiment.

    Args:
        api (TensorboardClientWithOverride):
            Required. An api for interfacing with tensorboard resources.
        experiment_name (str):
            Required. The name of the experiment to get or create.

    Returns:
        The name of the experiment.
    """
    tb_experiment = tensorboard_experiment.TensorboardExperiment()

    try:
        experiment = api.create_tensorboard_experiment(
            parent=training_utils.environment_variables.tensorboard_resource_name,
            tensorboard_experiment=tb_experiment,
            tensorboard_experiment_id=experiment_name,
        )
    except exceptions.AlreadyExists:
        logger.info("Creating experiment failed. Retrieving experiment.")
        experiment_name = os.path.join(
            training_utils.environment_variables.tensorboard_resource_name,
            "experiments",
            experiment_name,
        )
        experiment = api.get_tensorboard_experiment(name=experiment_name)

    return experiment.name


def create_profile_request_sender() -> profile_uploader.ProfileRequestSender:
    """Creates the `ProfileRequestSender` for the profile plugin.

    A profile request sender is created for the plugin so that after profiling runs
    have finished, data can be uploaded to the tensorboard backend.

    Returns:
        A ProfileRequestSender object.
    """
    api_client = _get_api_client()

    experiment_name = _get_or_create_experiment(
        api_client, training_utils.environment_variables.cloud_ml_job_id
    )

    upload_limits = _make_upload_limits()

    blob_rpc_rate_limiter = util.RateLimiter(
        upload_limits.min_blob_request_interval / 100
    )

    blob_storage_bucket, blob_storage_folder = _get_blob_items(
        api_client,
    )

    source_bucket = uploader_utils.get_source_bucket(
        training_utils.environment_variables.tensorboard_log_dir
    )

    profile_request_sender = profile_uploader.ProfileRequestSender(
        experiment_name,
        api_client,
        upload_limits=upload_limits,
        blob_rpc_rate_limiter=blob_rpc_rate_limiter,
        blob_storage_bucket=blob_storage_bucket,
        blob_storage_folder=blob_storage_folder,
        source_bucket=source_bucket,
        tracker=upload_tracker.UploadTracker(verbosity=1),
        logdir=training_utils.environment_variables.tensorboard_log_dir,
    )

    return profile_request_sender
