# -*- coding: utf-8 -*-

# Copyright 2023 Google LLC
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
"""Launches Tensorboard Uploader for SDK."""

import threading
from typing import FrozenSet, Optional

from google.cloud.aiplatform import base
from google.cloud.aiplatform import initializer
from google.cloud.aiplatform.metadata.metadata import _experiment_tracker
from google.cloud.aiplatform.compat.services import (
    tensorboard_service_client,
)
from google.cloud.aiplatform.utils import TensorboardClientWithOverride

_LOGGER = base.Logger(__name__)
TensorboardServiceClient = tensorboard_service_client.TensorboardServiceClient


class _TensorBoardTracker:
    "Tracks TensorBoard uploader and uploader thread"

    def __init__(self):
        self._tensorboard_uploader = None

    def upload_tb_log(
        self,
        tensorboard_experiment_name: str,
        logdir: str,
        tensorboard_id: Optional[str] = None,
        project: Optional[str] = None,
        location: Optional[str] = None,
        experiment_display_name: Optional[str] = None,
        run_name_prefix: Optional[str] = None,
        description: Optional[str] = None,
        verbosity: Optional[int] = 1,
        allowed_plugins: Optional[FrozenSet[str]] = None,
    ):
        """upload only the existing data in the logdir and then return immediately

        ```py
        Sample usage:
        aiplatform.init(location='us-central1', project='my-project')
        aiplatform.upload_tb_log(tensorboard_id='123',tensorboard_experiment_name='my-experiment',logdir='my-logdir')
        ```

        Args:
          tensorboard_experiment_name (str): Required. Name of this tensorboard
            experiment. Unique to the given
            projects/{project}/locations/{location}/tensorboards/{tensorboard_id}
          logdir (str): Required. The location of the TensorBoard logs that resides either in the local file system or Cloud Storage
          tensorboard_id (str): Optional. TensorBoard ID. If not set, tensorboard_id in aiplatform.init will be used.
          project (str): Optional. Project the TensorBoard is in. If not set, project set in aiplatform.init will be used.
          location (str): Optional. Location the TensorBoard is in. If not set, location set in aiplatform.init will be used.
          experiment_display_name (str): Optional. The display name of the
            experiment.
          run_name_prefix (str): Optional. If present, all runs created by this
            invocation will have their name prefixed by this value.
          description (str): Optional. String description to assign to the
            experiment.
          verbosity (str): Optional. Level of verbosity, an integer. Supported
            value: 0 - No upload statistics is printed. 1 - Print upload statistics
              while uploading data (default).
          allowed_plugins (FrozenSet[str]): Optional. List of additional allowed plugin names.
        """
        self._create_uploader(
            tensorboard_id=tensorboard_id,
            tensorboard_experiment_name=tensorboard_experiment_name,
            logdir=logdir,
            project=project,
            location=location,
            one_shot=True,
            experiment_display_name=experiment_display_name,
            run_name_prefix=run_name_prefix,
            description=description,
            verbosity=verbosity,
            allowed_plugins=allowed_plugins,
        ).start_uploading()
        _LOGGER.info("One time TensorBoard log upload completed.")

    def start_upload_tb_log(
        self,
        tensorboard_experiment_name: str,
        logdir: str,
        tensorboard_id: Optional[str] = None,
        project: Optional[str] = None,
        location: Optional[str] = None,
        experiment_display_name: Optional[str] = None,
        run_name_prefix: Optional[str] = None,
        description: Optional[str] = None,
        allowed_plugins: Optional[FrozenSet[str]] = None,
    ):
        """Continues to listen for new data in the logdir and uploads when it appears.

        Note that after calling `start_upload_tb_log()` your thread will kept alive even if
        an exception is thrown. To ensure the thread gets shut down, put any code after
        `start_upload_tb_log()` and before `end_upload_tb_log()` in a `try` statement, and call
        `end_upload_tb_log()` in `finally`.

        ```py
        Sample usage:
        aiplatform.init(location='us-central1', project='my-project')
        aiplatform.start_upload_tb_log(tensorboard_id='123',tensorboard_experiment_name='my-experiment',logdir='my-logdir')

        try:
          # your code here
        finally:
          aiplatform.end_upload_tb_log()
        ```

        Args:
          tensorboard_experiment_name (str): Required. Name of this tensorboard
            experiment. Unique to the given
            projects/{project}/locations/{location}/tensorboards/{tensorboard_id}.
          logdir (str): Required. path of the log directory to upload
          tensorboard_id (str): Optional. TensorBoard ID. If not set, tensorboard_id in aiplatform.init will be used.
          project (str): Optional. Project the TensorBoard is in. If not set, project set in aiplatform.init will be used.
          location (str): Optional. Location the TensorBoard is in. If not set, location set in aiplatform.init will be used.
          experiment_display_name (str): Optional. The display name of the
            experiment.
          run_name_prefix (str): Optional. If present, all runs created by this
            invocation will have their name prefixed by this value.
          description (str): Optional. String description to assign to the
            experiment.
          allowed_plugins (FrozenSet[str]): Optional. List of additional allowed plugin names.
        """
        if self._tensorboard_uploader:
            _LOGGER.info(
                "Continuous upload running for {}. To start a new"
                " uploading process, please call end_upload_tb_log() to end the current one".format(
                    self._tensorboard_uploader.get_experiment_resource_name()
                )
            )
            return
        self._tensorboard_uploader = self._create_uploader(
            tensorboard_id=tensorboard_id,
            tensorboard_experiment_name=tensorboard_experiment_name,
            logdir=logdir,
            project=project,
            location=location,
            one_shot=False,
            experiment_display_name=experiment_display_name,
            run_name_prefix=run_name_prefix,
            description=description,
            verbosity=0,
            allowed_plugins=allowed_plugins,
        )
        threading.Thread(target=self._tensorboard_uploader.start_uploading).start()

    def end_upload_tb_log(self):
        """Ends the current TensorBoard uploader

        ```py
        aiplatform.start_upload_tb_log(...)
        ...
        aiplatform.end_upload_tb_log()
        ```
        """
        if not self._tensorboard_uploader:
            _LOGGER.info(
                "No uploader is running. To start a new uploader, call"
                " start_upload_tb_log()."
            )
            return
        self._tensorboard_uploader._end_uploading()
        self._tensorboard_uploader = None

    def _create_uploader(
        self,
        tensorboard_id: str,
        tensorboard_experiment_name: str,
        logdir: str,
        one_shot: bool,
        project: Optional[str] = None,
        location: Optional[str] = None,
        experiment_display_name: Optional[str] = None,
        run_name_prefix: Optional[str] = None,
        description: Optional[str] = None,
        verbosity: Optional[int] = 1,
        allowed_plugins: Optional[FrozenSet[str]] = None,
    ) -> "TensorBoardUploader":  # noqa: F821
        """Create a TensorBoardUploader and a TensorBoard Experiment

        Args:
          tensorboard_id (str): Required. TensorBoard ID.
          tensorboard_experiment_name (str): Required. Name of this tensorboard experiment. Unique to the given projects/{project}/locations/{location}/tensorboards/{tensorboard_id}
          logdir (str): Required. path of the log directory to upload
          one_shot (bool): Required. True: upload only the existing data in the logdir and then return immediately. False: continue to listen for new data in the logdir and upload them when it appears.
          project (str): Optional. Project the TensorBoard is in. If not set, project set in aiplatform.init will be used.
          location (str): Optional. Location the TensorBoard is in. If not set, location set in aiplatform.init will be used.
          experiment_display_name (str): Optional. The display name of the experiment.
          run_name_prefix (str): Optional. If present, all runs created by this invocation will have their name prefixed by this value.
          description (str): Optional. String description to assign to the experiment.
          verbosity (int)): Optional. Level of verbosity. Supported value: 0 - No upload statistics is printed. 1 - Print upload statistics while uploading data (default).
          allowed_plugins (FrozenSet[str]): Optional. List of additional allowed plugin names.

        Returns:
            An instance of TensorBoardUploader.

        Raise:
            ImportError if TensorBoard package is not installed.
            ValueError if TensorBoard ID is not specified in argument nor set with an Experiment
        """
        try:
            from google.cloud.aiplatform.tensorboard.uploader import (
                TensorBoardUploader,
            )
            from google.cloud.aiplatform.tensorboard import uploader_utils
            from google.cloud.aiplatform.tensorboard import (
                uploader_constants,
            )
        except ImportError:
            raise ImportError(
                "TensorBoard is not installed. Please install TensorBoard using pip install google-cloud-aiplatform[tensorboard] to use it in the Vertex SDK."
            )
        project = project or initializer.global_config.project
        location = location or initializer.global_config.location
        if tensorboard_id:
            tensorboard_resource_name = TensorboardServiceClient.tensorboard_path(
                project, location, tensorboard_id
            )
        else:
            if _experiment_tracker._get_global_tensorboard():
                tensorboard_resource_name = (
                    _experiment_tracker._get_global_tensorboard().resource_name
                )
            elif _experiment_tracker._experiment:
                if _experiment_tracker._experiment._lookup_backing_tensorboard():
                    tensorboard_resource_name = (
                        _experiment_tracker._experiment._lookup_backing_tensorboard().resource_name
                    )
                else:
                    raise ValueError(
                        f"No TensorBoard associated with experiment {initializer.global_config.experiment_name}. Please provide tensorboard_id in the argument."
                    )
            else:
                raise ValueError(
                    "No TensorBoard found. Please provide tensorboard_id in the argument."
                )

        api_client = initializer.global_config.create_client(
            client_class=TensorboardClientWithOverride,
            location_override=location,
        )
        (
            blob_storage_bucket,
            blob_storage_folder,
        ) = uploader_utils.get_blob_storage_bucket_and_folder(
            api_client, tensorboard_resource_name, project
        )

        plugins = uploader_constants.ALLOWED_PLUGINS
        if allowed_plugins:
            plugins += [
                plugin
                for plugin in allowed_plugins
                if plugin not in uploader_constants.ALLOWED_PLUGINS
            ]

        tensorboard_uploader = TensorBoardUploader(
            experiment_name=tensorboard_experiment_name,
            tensorboard_resource_name=tensorboard_resource_name,
            experiment_display_name=experiment_display_name,
            blob_storage_bucket=blob_storage_bucket,
            blob_storage_folder=blob_storage_folder,
            allowed_plugins=plugins,
            writer_client=api_client,
            logdir=logdir,
            one_shot=one_shot,
            run_name_prefix=run_name_prefix,
            description=description,
            verbosity=verbosity,
        )
        tensorboard_uploader.create_experiment()
        print(
            "View your Tensorboard at https://{}.{}/experiment/{}".format(
                location,
                "tensorboard.googleusercontent.com",
                tensorboard_uploader.get_experiment_resource_name().replace("/", "+"),
            )
        )
        return tensorboard_uploader


_tensorboard_tracker = _TensorBoardTracker()
