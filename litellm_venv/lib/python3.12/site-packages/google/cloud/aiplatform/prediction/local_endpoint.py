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
from pathlib import Path
import requests
import time
from typing import Any, Dict, List, Optional, Sequence

from google.auth.exceptions import GoogleAuthError

from google.cloud.aiplatform import initializer
from google.cloud.aiplatform.constants import prediction
from google.cloud.aiplatform.docker_utils import run
from google.cloud.aiplatform.docker_utils.errors import DockerError
from google.cloud.aiplatform.utils import prediction_utils

_logger = logging.getLogger(__name__)

_DEFAULT_CONTAINER_READY_TIMEOUT = 300
_DEFAULT_CONTAINER_READY_CHECK_INTERVAL = 1

_GCLOUD_PROJECT_ENV = "GOOGLE_CLOUD_PROJECT"


class LocalEndpoint:
    """Class that represents a local endpoint."""

    def __init__(
        self,
        serving_container_image_uri: str,
        artifact_uri: Optional[str] = None,
        serving_container_predict_route: Optional[str] = None,
        serving_container_health_route: Optional[str] = None,
        serving_container_command: Optional[Sequence[str]] = None,
        serving_container_args: Optional[Sequence[str]] = None,
        serving_container_environment_variables: Optional[Dict[str, str]] = None,
        serving_container_ports: Optional[Sequence[int]] = None,
        credential_path: Optional[str] = None,
        host_port: Optional[str] = None,
        gpu_count: Optional[int] = None,
        gpu_device_ids: Optional[List[str]] = None,
        gpu_capabilities: Optional[List[List[str]]] = None,
        container_ready_timeout: Optional[int] = None,
        container_ready_check_interval: Optional[int] = None,
    ):
        """Creates a local endpoint instance.

        Args:
            serving_container_image_uri (str):
                Required. The URI of the Model serving container.
            artifact_uri (str):
                Optional. The path to the directory containing the Model artifact and any of its
                supporting files. The path is either a GCS uri or the path to a local directory.
                If this parameter is set to a GCS uri:
                (1) ``credential_path`` must be specified for local prediction.
                (2) The GCS uri will be passed directly to ``Predictor.load``.
                If this parameter is a local directory:
                (1) The directory will be mounted to a default temporary model path.
                (2) The mounted path will be passed to ``Predictor.load``.
            serving_container_predict_route (str):
                Optional. An HTTP path to send prediction requests to the container, and
                which must be supported by it. If not specified a default HTTP path will
                be used by Vertex AI.
            serving_container_health_route (str):
                Optional. An HTTP path to send health check requests to the container, and which
                must be supported by it. If not specified a standard HTTP path will be
                used by Vertex AI.
            serving_container_command (Sequence[str]):
                Optional. The command with which the container is run. Not executed within a
                shell. The Docker image's ENTRYPOINT is used if this is not provided.
                Variable references $(VAR_NAME) are expanded using the container's
                environment. If a variable cannot be resolved, the reference in the
                input string will be unchanged. The $(VAR_NAME) syntax can be escaped
                with a double $$, ie: $$(VAR_NAME). Escaped references will never be
                expanded, regardless of whether the variable exists or not.
            serving_container_args: (Sequence[str]):
                Optional. The arguments to the command. The Docker image's CMD is used if this is
                not provided. Variable references $(VAR_NAME) are expanded using the
                container's environment. If a variable cannot be resolved, the reference
                in the input string will be unchanged. The $(VAR_NAME) syntax can be
                escaped with a double $$, ie: $$(VAR_NAME). Escaped references will
                never be expanded, regardless of whether the variable exists or not.
            serving_container_environment_variables (Dict[str, str]):
                Optional. The environment variables that are to be present in the container.
                Should be a dictionary where keys are environment variable names
                and values are environment variable values for those names.
            serving_container_ports (Sequence[int]):
                Optional. Declaration of ports that are exposed by the container. This field is
                primarily informational, it gives Vertex AI information about the
                network connections the container uses. Listing or not a port here has
                no impact on whether the port is actually exposed, any port listening on
                the default "0.0.0.0" address inside a container will be accessible from
                the network.
            credential_path (str):
                Optional. The path to the credential key that will be mounted to the container.
                If it's unset, the environment variable, ``GOOGLE_APPLICATION_CREDENTIALS``, will
                be used if set.
            host_port (str):
                Optional. The port on the host that the port, ``AIP_HTTP_PORT``, inside the container
                will be exposed as. If it's unset, a random host port will be assigned.
            gpu_count (int):
                Optional. Number of devices to request. Set to -1 to request all available devices.
                To use GPU, set either ``gpu_count`` or ``gpu_device_ids``.
                The default value is -1 if ``gpu_capabilities`` is set but both ``gpu_count`` and
                ``gpu_device_ids`` are not set.
            gpu_device_ids (List[str]):
                Optional. This parameter corresponds to ``NVIDIA_VISIBLE_DEVICES`` in the NVIDIA
                Runtime.
                To use GPU, set either ``gpu_count`` or ``gpu_device_ids``.
            gpu_capabilities (List[List[str]]):
                Optional. This parameter corresponds to ``NVIDIA_DRIVER_CAPABILITIES`` in the NVIDIA
                Runtime. The outer list acts like an OR, and each sub-list acts like an AND. The
                driver will try to satisfy one of the sub-lists.
                Available capabilities for the NVIDIA driver can be found in
                https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/user-guide.html#driver-capabilities.
                The default value is ``[["utility", "compute"]]`` if ``gpu_count`` or ``gpu_device_ids`` is
                set.
            container_ready_timeout (int):
                Optional. The timeout in second used for starting the container or succeeding the
                first health check.
            container_ready_check_interval (int):
                Optional. The time interval in second to check if the container is ready or the
                first health check succeeds.

        Raises:
            ValueError: If both ``gpu_count`` and ``gpu_device_ids`` are set.
        """
        self.container = None
        self.container_is_running = False
        self.log_start_index = 0
        self.serving_container_image_uri = serving_container_image_uri
        self.artifact_uri = artifact_uri
        self.serving_container_predict_route = (
            serving_container_predict_route or prediction.DEFAULT_LOCAL_PREDICT_ROUTE
        )
        self.serving_container_health_route = (
            serving_container_health_route or prediction.DEFAULT_LOCAL_HEALTH_ROUTE
        )
        self.serving_container_command = serving_container_command
        self.serving_container_args = serving_container_args
        self.serving_container_environment_variables = (
            serving_container_environment_variables
        )
        self.serving_container_ports = serving_container_ports
        self.container_port = prediction_utils.get_prediction_aip_http_port(
            serving_container_ports
        )

        self.credential_path = credential_path
        self.host_port = host_port
        # assigned_host_port will be updated according to the running container
        # if host_port is None.
        self.assigned_host_port = host_port

        self.gpu_count = gpu_count
        self.gpu_device_ids = gpu_device_ids
        self.gpu_capabilities = gpu_capabilities

        if self.gpu_count and self.gpu_device_ids:
            raise ValueError(
                "At most one gpu_count or gpu_device_ids can be set but both are set."
            )
        if (self.gpu_count or self.gpu_device_ids) and self.gpu_capabilities is None:
            self.gpu_capabilities = prediction.DEFAULT_LOCAL_RUN_GPU_CAPABILITIES
        if self.gpu_capabilities and not self.gpu_count and not self.gpu_device_ids:
            self.gpu_count = prediction.DEFAULT_LOCAL_RUN_GPU_COUNT

        self.container_ready_timeout = (
            container_ready_timeout or _DEFAULT_CONTAINER_READY_TIMEOUT
        )
        self.container_ready_check_interval = (
            container_ready_check_interval or _DEFAULT_CONTAINER_READY_CHECK_INTERVAL
        )

    def __enter__(self):
        """Enters the runtime context related to this object."""
        try:
            self.serve()
        except Exception as exception:
            _logger.error(f"Exception during entering a context: {exception}.")
            raise
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """Exits the runtime context related to this object.

        Args:
            exc_type:
                Optional. Class of the exception.
            exc_value:
                Optional. Type of the exception.
            exc_traceback:
                Optional. Traceback that has the information of the exception.
        """
        self.stop()

    def __del__(self):
        """Stops the container when the instance is about to be destroyed."""
        self.stop()

    def serve(self):
        """Starts running the container and serves the traffic locally.

        An environment variable, ``GOOGLE_CLOUD_PROJECT``, will be set to the project in the global config.
        This is required if the credentials file does not have project specified and used to
        recognize the project by the Cloud Storage client.

        Raises:
            DockerError: If the container is not ready or health checks do not succeed after the
                timeout.
        """
        if self.container and self.container_is_running:
            _logger.warning(
                "The local endpoint has started serving traffic. "
                "No need to call `serve()` again."
            )
            return

        try:
            try:
                project_id = initializer.global_config.project
                _logger.info(
                    f"Got the project id from the global config: {project_id}."
                )
            except (GoogleAuthError, ValueError):
                project_id = None

            envs = (
                dict(self.serving_container_environment_variables)
                if self.serving_container_environment_variables is not None
                else {}
            )
            if project_id is not None:
                envs[_GCLOUD_PROJECT_ENV] = project_id

            self.container = run.run_prediction_container(
                self.serving_container_image_uri,
                artifact_uri=self.artifact_uri,
                serving_container_predict_route=self.serving_container_predict_route,
                serving_container_health_route=self.serving_container_health_route,
                serving_container_command=self.serving_container_command,
                serving_container_args=self.serving_container_args,
                serving_container_environment_variables=envs,
                serving_container_ports=self.serving_container_ports,
                credential_path=self.credential_path,
                host_port=self.host_port,
                gpu_count=self.gpu_count,
                gpu_device_ids=self.gpu_device_ids,
                gpu_capabilities=self.gpu_capabilities,
            )

            # Retrieves the assigned host port.
            self._wait_until_container_runs()
            if self.host_port is None:
                self.container.reload()
                self.assigned_host_port = self.container.ports[
                    f"{self.container_port}/tcp"
                ][0]["HostPort"]
            self.container_is_running = True
            # Waits until the model server starts.
            self._wait_until_health_check_succeeds()
        except Exception as exception:
            _logger.error(f"Exception during starting serving: {exception}.")
            self._stop_container_if_exists()
            self.container_is_running = False
            raise

    def stop(self) -> None:
        """Explicitly stops the container."""
        self._stop_container_if_exists()
        self.container_is_running = False

    def _wait_until_container_runs(self) -> None:
        """Waits until the container is in running status or timeout.

        Raises:
            DockerError: If timeout.
        """
        elapsed_time = 0
        while (
            self.get_container_status() != run.CONTAINER_RUNNING_STATUS
            and elapsed_time < self.container_ready_timeout
        ):
            time.sleep(self.container_ready_check_interval)
            elapsed_time += self.container_ready_check_interval

        if elapsed_time >= self.container_ready_timeout:
            raise DockerError("The container never starts running.", "", 1)

    def _wait_until_health_check_succeeds(self):
        """Waits until a health check succeeds or timeout.

        Raises:
            DockerError: If container exits or timeout.
        """
        elapsed_time = 0
        try:
            response = self.run_health_check(verbose=False)
        except requests.exceptions.RequestException:
            response = None

        while elapsed_time < self.container_ready_timeout and (
            response is None or response.status_code != 200
        ):
            time.sleep(self.container_ready_check_interval)
            elapsed_time += self.container_ready_check_interval
            try:
                response = self.run_health_check(verbose=False)
            except requests.exceptions.RequestException:
                response = None

            if self.get_container_status() != run.CONTAINER_RUNNING_STATUS:
                self.print_container_logs(
                    show_all=True,
                    message="Container already exited, all container logs:",
                )
                raise DockerError(
                    "Container exited before the first health check succeeded.", "", 1
                )

        if elapsed_time >= self.container_ready_timeout:
            self.print_container_logs(
                show_all=True,
                message="Health check never succeeds, all container logs:",
            )
            raise DockerError("The health check never succeeded.", "", 1)

    def _stop_container_if_exists(self):
        """Stops the container if the container exists."""
        if self.container is not None:
            self.container.stop()

    def predict(
        self,
        request: Optional[Any] = None,
        request_file: Optional[str] = None,
        headers: Optional[Dict] = None,
        verbose: bool = True,
    ) -> requests.models.Response:
        """Executes a prediction.

        Args:
            request (Any):
                Optional. The request sent to the container.
            request_file (str):
                Optional. The path to a request file sent to the container.
            headers (Dict):
                Optional. The headers in the prediction request.
            verbose (bool):
                Required. Whether or not print logs if any.

        Returns:
            The prediction response.

        Raises:
            RuntimeError: If the local endpoint has been stopped.
            ValueError: If both ``request`` and ``request_file`` are specified, both
                ``request`` and ``request_file`` are not provided, or ``request_file``
                is specified but does not exist.
            requests.exception.RequestException: If the request fails with an exception.
        """
        if self.container_is_running is False:
            raise RuntimeError(
                "The local endpoint is not serving traffic. Please call `serve()`."
            )

        if request is not None and request_file is not None:
            raise ValueError(
                "request and request_file can not be specified at the same time."
            )
        if request is None and request_file is None:
            raise ValueError("One of request and request_file needs to be specified.")

        try:
            url = f"http://localhost:{self.assigned_host_port}{self.serving_container_predict_route}"
            if request is not None:
                response = requests.post(url, data=request, headers=headers)
            elif request_file is not None:
                if not Path(request_file).expanduser().resolve().exists():
                    raise ValueError(f"request_file does not exist: {request_file}.")
                with open(request_file) as data:
                    response = requests.post(url, data=data, headers=headers)
            return response
        except requests.exceptions.RequestException as exception:
            if verbose:
                _logger.warning(f"Exception during prediction: {exception}")
            raise

    def run_health_check(self, verbose: bool = True) -> requests.models.Response:
        """Runs a health check.

        Args:
            verbose (bool):
                Required. Whether or not print logs if any.

        Returns:
            The health check response.

        Raises:
            RuntimeError: If the local endpoint has been stopped.
            requests.exception.RequestException: If the request fails with an exception.
        """
        if self.container_is_running is False:
            raise RuntimeError(
                "The local endpoint is not serving traffic. Please call `serve()`."
            )

        try:
            url = f"http://localhost:{self.assigned_host_port}{self.serving_container_health_route}"
            response = requests.get(url)
            return response
        except requests.exceptions.RequestException as exception:
            if verbose:
                _logger.warning(f"Exception during health check: {exception}")
            raise

    def print_container_logs(
        self, show_all: bool = False, message: Optional[str] = None
    ) -> None:
        """Prints container logs.

        Args:
            show_all (bool):
                Required. If True, prints all logs since the container starts.
            message (str):
                Optional. The message to be printed before printing the logs.
        """
        start_index = None if show_all else self.log_start_index
        self.log_start_index = run.print_container_logs(
            self.container, start_index=start_index, message=message
        )

    def print_container_logs_if_container_is_not_running(
        self, show_all: bool = False, message: Optional[str] = None
    ) -> None:
        """Prints container logs if the container is not in "running" status.

        Args:
            show_all (bool):
                Required. If True, prints all logs since the container starts.
            message (str):
                Optional. The message to be printed before printing the logs.
        """
        if self.get_container_status() != run.CONTAINER_RUNNING_STATUS:
            self.print_container_logs(show_all=show_all, message=message)

    def get_container_status(self) -> str:
        """Gets the container status.

        Returns:
            The container status. One of restarting, running, paused, exited.
        """
        self.container.reload()
        return self.container.status
