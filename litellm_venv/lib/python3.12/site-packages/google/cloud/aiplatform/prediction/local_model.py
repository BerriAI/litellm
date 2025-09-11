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

from copy import copy
from typing import Dict, List, Optional, Sequence, Type

from google.cloud import aiplatform
from google.cloud.aiplatform import helpers

from google.cloud.aiplatform.compat.types import (
    model as gca_model_compat,
    env_var as gca_env_var_compat,
)

from google.cloud.aiplatform.docker_utils import build
from google.cloud.aiplatform.docker_utils import errors
from google.cloud.aiplatform.docker_utils import local_util
from google.cloud.aiplatform.docker_utils import utils
from google.cloud.aiplatform.prediction import LocalEndpoint
from google.cloud.aiplatform.prediction.handler import Handler
from google.cloud.aiplatform.prediction.handler import PredictionHandler
from google.cloud.aiplatform.prediction.predictor import Predictor
from google.cloud.aiplatform.utils import prediction_utils

from google.protobuf import duration_pb2

DEFAULT_PREDICT_ROUTE = "/predict"
DEFAULT_HEALTH_ROUTE = "/health"
DEFAULT_HTTP_PORT = 8080
_DEFAULT_SDK_REQUIREMENTS = ["google-cloud-aiplatform[prediction]>=1.27.0"]
_DEFAULT_HANDLER_MODULE = "google.cloud.aiplatform.prediction.handler"
_DEFAULT_HANDLER_CLASS = "PredictionHandler"
_DEFAULT_PYTHON_MODULE = "google.cloud.aiplatform.prediction.model_server"


class LocalModel:
    """Class that represents a local model."""

    def __init__(
        self,
        serving_container_spec: Optional[aiplatform.gapic.ModelContainerSpec] = None,
        serving_container_image_uri: Optional[str] = None,
        serving_container_predict_route: Optional[str] = None,
        serving_container_health_route: Optional[str] = None,
        serving_container_command: Optional[Sequence[str]] = None,
        serving_container_args: Optional[Sequence[str]] = None,
        serving_container_environment_variables: Optional[Dict[str, str]] = None,
        serving_container_ports: Optional[Sequence[int]] = None,
        serving_container_grpc_ports: Optional[Sequence[int]] = None,
        serving_container_deployment_timeout: Optional[int] = None,
        serving_container_shared_memory_size_mb: Optional[int] = None,
        serving_container_startup_probe_exec: Optional[Sequence[str]] = None,
        serving_container_startup_probe_period_seconds: Optional[int] = None,
        serving_container_startup_probe_timeout_seconds: Optional[int] = None,
        serving_container_health_probe_exec: Optional[Sequence[str]] = None,
        serving_container_health_probe_period_seconds: Optional[int] = None,
        serving_container_health_probe_timeout_seconds: Optional[int] = None,
    ):
        """Creates a local model instance.

        Args:
            serving_container_spec (aiplatform.gapic.ModelContainerSpec):
                Optional. The container spec of the LocalModel instance.
            serving_container_image_uri (str):
                Optional. The URI of the Model serving container.
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
            serving_container_grpc_ports: Optional[Sequence[int]]=None,
                Declaration of ports that are exposed by the container. Vertex AI sends gRPC
                prediction requests that it receives to the first port on this list. Vertex
                AI also sends liveness and health checks to this port.
                If you do not specify this field, gRPC requests to the container will be
                disabled.
                Vertex AI does not use ports other than the first one listed. This field
                corresponds to the `ports` field of the Kubernetes Containers v1 core API.
            serving_container_deployment_timeout (int):
                Optional. Deployment timeout in seconds.
            serving_container_shared_memory_size_mb (int):
                Optional. The amount of the VM memory to reserve as the shared
                memory for the model in megabytes.
            serving_container_startup_probe_exec (Sequence[str]):
                Optional. Exec specifies the action to take. Used by startup
                probe. An example of this argument would be
                ["cat", "/tmp/healthy"]
            serving_container_startup_probe_period_seconds (int):
                Optional. How often (in seconds) to perform the startup probe.
                Default to 10 seconds. Minimum value is 1.
            serving_container_startup_probe_timeout_seconds (int):
                Optional. Number of seconds after which the startup probe times
                out. Defaults to 1 second. Minimum value is 1.
            serving_container_health_probe_exec (Sequence[str]):
                Optional. Exec specifies the action to take. Used by health
                probe. An example of this argument would be
                ["cat", "/tmp/healthy"]
            serving_container_health_probe_period_seconds (int):
                Optional. How often (in seconds) to perform the health probe.
                Default to 10 seconds. Minimum value is 1.
            serving_container_health_probe_timeout_seconds (int):
                Optional. Number of seconds after which the health probe times
                out. Defaults to 1 second. Minimum value is 1.

        Raises:
            ValueError: If ``serving_container_spec`` is specified but ``serving_container_spec.image_uri``
                is ``None``. Also if ``serving_container_spec`` is None but ``serving_container_image_uri`` is
                ``None``.
        """
        if serving_container_spec:
            if not serving_container_spec.image_uri:
                raise ValueError(
                    "Image uri is required for the serving container spec to initialize a LocalModel instance."
                )

            self.serving_container_spec = serving_container_spec
        else:
            if not serving_container_image_uri:
                raise ValueError(
                    "Serving container image uri is required to initialize a LocalModel instance."
                )

            env = None
            ports = None
            grpc_ports = None
            deployment_timeout = (
                duration_pb2.Duration(seconds=serving_container_deployment_timeout)
                if serving_container_deployment_timeout
                else None
            )
            startup_probe = None
            health_probe = None

            if serving_container_environment_variables:
                env = [
                    gca_env_var_compat.EnvVar(name=str(key), value=str(value))
                    for key, value in serving_container_environment_variables.items()
                ]
            if serving_container_ports:
                ports = [
                    gca_model_compat.Port(container_port=port)
                    for port in serving_container_ports
                ]
            if serving_container_grpc_ports:
                grpc_ports = [
                    gca_model_compat.Port(container_port=port)
                    for port in serving_container_grpc_ports
                ]
            if (
                serving_container_startup_probe_exec
                or serving_container_startup_probe_period_seconds
                or serving_container_startup_probe_timeout_seconds
            ):
                startup_probe_exec = None
                if serving_container_startup_probe_exec:
                    startup_probe_exec = gca_model_compat.Probe.ExecAction(
                        command=serving_container_startup_probe_exec
                    )
                startup_probe = gca_model_compat.Probe(
                    exec=startup_probe_exec,
                    period_seconds=serving_container_startup_probe_period_seconds,
                    timeout_seconds=serving_container_startup_probe_timeout_seconds,
                )
            if (
                serving_container_health_probe_exec
                or serving_container_health_probe_period_seconds
                or serving_container_health_probe_timeout_seconds
            ):
                health_probe_exec = None
                if serving_container_health_probe_exec:
                    health_probe_exec = gca_model_compat.Probe.ExecAction(
                        command=serving_container_health_probe_exec
                    )
                health_probe = gca_model_compat.Probe(
                    exec=health_probe_exec,
                    period_seconds=serving_container_health_probe_period_seconds,
                    timeout_seconds=serving_container_health_probe_timeout_seconds,
                )

            self.serving_container_spec = gca_model_compat.ModelContainerSpec(
                image_uri=serving_container_image_uri,
                command=serving_container_command,
                args=serving_container_args,
                env=env,
                ports=ports,
                grpc_ports=grpc_ports,
                predict_route=serving_container_predict_route,
                health_route=serving_container_health_route,
                deployment_timeout=deployment_timeout,
                shared_memory_size_mb=serving_container_shared_memory_size_mb,
                startup_probe=startup_probe,
                health_probe=health_probe,
            )

    @classmethod
    def build_cpr_model(
        cls,
        src_dir: str,
        output_image_uri: str,
        predictor: Optional[Type[Predictor]] = None,
        handler: Type[Handler] = PredictionHandler,
        base_image: str = "python:3.10",
        requirements_path: Optional[str] = None,
        extra_packages: Optional[List[str]] = None,
        no_cache: bool = False,
    ) -> "LocalModel":
        """Builds a local model from a custom predictor.

        This method builds a docker image to include user-provided predictor, and handler.

        Sample ``src_dir`` contents (e.g. ``./user_src_dir``):

        .. code-block:: python

            user_src_dir/
            |-- predictor.py
            |-- requirements.txt
            |-- user_code/
            |   |-- utils.py
            |   |-- custom_package.tar.gz
            |   |-- ...
            |-- ...

        To build a custom container:

        .. code-block:: python

            local_model = LocalModel.build_cpr_model(
                "./user_src_dir",
                "us-docker.pkg.dev/$PROJECT/$REPOSITORY/$IMAGE_NAME$",
                predictor=$CUSTOM_PREDICTOR_CLASS,
                requirements_path="./user_src_dir/requirements.txt",
                extra_packages=["./user_src_dir/user_code/custom_package.tar.gz"],
            )

        In the built image, user provided files will be copied as follows:

        .. code-block:: python

            container_workdir/
            |-- predictor.py
            |-- requirements.txt
            |-- user_code/
            |   |-- utils.py
            |   |-- custom_package.tar.gz
            |   |-- ...
            |-- ...

        To exclude files and directories from being copied into the built container images, create a
        ``.dockerignore`` file in the ``src_dir``. See
        https://docs.docker.com/engine/reference/builder/#dockerignore-file for more details about
        usage.

        In order to save and restore class instances transparently with Pickle, the class definition
        must be importable and live in the same module as when the object was stored. If you want to
        use Pickle, you must save your objects right under the ``src_dir`` you provide.

        The created CPR images default the number of model server workers to the number of cores.
        Depending on the characteristics of your model, you may need to adjust the number of workers.
        You can set the number of workers with the following environment variables:

        .. code-block:: python

            VERTEX_CPR_WEB_CONCURRENCY:
                The number of the workers. This will overwrite the number calculated by the other
                variables, min(VERTEX_CPR_WORKERS_PER_CORE * number_of_cores, VERTEX_CPR_MAX_WORKERS).
            VERTEX_CPR_WORKERS_PER_CORE:
                The number of the workers per core. The default is 1.
            VERTEX_CPR_MAX_WORKERS:
                The maximum number of workers can be used given the value of VERTEX_CPR_WORKERS_PER_CORE
                and the number of cores.

        If you hit the error showing "model server container out of memory" when you deploy models
        to endpoints, you should decrease the number of workers.

        Args:
            src_dir (str):
                Required. The path to the local directory including all needed files such as
                predictor. The whole directory will be copied to the image.
            output_image_uri (str):
                Required. The image uri of the built image.
            predictor (Type[Predictor]):
                Optional. The custom predictor class consumed by handler to do prediction.
            handler (Type[Handler]):
                Required. The handler class to handle requests in the model server.
            base_image (str):
                Required. The base image used to build the custom images. The base image must
                have python and pip installed where the two commands ``python`` and ``pip`` must be
                available.
            requirements_path (str):
                Optional. The path to the local requirements.txt file. This file will be copied
                to the image and the needed packages listed in it will be installed.
            extra_packages (List[str]):
                Optional. The list of user custom dependency packages to install.
            no_cache (bool):
                Required. Do not use cache when building the image. Using build cache usually
                reduces the image building time. See
                https://docs.docker.com/develop/develop-images/dockerfile_best-practices/#leverage-build-cache
                for more details.

        Returns:
            local model: Instantiated representation of the local model.

        Raises:
            ValueError: If handler is ``None`` or if handler is ``PredictionHandler`` but predictor is ``None``.
        """
        handler_module = _DEFAULT_HANDLER_MODULE
        handler_class = _DEFAULT_HANDLER_CLASS
        if handler is None:
            raise ValueError("A handler must be provided but handler is None.")
        elif handler == PredictionHandler:
            if predictor is None:
                raise ValueError(
                    "PredictionHandler must have a predictor class but predictor is None."
                )
        else:
            handler_module, handler_class = prediction_utils.inspect_source_from_class(
                handler, src_dir
            )
        environment_variables = {
            "HANDLER_MODULE": handler_module,
            "HANDLER_CLASS": handler_class,
        }

        predictor_module = None
        predictor_class = None
        if predictor is not None:
            (
                predictor_module,
                predictor_class,
            ) = prediction_utils.inspect_source_from_class(predictor, src_dir)
            environment_variables["PREDICTOR_MODULE"] = predictor_module
            environment_variables["PREDICTOR_CLASS"] = predictor_class

        is_prebuilt_prediction_image = helpers.is_prebuilt_prediction_container_uri(
            base_image
        )
        _ = build.build_image(
            base_image,
            src_dir,
            output_image_uri,
            python_module=_DEFAULT_PYTHON_MODULE,
            requirements_path=requirements_path,
            extra_requirements=_DEFAULT_SDK_REQUIREMENTS,
            extra_packages=extra_packages,
            exposed_ports=[DEFAULT_HTTP_PORT],
            environment_variables=environment_variables,
            pip_command="pip3" if is_prebuilt_prediction_image else "pip",
            python_command="python3" if is_prebuilt_prediction_image else "python",
            no_cache=no_cache,
        )

        container_spec = gca_model_compat.ModelContainerSpec(
            image_uri=output_image_uri,
            predict_route=DEFAULT_PREDICT_ROUTE,
            health_route=DEFAULT_HEALTH_ROUTE,
        )

        return cls(serving_container_spec=container_spec)

    def deploy_to_local_endpoint(
        self,
        artifact_uri: Optional[str] = None,
        credential_path: Optional[str] = None,
        host_port: Optional[str] = None,
        gpu_count: Optional[int] = None,
        gpu_device_ids: Optional[List[str]] = None,
        gpu_capabilities: Optional[List[List[str]]] = None,
        container_ready_timeout: Optional[int] = None,
        container_ready_check_interval: Optional[int] = None,
    ) -> LocalEndpoint:
        """Deploys the local model instance to a local endpoint.

        An environment variable, ``GOOGLE_CLOUD_PROJECT``, will be set to the project in the global config.
        This is required if the credentials file does not have project specified and used to
        recognize the project by the Cloud Storage client.

        Example 1:

        .. code-block:: python

            with local_model.deploy_to_local_endpoint(
                artifact_uri="gs://path/to/your/model",
                credential_path="local/path/to/your/credentials",
            ) as local_endpoint:
                health_check_response = local_endpoint.run_health_check()
                print(health_check_response, health_check_response.content)

                predict_response = local_endpoint.predict(
                    request='{"instances": [[1, 2, 3, 4]]}',
                    headers={"header-key": "header-value"},
                )
                print(predict_response, predict_response.content)

                local_endpoint.print_container_logs()

        Example 2:

        .. code-block:: python

            local_endpoint = local_model.deploy_to_local_endpoint(
                artifact_uri="gs://path/to/your/model",
                credential_path="local/path/to/your/credentials",
            )
            local_endpoint.serve()

            health_check_response = local_endpoint.run_health_check()
            print(health_check_response, health_check_response.content)

            predict_response = local_endpoint.predict(
                request='{"instances": [[1, 2, 3, 4]]}',
                headers={"header-key": "header-value"},
            )
            print(predict_response, predict_response.content)

            local_endpoint.print_container_logs()
            local_endpoint.stop()

        Args:
            artifact_uri (str):
                Optional. The path to the directory containing the Model artifact and any of its
                supporting files. The path is either a GCS uri or the path to a local directory.
                If this parameter is set to a GCS uri:
                (1) ``credential_path`` must be specified for local prediction.
                (2) The GCS uri will be passed directly to ``Predictor.load``.
                If this parameter is a local directory:
                (1) The directory will be mounted to a default temporary model path.
                (2) The mounted path will be passed to ``Predictor.load``.
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

        Returns:
            A the local endpoint object.
        """
        envs = {env.name: env.value for env in self.serving_container_spec.env}
        ports = [port.container_port for port in self.serving_container_spec.ports]

        return LocalEndpoint(
            serving_container_image_uri=self.serving_container_spec.image_uri,
            artifact_uri=artifact_uri,
            serving_container_predict_route=self.serving_container_spec.predict_route,
            serving_container_health_route=self.serving_container_spec.health_route,
            serving_container_command=self.serving_container_spec.command,
            serving_container_args=self.serving_container_spec.args,
            serving_container_environment_variables=envs,
            serving_container_ports=ports,
            credential_path=credential_path,
            host_port=host_port,
            gpu_count=gpu_count,
            gpu_device_ids=gpu_device_ids,
            gpu_capabilities=gpu_capabilities,
            container_ready_timeout=container_ready_timeout,
            container_ready_check_interval=container_ready_check_interval,
        )

    def get_serving_container_spec(self) -> aiplatform.gapic.ModelContainerSpec:
        """Returns the container spec for the image.

        Returns:
            The serving container spec of this local model instance.
        """
        return self.serving_container_spec

    def copy_image(self, dst_image_uri: str) -> "LocalModel":
        """Copies the image to another image uri.

        Args:
            dst_image_uri (str):
                The destination image uri to copy the image to.

        Returns:
            local model: Instantiated representation of the local model with the copied
            image.

        Raises:
            DockerError: If the command fails.
        """
        self.pull_image_if_not_exists()

        command = [
            "docker",
            "tag",
            f"{self.serving_container_spec.image_uri}",
            f"{dst_image_uri}",
        ]
        return_code = local_util.execute_command(command)
        if return_code != 0:
            errors.raise_docker_error_with_command(command, return_code)

        new_container_spec = copy(self.serving_container_spec)
        new_container_spec.image_uri = dst_image_uri

        return LocalModel(new_container_spec)

    def push_image(self) -> None:
        """Pushes the image to a registry.

        If you hit permission errors while calling this function, please refer to
        https://cloud.google.com/artifact-registry/docs/docker/authentication to set
        up the authentication.

        For Artifact Registry, the repository must be created before you are able to
        push images to it. Otherwise, you will hit the error, "Repository {REPOSITORY} not found".
        To create Artifact Registry repositories, use UI or call the following gcloud command.

        .. code-block:: bash

            gcloud artifacts repositories create {REPOSITORY} \
                --project {PROJECT} \
                --location {REGION} \
                --repository-format docker

        See https://cloud.google.com/artifact-registry/docs/manage-repos#create for more details.

        If you hit a "Permission artifactregistry.repositories.uploadArtifacts denied" error,
        set up authentication for Docker.

        .. code-block:: bash

            gcloud auth configure-docker {REPOSITORY}

        See https://cloud.google.com/artifact-registry/docs/docker/authentication for mode details.

        Raises:
            ValueError: If the image uri is not a container registry or artifact registry
                uri.
            DockerError: If the command fails.
        """
        if (
            prediction_utils.is_registry_uri(self.serving_container_spec.image_uri)
            is False
        ):
            raise ValueError(
                "The image uri must be a container registry or artifact registry "
                f"uri but it is: {self.serving_container_spec.image_uri}."
            )

        command = ["docker", "push", f"{self.serving_container_spec.image_uri}"]
        return_code = local_util.execute_command(command)
        if return_code != 0:
            errors.raise_docker_error_with_command(command, return_code)

    def pull_image_if_not_exists(self):
        """Pulls the image if the image does not exist locally.

        Raises:
            DockerError: If the command fails.
        """
        if not utils.check_image_exists_locally(self.serving_container_spec.image_uri):
            command = [
                "docker",
                "pull",
                f"{self.serving_container_spec.image_uri}",
            ]
            return_code = local_util.execute_command(command)
            if return_code != 0:
                errors.raise_docker_error_with_command(command, return_code)
