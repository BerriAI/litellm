# -*- coding: utf-8 -*-

# Copyright 2023 Google LLC
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
import dataclasses
from typing import List, Optional
from vertexai.preview._workflow.serialization_engine import (
    serializers_base,
)


@dataclasses.dataclass
class _BaseConfig:
    """A class that holds configuration that can be shared across different remote services.

    Attributes:
        display_name (str):
            The display name of the remote job.
        staging_bucket (str):
            Base GCS directory of the remote job. All the input and
            output artifacts will be saved here. If not provided a timestamped
            directory in the default staging bucket will be used.
        container_uri (str):
            Uri of the training container image to use for remote job.
            Support images in Artifact Registry, Container Registry, or Docker Hub.
        machine_type (str):
            The type of machine to use for remote training.
        accelerator_type (str):
            Hardware accelerator type. One of ACCELERATOR_TYPE_UNSPECIFIED,
            NVIDIA_TESLA_A100, NVIDIA_TESLA_P100, NVIDIA_TESLA_V100,
            NVIDIA_TESLA_K80, NVIDIA_TESLA_T4, NVIDIA_TESLA_P4
        accelerator_count (int):
            The number of accelerators to attach to a worker replica.
        worker_pool_specs (vertexai.preview.developer.remote_specs.WorkerPoolSpecs):
            The worker pool specs configuration for a remote job.
    """

    display_name: Optional[str] = None
    staging_bucket: Optional[str] = None
    container_uri: Optional[str] = None
    machine_type: Optional[str] = None
    accelerator_type: Optional[str] = None
    accelerator_count: Optional[int] = None
    worker_pool_specs: Optional[
        "vertexai.preview.developer.remote_specs.WorkerPoolSpecs"  # noqa: F821
    ] = None


@dataclasses.dataclass
class RemoteConfig(_BaseConfig):
    """A class that holds the configuration for Vertex remote training.

    Example usage:
        # Specify requirements
        model.train.vertex.remote_config.requirements = [
            "requirement1==1.0.0",
            "requirement2>=2.0.1",
        ]

        # Specify custom commands to run before installing other requirements
        model.train.vertex.remote_config.custom_commands = [
            "export SOME_CONSTANT=value",
        ]

        # Specify the extra parameters needed for serializing objects.
        from vertexai.preview.developer import SerializerArgs

        # You can put all the hashable objects with their arguments in the
        # SerializerArgs all at once in a dict. Here we assume "model" is
        # hashable.
        model.train.vertex.remote_config.serializer_args = SerializerArgs({
                model: {
                        "extra_serializer_param1_for_model": param1_value,
                        "extra_serializer_param2_for_model": param2_value,
                        },
                hashable_obj2: {
                        "extra_serializer_param1_for_hashable2": param1_value,
                        "extra_serializer_param2_for_hashable2": param2_value,
                        },
        })
        # Or if the object to be serialized is unhashable, put them into the
        # serializer_args one by one. If this is the only use case, there is
        # no need to import `SerializerArgs`. Here we assume "X_train" and
        # "y_train" is not hashable.
        model.train.vertex.remote_config.serializer_args[X_train] = {
                "extra_serializer_param1_for_X_train": param1_value,
                "extra_serializer_param2_for_X_train": param2_value,
            },
        model.train.vertex.remote_config.serializer_args[y_train] = {
                "extra_serializer_param1_for_y_train": param1_value,
                "extra_serializer_param2_for_y_train": param2_value,
            }

        # Train the model as usual
        model.train(X_train, y_train)

    Attributes:
        enable_cuda (bool):
            When set to True, Vertex will automatically choose a GPU image and
            accelerators for the remote job and train the model on cuda devices.
            You can also specify the image and accelerators by yourself through
            `container_uri`, `accelerator_type`, `accelerator_count`.
            Supported frameworks: keras, torch.nn.Module
            Default configs:
            container_uri=(
                "pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime"
                or
                "us-docker.pkg.dev/vertex-ai/training/tf-gpu.2-11.py310:latest"
            )
            machine_type="n1-standard-16"
            accelerator_type="NVIDIA_TESLA_P100"
            accelerator_count=1
        enable_distributed (bool):
            When set to True, Vertex will automatically choose a GPU or CPU
            distributed training configuration depending on the value of `enable_cuda`.
            You can also specify a custom configuration by yourself through `worker_pool_specs`.
            Supported frameworks: keras (requires TensorFlow >= 2.12.0), torch.nn.Module
            Default configs:
            If `enable_cuda` = True, for both the `chief` and `worker` specs:
                machine_type="n1-standard-16"
                accelerator_type="NVIDIA_TESLA_P100"
                accelerator_count=1
            If `enable_cuda` = False, for both the `chief` and `worker` specs:
                machine_type="n1-standard-4"
                replica_count=1
        enable_full_logs (bool):
            When set to True, all the logs from the remote job will be shown locally.
            Otherwise, only training related logs will be shown.
        service_account (str):
            Specifies the service account for running the remote job. To use
            autologging feature, you need to set it to "gce", which refers
            to the GCE service account, or set it to another service account.
            Please make sure your own service account has the Storage Admin role
            and Vertex AI User role.
        requirements (List[str]):
            List of python packages dependencies that will be installed in the remote
            job environment. In most cases Vertex will handle the installation of
            dependencies that are required for running the remote job. You can use
            this field to specify extra packages to install in the remote environment.
        custom_commands (List[str]):
            List of custom commands to be run in the remote job environment.
            These commands will be run before the requirements are installed.
        serializer_args: serializers_base.SerializerArgs:
            Map from object to extra arguments when serializing the object. The extra
            arguments is a dictionary from the argument names to the argument values.
    """

    enable_cuda: bool = False
    enable_distributed: bool = False
    enable_full_logs: bool = False
    service_account: Optional[str] = None
    requirements: List[str] = dataclasses.field(default_factory=list)
    custom_commands: List[str] = dataclasses.field(default_factory=list)
    serializer_args: serializers_base.SerializerArgs = dataclasses.field(
        default_factory=serializers_base.SerializerArgs
    )


@dataclasses.dataclass
class DistributedTrainingConfig(_BaseConfig):
    """A class that holds the configs for a distributed training remote job.

    Attributes:
        replica_count (int):
            The number of worker replicas. Assigns 1 chief replica and
            replica_count - 1 worker replicas.
        boot_disk_type (str):
            Type of the boot disk (default is `pd-ssd`).
            Valid values: `pd-ssd` (Persistent Disk Solid State Drive) or
            `pd-standard` (Persistent Disk Hard Disk Drive).
        boot_disk_size_gb (int):
            Size in GB of the boot disk (default is 100GB).
            boot disk size must be within the range of [100, 64000].
    """

    replica_count: Optional[int] = None
    boot_disk_type: Optional[str] = None
    boot_disk_size_gb: Optional[int] = None


@dataclasses.dataclass
class VertexConfig:
    """A class that holds the configuration for the method wrapped by Vertex.

    Attributes:
        remote (bool):
            Whether or not this method will be executed remotely on Vertex. If not
            set, Vertex will check the remote setting in `vertexai.preview.init(...)`
        remote_config (RemoteConfig):
            A class that holds the configuration for the remote job.
    """

    remote: Optional[bool] = None
    remote_config: RemoteConfig = dataclasses.field(default_factory=RemoteConfig)

    def set_config(
        self,
        display_name: Optional[str] = None,
        staging_bucket: Optional[str] = None,
        container_uri: Optional[str] = None,
        machine_type: Optional[str] = None,
        accelerator_type: Optional[str] = None,
        accelerator_count: Optional[int] = None,
        worker_pool_specs: Optional[
            "vertexai.preview.developer.remote_specs.WorkerPoolSpecs"  # noqa: F821
        ] = None,
        enable_cuda: bool = False,
        enable_distributed: bool = False,
        enable_full_logs: bool = False,
        service_account: Optional[str] = None,
        requirements: List[str] = [],
        custom_commands: List[str] = [],
        replica_count: Optional[int] = None,
        boot_disk_type: Optional[str] = None,
        boot_disk_size_gb: Optional[int] = None,
    ):
        """Sets configuration attributes for a remote job.

        Calling this will overwrite any previously set job configuration attributes.

        Example usage:
            vertexai.init(
                project=_TEST_PROJECT,
                location=_TEST_LOCATION,
                staging_bucket=_TEST_BUCKET_NAME,
            )
            vertexai.preview.init(remote=True)

            LogisticRegression = vertexai.preview.remote(_logistic.LogisticRegression)
            model = LogisticRegression()

            model.fit.vertex.set_config(
                display_name="my-display-name",
                staging_bucket="gs://my-bucket",
                container_uri="gcr.io/custom-image,
            )

        Args:
            display_name (str):
                The display name of the remote job.
            staging_bucket (str):
                Base GCS directory of the remote job. All the input and
                output artifacts will be saved here. If not provided a timestamped
                directory in the default staging bucket will be used.
            container_uri (str):
                Uri of the training container image to use for remote job.
                Support images in Artifact Registry, Container Registry, or Docker Hub.
            machine_type (str):
                The type of machine to use for remote training.
            accelerator_type (str):
                Hardware accelerator type. One of ACCELERATOR_TYPE_UNSPECIFIED,
                NVIDIA_TESLA_A100, NVIDIA_TESLA_P100, NVIDIA_TESLA_V100,
                NVIDIA_TESLA_K80, NVIDIA_TESLA_T4, NVIDIA_TESLA_P4
            accelerator_count (int):
                The number of accelerators to attach to a worker replica.
            worker_pool_specs (vertexai.preview.developer.remote_specs.WorkerPoolSpecs):
                The worker pool specs configuration for a remote job.
            enable_cuda (bool):
                When set to True, Vertex will automatically choose a GPU image and
                accelerators for the remote job and train the model on cuda devices.
                This parameter is specifically for TrainingConfig.
                You can also specify the image and accelerators by yourself through
                `container_uri`, `accelerator_type`, `accelerator_count`.
                Supported frameworks: keras, torch.nn.Module
                Default configs:
                container_uri="pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime" or "tensorflow/tensorflow:2.12.0-gpu"
                machine_type="n1-standard-16"
                accelerator_type="NVIDIA_TESLA_P100"
                accelerator_count=1
            enable_distributed (bool):
                When set to True, Vertex will automatically choose a GPU or CPU
                distributed training configuration depending on the value of `enable_cuda`.
                You can also specify a custom configuration by yourself through `worker_pool_specs`.
                This parameter is specifically for TrainingConfig.
                Supported frameworks: keras (requires TensorFlow >= 2.12.0), torch.nn.Module
                Default configs:
                If `enable_cuda` = True, for both the `chief` and `worker` specs:
                    machine_type="n1-standard-16"
                    accelerator_type="NVIDIA_TESLA_P100"
                    accelerator_count=1
                If `enable_cuda` = False, for both the `chief` and `worker` specs:
                    machine_type="n1-standard-4"
                    replica_count=1
            enable_full_logs (bool):
                When set to True, all the logs from the remote job will be shown locally.
                Otherwise, only training related logs will be shown.
            service_account (str):
                Specifies the service account for running the remote job. To use
                autologging feature, you need to set it to "gce", which refers
                to the GCE service account, or set it to another service account.
                Please make sure your own service account has the Storage Admin role
                and Vertex AI User role. This parameter is specifically for TrainingConfig.
            requirements (List[str]):
                List of python packages dependencies that will be installed in the remote
                job environment. In most cases Vertex will handle the installation of
                dependencies that are required for running the remote job. You can use
                this field to specify extra packages to install in the remote environment.
                This parameter is specifically for TrainingConfig.
            custom_commands (List[str]):
                List of custom commands to be run in the remote job environment.
                These commands will be run before the requirements are installed.
            replica_count (int):
                The number of worker replicas. Assigns 1 chief replica and
                replica_count - 1 worker replicas. This is specifically for
                DistributedTrainingConfig.
            boot_disk_type (str):
                Type of the boot disk (default is `pd-ssd`).
                Valid values: `pd-ssd` (Persistent Disk Solid State Drive) or
                `pd-standard` (Persistent Disk Hard Disk Drive). This is specifically for
                DistributedTrainingConfig.
            boot_disk_size_gb (int):
                Size in GB of the boot disk (default is 100GB).
                boot disk size must be within the range of [100, 64000]. This is specifically for
                DistributedTrainingConfig.
        """

        # locals() contains a 'self' key in addition to function args
        kwargs = locals()

        config = self.remote_config.__class__()

        for config_arg in kwargs:
            if hasattr(config, config_arg):
                setattr(config, config_arg, kwargs[config_arg])

            # raise if a value was passed for an unsupported config attribute (i.e. boot_disk_type on TrainingConfig)
            elif config_arg != "self" and kwargs[config_arg]:
                raise ValueError(
                    f"{type(self.remote_config)} has no attribute {config_arg}."
                )

        self.remote_config = config


@dataclasses.dataclass
class PersistentResourceConfig:
    """A class that holds persistent resource configuration during initialization.

    Attributes:
        name (str):
            The cluster name of the remote job. This value may be up to 63
            characters, and valid characters are `[a-z0-9_-]`. The first character
            cannot be a number or hyphen.
        resource_pool_specs (vertexai.preview.developer.remote_specs.ResourcePoolSpecs):
            The worker pool specs configuration for a remote job.
        service_account (str):
            If intended for experiment autologging, this service account should
            be specified and consistent with per instance service account, which
            is configured in `model.fit.vertex.remote_config.service_account`.
        disable (bool):
            By default is False, meaning the remote execution runs on
            the persistent cluster. If users want to disable it (so the remote
            execution runs on an ephemeral cluster), set it as True.
    """

    name: Optional[str] = None
    resource_pools: Optional[
        "vertexai.preview.developer.remote_specs.ResourcePool"  # noqa: F821
    ] = None
    service_account: Optional[str] = None
    disable: Optional[bool] = False
