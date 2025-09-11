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
"""Remote container training and helper functions.
"""
from typing import Any, Dict, List
import uuid

from google.cloud import aiplatform
from google.cloud.aiplatform.utils import worker_spec_utils
import vertexai
from vertexai.preview._workflow import shared
from vertexai.preview.developer import remote_specs
from vertexai.preview._workflow.shared import model_utils

# job_dir container argument name
_JOB_DIR = "job_dir"

# Worker pool specs default value constants
_DEFAULT_REPLICA_COUNT: int = 1
_DEFAULT_MACHINE_TYPE: str = "n1-standard-4"
_DEFAULT_ACCELERATOR_COUNT: int = 0
_DEFAULT_ACCELERATOR_TYPE: str = "ACCELERATOR_TYPE_UNSPECIFIED"
_DEFAULT_BOOT_DISK_TYPE: str = "pd-ssd"
_DEFAULT_BOOT_DISK_SIZE_GB: int = 100

# Custom job default name
_DEFAULT_DISPLAY_NAME = "remote-fit"


def _generate_worker_pool_specs(
    image_uri: str,
    inputs: List[str],
    replica_count: int = _DEFAULT_REPLICA_COUNT,
    machine_type: str = _DEFAULT_MACHINE_TYPE,
    accelerator_count: int = _DEFAULT_ACCELERATOR_COUNT,
    accelerator_type: str = _DEFAULT_ACCELERATOR_TYPE,
    boot_disk_type: str = _DEFAULT_BOOT_DISK_TYPE,
    boot_disk_size_gb: int = _DEFAULT_BOOT_DISK_SIZE_GB,
) -> List[Dict[str, Any]]:
    """Helper function to generate worker pool specs for CustomJob.

    TODO(b/278786170): Use customized worker_pool_specs to specify
    replica_count, machine types, number/type of worker pools, etc. for
    distributed training.

    Args:
        image_uri (str):
            Required. The docker image uri for CustomJob.
        inputs (List[str]):
            Required. A list of inputs for CustomJob. Each item would look like
            "--arg_0=value_for_arg_0".
        replica_count (int):
            Optional. The number of worker replicas. Assigns 1 chief replica and
            replica_count - 1 worker replicas.
        machine_type (str):
            Optional. The type of machine to use for training.
        accelerator_count (int):
            Optional. The number of accelerators to attach to a worker replica.
        accelerator_type (str):
            Optional. Hardware accelerator type. One of
            ACCELERATOR_TYPE_UNSPECIFIED, NVIDIA_TESLA_K80, NVIDIA_TESLA_P100,
            NVIDIA_TESLA_V100, NVIDIA_TESLA_P4, NVIDIA_TESLA_T4
        boot_disk_type (str):
            Optional. Type of the boot disk (default is `pd-ssd`).
            Valid values: `pd-ssd` (Persistent Disk Solid State Drive) or
            `pd-standard` (Persistent Disk Hard Disk Drive).
        boot_disk_size_gb (int):
            Optional. Size in GB of the boot disk (default is 100GB).
            boot disk size must be within the range of [100, 64000].

    Returns:
        A list of worker pool specs in the form of dictionaries. For
        replica = 1, there is one worker pool spec. For replica > 1, there are
        two worker pool specs.

    Raises:
        ValueError if replica_count is less than 1.
    """
    if replica_count < 1:
        raise ValueError(
            "replica_count must be a positive number but is " f"{replica_count}."
        )

    # pylint: disable=protected-access
    worker_pool_specs = worker_spec_utils._DistributedTrainingSpec.chief_worker_pool(
        replica_count=replica_count,
        machine_type=machine_type,
        accelerator_count=accelerator_count,
        accelerator_type=accelerator_type,
        boot_disk_type=boot_disk_type,
        boot_disk_size_gb=boot_disk_size_gb,
    ).pool_specs

    # Attach a container_spec to each worker pool spec.
    for spec in worker_pool_specs:
        spec["container_spec"] = {
            "image_uri": image_uri,
            "args": inputs,
        }

    return worker_pool_specs


# pylint: disable=protected-access
def train(invokable: shared._Invokable):
    """Wrapper function that runs remote container training."""
    training_config = invokable.vertex_config.remote_config

    # user can specify either worker_pool_specs OR machine_type, replica_count etc.
    remote_specs._verify_specified_remote_config_values(
        training_config.worker_pool_specs,
        training_config.machine_type,
        training_config.replica_count,
        training_config.accelerator_type,
        training_config.accelerator_count,
        training_config.boot_disk_type,
        training_config.boot_disk_size_gb,
    )

    staging_bucket = (
        training_config.staging_bucket or vertexai.preview.global_config.staging_bucket
    )
    if not staging_bucket:
        raise ValueError(
            "No default staging bucket set. "
            "Please call `vertexai.init(staging_bucket='gs://my-bucket')."
        )
    input_dir = remote_specs._gen_gcs_path(staging_bucket, model_utils._INPUT_DIR)
    output_dir = remote_specs._gen_gcs_path(staging_bucket, model_utils._OUTPUT_DIR)

    # Creates a complete set of binding.
    instance_binding = invokable.instance._binding
    binding = invokable.bound_arguments.arguments
    for arg in instance_binding:
        binding[arg] = instance_binding[arg]

    # If a container accepts a job_dir argument and the user does not specify
    # it, set job_dir based on the staging bucket.
    if _JOB_DIR in binding and not binding[_JOB_DIR]:
        binding[_JOB_DIR] = remote_specs._gen_gcs_path(
            staging_bucket, model_utils._CUSTOM_JOB_DIR
        )

    # Formats arguments.
    formatted_args = {}
    output_specs = []
    for data in invokable.remote_executor_kwargs["additional_data"]:
        if isinstance(data, remote_specs._InputParameterSpec):
            formatted_args[data.argument_name] = data.format_arg(input_dir, binding)
        elif isinstance(data, remote_specs._OutputParameterSpec):
            formatted_args[data.argument_name] = remote_specs._gen_gcs_path(
                output_dir, data.argument_name
            )
            output_specs.append(data)
        else:
            raise ValueError(f"Invalid data type {type(data)}.")
    inputs = [f"--{key}={val}" for key, val in formatted_args.items()]

    # Launches a custom job.
    display_name = training_config.display_name or _DEFAULT_DISPLAY_NAME
    if training_config.worker_pool_specs:
        worker_pool_specs = remote_specs._prepare_worker_pool_specs(
            worker_pool_specs=training_config.worker_pool_specs,
            image_uri=invokable.remote_executor_kwargs["image_uri"],
            args=inputs,
        )
    else:
        worker_pool_specs = _generate_worker_pool_specs(
            image_uri=invokable.remote_executor_kwargs["image_uri"],
            inputs=inputs,
            replica_count=(training_config.replica_count or _DEFAULT_REPLICA_COUNT),
            machine_type=(training_config.machine_type or _DEFAULT_MACHINE_TYPE),
            accelerator_count=(
                training_config.accelerator_count or _DEFAULT_ACCELERATOR_COUNT
            ),
            accelerator_type=(
                training_config.accelerator_type or _DEFAULT_ACCELERATOR_TYPE
            ),
            boot_disk_type=(training_config.boot_disk_type or _DEFAULT_BOOT_DISK_TYPE),
            boot_disk_size_gb=(
                training_config.boot_disk_size_gb or _DEFAULT_BOOT_DISK_SIZE_GB
            ),
        )

    job = aiplatform.CustomJob(
        display_name=f"{invokable.instance.__class__.__name__}-{display_name}"
        f"-{uuid.uuid4()}",
        worker_pool_specs=worker_pool_specs,
        base_output_dir=remote_specs._gen_gcs_path(
            staging_bucket, model_utils._CUSTOM_JOB_DIR
        ),
        staging_bucket=remote_specs._gen_gcs_path(
            staging_bucket, model_utils._CUSTOM_JOB_DIR
        ),
    )
    job.run()

    # Sets output values from the custom job.
    for data in output_specs:
        deserialized_output = data.deserialize_output(
            formatted_args[data.argument_name]
        )
        invokable.instance.__setattr__(data.name, deserialized_output)

    # Calls the decorated function for post-processing.
    return invokable.method(
        *invokable.bound_arguments.args, **invokable.bound_arguments.kwargs
    )
