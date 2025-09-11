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

import collections
import datetime
import inspect
import logging
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import warnings

from google.api_core import exceptions as api_exceptions
from google.cloud import aiplatform
import vertexai
from google.cloud.aiplatform import base
from google.cloud.aiplatform.preview import jobs
from google.cloud.aiplatform import utils
from google.cloud.aiplatform.metadata import metadata
from google.cloud.aiplatform.utils import resource_manager_utils
from vertexai.preview._workflow import shared
from vertexai.preview._workflow.serialization_engine import (
    any_serializer,
)
from vertexai.preview._workflow.serialization_engine import (
    serializers_base,
)
from vertexai.preview._workflow.shared import constants
from vertexai.preview._workflow.shared import (
    supported_frameworks,
)
from vertexai.preview._workflow.shared import model_utils
from vertexai.preview.developer import remote_specs
from packaging import version


try:
    from importlib import metadata as importlib_metadata
except ImportError:
    import importlib_metadata

try:
    import bigframes as bf
    from bigframes.dataframe import DataFrame

    BigframesData = DataFrame
except ImportError:
    bf = None
    BigframesData = Any


try:
    from google.cloud import logging as cloud_logging
except ImportError:
    cloud_logging = None


_LOGGER = base.Logger("vertexai.remote_execution")
_LOG_POLL_INTERVAL = 5
_LOG_WAIT_INTERVAL = 30


# TODO(b/271855597) Serialize all input args
PASS_THROUGH_ARG_TYPES = [str, int, float, bool]

VERTEX_AI_DEPENDENCY_PATH = (
    f"google-cloud-aiplatform[preview]=={aiplatform.__version__}"
)
VERTEX_AI_DEPENDENCY_PATH_AUTOLOGGING = (
    f"google-cloud-aiplatform[preview,autologging]=={aiplatform.__version__}"
)

_DEFAULT_GPU_WORKER_POOL_SPECS = remote_specs.WorkerPoolSpecs(
    remote_specs.WorkerPoolSpec(1, "n1-standard-16", 1, "NVIDIA_TESLA_P100"),
    remote_specs.WorkerPoolSpec(1, "n1-standard-16", 1, "NVIDIA_TESLA_P100"),
)
_DEFAULT_CPU_WORKER_POOL_SPECS = remote_specs.WorkerPoolSpecs(
    remote_specs.WorkerPoolSpec(1, "n1-standard-4"),
    remote_specs.WorkerPoolSpec(1, "n1-standard-4"),
)


def _get_package_name(requirement: str) -> str:
    """Given a requirement specification, returns the package name."""
    return re.match("[a-zA-Z-_]+", requirement).group()


def _get_package_extras(requirement: str) -> Set:
    """Given a requirement specification, returns the extra component in it."""
    # searching for patterns like [extra1,extra2,...]
    extras = re.search(r"\[.*\]", requirement)
    if extras:
        return set([extra.strip() for extra in extras.group()[1:-1].split(",")])
    return set()


def _add_indirect_dependency_versions(direct_requirements: List[str]) -> List[str]:
    """Helper method to get versions of libraries in the dep tree."""
    versions = {}
    dependencies_and_extras = collections.deque([])
    direct_deps_packages = set()
    for direct_requirement in direct_requirements:
        package_name = _get_package_name(direct_requirement)
        extras = _get_package_extras(direct_requirement)
        direct_deps_packages.add(package_name)
        try:
            versions[package_name] = importlib_metadata.version(package_name)
            dependencies_and_extras.append((package_name, extras))
        except importlib_metadata.PackageNotFoundError:
            pass

    while dependencies_and_extras:
        dependency, extras = dependencies_and_extras.popleft()
        child_requirements = importlib_metadata.requires(dependency)
        if not child_requirements:
            continue
        for child_requirement in child_requirements:
            child_dependency = _get_package_name(child_requirement)
            child_dependency_extras = _get_package_extras(child_requirement)
            if child_dependency not in versions:
                if "extra" in child_requirement:
                    # Matching patter "extra == 'extra_component'" in a requirement
                    # specification like
                    # "dependency_name (>=1.0.0) ; extra == 'extra_component'"
                    extra_component = (
                        re.search(r"extra == .*", child_requirement)
                        .group()[len("extra == ") :]
                        .strip("'")
                    )
                    # If the corresponding extra_component is not in the needed
                    # extras set of the parent dependency, skip this package
                    if extra_component not in extras:
                        continue
                try:
                    versions[child_dependency] = importlib_metadata.version(
                        child_dependency
                    )
                    dependencies_and_extras.append(
                        (child_dependency, child_dependency_extras)
                    )
                except importlib_metadata.PackageNotFoundError:
                    pass

    return [
        "==".join([package_name, package_version]) if package_version else package_name
        for package_name, package_version in versions.items()
        if package_name not in direct_deps_packages
    ] + direct_requirements


def _create_worker_pool_specs(
    machine_type: str,
    command: str,
    image_uri: str,
    replica_count: int = 1,
    accelerator_type: Optional[str] = None,
    accelerator_count: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Helper method to create worker pool specs for CustomJob."""
    worker_pool_specs = [
        {
            "machine_spec": {
                "machine_type": machine_type,
                "accelerator_type": accelerator_type,
                "accelerator_count": accelerator_count,
            },
            "replica_count": replica_count,
            "container_spec": {
                "image_uri": image_uri,
                "command": command,
                "args": [],
            },
        }
    ]
    return worker_pool_specs


def _get_worker_pool_specs(
    config: shared.configs.RemoteConfig, image_uri: str, command: List[str]
) -> List[Dict[str, Any]]:
    """Helper method to return worker_pool_specs based on user specification in training config."""
    if config.enable_distributed:
        if config.worker_pool_specs:
            # validate user-specified worker_pool_specs support distributed training.
            # must be single worker, multi-GPU OR multi-worker, single/multi-GPU
            if (
                config.worker_pool_specs.chief.accelerator_count < 2
                and not config.worker_pool_specs.worker
            ):
                raise ValueError(
                    "`enable_distributed=True` in Vertex config, but `worker_pool_specs` do not support distributed training."
                )
            return remote_specs._prepare_worker_pool_specs(
                config.worker_pool_specs, image_uri, command, args=[]
            )
        else:
            default_worker_pool_specs = (
                _DEFAULT_GPU_WORKER_POOL_SPECS
                if config.enable_cuda
                else _DEFAULT_CPU_WORKER_POOL_SPECS
            )
            return remote_specs._prepare_worker_pool_specs(
                default_worker_pool_specs, image_uri, command, args=[]
            )

    if config.worker_pool_specs:
        warnings.warn(
            "config.worker_pool_specs will not take effect since `enable_distributed=False`."
        )

    if config.enable_cuda:
        default_machine_type = "n1-standard-16"
        default_accelerator_type = "NVIDIA_TESLA_P100"
        default_accelerator_count = 1
    else:
        default_machine_type = "n1-standard-4"
        default_accelerator_type = None
        default_accelerator_count = None

    machine_type = config.machine_type or default_machine_type
    accelerator_type = config.accelerator_type or default_accelerator_type
    accelerator_count = config.accelerator_count or default_accelerator_count

    return _create_worker_pool_specs(
        machine_type=machine_type,
        command=command,
        image_uri=image_uri,
        accelerator_type=accelerator_type,
        accelerator_count=accelerator_count,
    )


def _common_update_model_inplace(old_estimator, new_estimator):
    for attr_name, attr_value in new_estimator.__dict__.items():
        if not attr_name.startswith("__") and not inspect.ismethod(
            getattr(old_estimator, attr_name, None)
        ):
            setattr(old_estimator, attr_name, attr_value)


def _update_sklearn_model_inplace(old_estimator, new_estimator):
    _common_update_model_inplace(old_estimator, new_estimator)


def _update_torch_model_inplace(old_estimator, new_estimator):
    # make sure estimators are on the same device
    device = next(old_estimator.parameters()).device
    new_estimator.to(device)
    _common_update_model_inplace(old_estimator, new_estimator)


def _update_lightning_trainer_inplace(old_estimator, new_estimator):
    _common_update_model_inplace(old_estimator, new_estimator)


def _update_keras_model_inplace(old_estimator, new_estimator):
    import tensorflow as tf

    @tf.__internal__.tracking.no_automatic_dependency_tracking
    def _no_tracking_setattr(instance, name, value):
        setattr(instance, name, value)

    for attr_name, attr_value in new_estimator.__dict__.items():
        if not attr_name.startswith("__") and not inspect.ismethod(
            getattr(old_estimator, attr_name, None)
        ):
            # for Keras model, we update self's attributes with a decorated
            # setattr. See b/277939758 for the details.
            _no_tracking_setattr(old_estimator, attr_name, attr_value)


def _get_service_account(
    config: shared.configs.RemoteConfig,
    autolog: bool,
) -> Optional[str]:
    """Helper method to get service account from RemoteConfig."""
    service_account = (
        config.service_account or vertexai.preview.global_config.service_account
    )
    if service_account:
        if service_account.lower() == "gce":
            project = vertexai.preview.global_config.project
            project_number = resource_manager_utils.get_project_number(project)
            return f"{project_number}-compute@developer.gserviceaccount.com"
        else:
            return service_account
    else:
        if autolog:
            raise ValueError(
                "Service account has to be provided for autologging. You can "
                "either use your own service account by setting "
                "`model.<your-method>.vertex.remote_config.service_account = <your-service-account>`, "
                "or use the GCE service account by setting "
                "`model.<your-method>.vertex.remote_config.service_account = 'GCE'`."
            )
        else:
            return None


def _dedupe_requirements(requirements: List[str]) -> List[str]:
    """Helper method to deduplicate requirements by the package name.

    Args:
        requirements (List[str]):
            Required. A list of python packages. Can be either "my-package" or
            "my-package==1.0.0".

    Returns:
        A list of unique python packages. if duplicate in the original list, will
        keep the first one.
    """
    res = []
    req_names = set()
    for req in requirements:
        req_name = req.split("==")[0]
        if req_name not in req_names:
            req_names.add(req_name)
            res.append(req)

    return res


def _get_remote_logs(
    job_id: str,
    logger: "google.cloud.logging.Logger",  # noqa: F821
    log_time: datetime.datetime,
    log_level: str = "INFO",
    is_training_log: bool = False,
) -> Tuple[datetime.datetime, bool]:
    """Helper method to get CustomJob logs from Cloud Logging.

    Args:
        job_id (str):
            Required. The resource id of the CustomJob.
        logger (cloud_logging.Logger):
            Required. A google-cloud-logging Logger object corresponding to the
            CustomJob.
        log_time (datetime.datetime):
            Required. Logs generated after this time will get pulled.
        log_level (str):
            Optional. Logs greater than or equal to this level will get pulled.
            Default is `INFO` level.
        is_training_log (bool):
            Optional. Indicates if logs after the `log_time` are training logs.

    Returns:
        A tuple indicates the end time of logs and whether the training log has
        started.
    """
    filter_msg = [
        f"resource.labels.job_id={job_id}",
        f"severity>={log_level}",
        f'timestamp>"{log_time.isoformat()}"',
    ]
    filter_msg = " AND ".join(filter_msg)
    try:
        entries = logger.list_entries(
            filter_=filter_msg, order_by=cloud_logging.ASCENDING
        )
        for entry in entries:
            log_time = entry.timestamp
            message = entry.payload["message"]
            if constants._START_EXECUTION_MSG in message:
                is_training_log = True
            if is_training_log:
                _LOGGER.log(getattr(logging, entry.severity), message)
            if constants._END_EXECUTION_MSG in message:
                is_training_log = False

        return log_time, is_training_log

    except api_exceptions.ResourceExhausted:
        _LOGGER.warning(
            "Reach the limit for reading cloud logs per minute. "
            f"Will try again in {_LOG_WAIT_INTERVAL} seconds."
        )
        time.sleep(_LOG_WAIT_INTERVAL - _LOG_POLL_INTERVAL)

        return log_time, is_training_log

    except api_exceptions.PermissionDenied as e:
        _LOGGER.warning(
            f"Failed to get logs due to: {e}. "
            "Remote execution logging is disabled. "
            "Please add 'Logging Admin' role to your principal."
        )

        return None, None


def _get_remote_logs_until_complete(
    job: Union[str, aiplatform.CustomJob],
    start_time: Optional[datetime.datetime] = None,
    system_logs: bool = False,
):
    """Helper method to get CustomJob logs in real time until the job is complete.

    Args:
        job (Union[str, aiplatform.CustomJob]):
            Required. A CustomJob ID or `aiplatform.CustomJob` object.
        start_time (datetime.datetime):
            Optional. Get logs generated after this start time. Default is the
            start time of the CustomJob or the current time.
        system_logs (bool):
            Optional. If set to True, all the logs from remote job will be logged
            locally. Otherwise, only training logs will be shown.

    """
    if isinstance(job, str):
        job = aiplatform.CustomJob.get(job)

    if not cloud_logging:
        _LOGGER.warning(
            "google-cloud-logging is not installed, remote execution logging is disabled. "
            "To enable logs, call `pip install google-cloud-aiplatform[preview]`."
        )
        while job.state not in jobs._JOB_COMPLETE_STATES:
            time.sleep(_LOG_POLL_INTERVAL)

        return

    logging_client = cloud_logging.Client(project=job.project)
    # TODO(b/295375379): support remote distributed training logs
    logger = logging_client.logger("workerpool0-0")

    previous_time = (
        start_time or job.start_time or datetime.datetime.now(tz=datetime.timezone.utc)
    )
    is_training_log = system_logs

    while job.state not in jobs._JOB_COMPLETE_STATES:
        if previous_time:
            previous_time, is_training_log = _get_remote_logs(
                job_id=job.name,
                logger=logger,
                log_time=previous_time,
                log_level="INFO",
                is_training_log=is_training_log,
            )
        time.sleep(_LOG_POLL_INTERVAL)

    if previous_time:
        _get_remote_logs(
            job_id=job.name,
            logger=logger,
            log_time=previous_time,
            log_level="INFO",
            is_training_log=is_training_log,
        )


def _set_job_labels(method_name: str) -> Dict[str, str]:
    """Helper method to set the label for the CustomJob.

    Remote training, feature transform, and prediction jobs should each have
    different labels.

    Args:
        method_Name (str):
            Required. The method name used to invoke the remote job.

    Returns:
        A dictionary of the label key/value to use for the CustomJob.
    """

    if method_name in supported_frameworks.REMOTE_TRAINING_STATEFUL_OVERRIDE_LIST:
        return {"trained_by_vertex_ai": "true"}

    if method_name in supported_frameworks.REMOTE_TRAINING_FUNCTIONAL_OVERRIDE_LIST:
        return {"feature_transformed_by_vertex_ai": "true"}

    if method_name in supported_frameworks.REMOTE_PREDICTION_OVERRIDE_LIST:
        return {"predicted_by_vertex_ai": "true"}


def remote_training(invokable: shared._Invokable, rewrapper: Any):
    """Wrapper function that makes a method executable by Vertex CustomJob."""

    self = invokable.instance
    method = invokable.method
    method_name = method.__name__
    bound_args = invokable.bound_arguments
    config = invokable.vertex_config.remote_config
    serializer_args = invokable.vertex_config.remote_config.serializer_args
    if not isinstance(serializer_args, serializers_base.SerializerArgs):
        raise ValueError("serializer_args must be an instance of SerializerArgs.")

    autolog = vertexai.preview.global_config.autolog
    service_account = _get_service_account(config, autolog=autolog)
    if (
        autolog
        and vertexai.preview.global_config.cluster is not None
        and (service_account != vertexai.preview.global_config.cluster.service_account)
    ):
        raise ValueError(
            f"The service account for autologging ({service_account}) is mismatched with the cluster's service account ({vertexai.preview.global_config.service_account}). "
        )
    if autolog:
        vertex_requirements = [
            VERTEX_AI_DEPENDENCY_PATH_AUTOLOGGING,
            "absl-py==1.4.0",
        ]
    else:
        vertex_requirements = [
            VERTEX_AI_DEPENDENCY_PATH,
            "absl-py==1.4.0",
        ]

    requirements = []
    custom_commands = []

    enable_cuda = config.enable_cuda

    # TODO(b/274979556): consider other approaches to pass around the primitives
    pass_through_int_args = {}
    pass_through_float_args = {}
    pass_through_str_args = {}
    pass_through_bool_args = {}
    serialized_args = {}

    for arg_name, arg_value in bound_args.arguments.items():
        if arg_name == "self":
            pass
        elif isinstance(arg_value, int):
            pass_through_int_args[arg_name] = arg_value
        elif isinstance(arg_value, float):
            pass_through_float_args[arg_name] = arg_value
        elif isinstance(arg_value, str):
            pass_through_str_args[arg_name] = arg_value
        elif isinstance(arg_value, bool):
            pass_through_bool_args[arg_name] = arg_value
        else:
            serialized_args[arg_name] = arg_value

    # set base gcs path for the remote job
    staging_bucket = (
        config.staging_bucket or vertexai.preview.global_config.staging_bucket
    )
    if not staging_bucket:
        raise ValueError(
            "No default staging bucket set. "
            "Please call `vertexai.init(staging_bucket='gs://my-bucket')."
        )
    remote_job = f"remote-job-{utils.timestamped_unique_name()}"
    remote_job_base_path = os.path.join(staging_bucket, remote_job)
    remote_job_input_path = os.path.join(remote_job_base_path, "input")
    remote_job_output_path = model_utils._generate_remote_job_output_path(
        remote_job_base_path
    )

    detected_framework = None
    if supported_frameworks._is_sklearn(self):
        detected_framework = "sklearn"
    elif supported_frameworks._is_keras(self):
        detected_framework = "tensorflow"
        # TODO(b/295580335): Investigate Tensorflow 2.13 GPU Hanging
        import tensorflow as tf

        accelerator_count = config.accelerator_count if config.accelerator_count else 0
        if (
            version.Version(tf.__version__).base_version >= "2.13.0"
            and accelerator_count > 1
        ):
            raise ValueError(
                f"Currently Tensorflow {tf.__version__} doesn't support multi-gpu training."
            )
    elif supported_frameworks._is_torch(self):
        detected_framework = "torch"
        # TODO(b/296944997): Support remote training on torch<2
        import torch

        if version.Version(torch.__version__).base_version < "2.0.0":
            raise ValueError(
                f"Currently Vertex remote training doesn't support torch {torch.__version__}. "
                "Please use torch>=2.0.0"
            )

    # serialize the estimator
    serializer = any_serializer.AnySerializer()
    serialization_metadata = serializer.serialize(
        to_serialize=self,
        gcs_path=os.path.join(remote_job_input_path, "input_estimator"),
        **serializer_args.get(self, {}),
    )
    requirements += serialization_metadata[
        serializers_base.SERIALIZATION_METADATA_DEPENDENCIES_KEY
    ]
    # serialize args
    for arg_name, arg_value in serialized_args.items():
        if supported_frameworks._is_bigframe(arg_value):
            # Throw error for Python 3.11+ and Bigframes Torch
            if detected_framework == "torch" and sys.version_info[1] >= 11:
                raise ValueError(
                    "Currently Bigframes Torch serializer does not support"
                    "Python 3.11+ since torcharrow is not supported on Python 3.11+."
                )
            serialization_metadata = serializer.serialize(
                to_serialize=arg_value,
                gcs_path=os.path.join(remote_job_input_path, f"{arg_name}"),
                framework=detected_framework,
                **serializer_args.get(arg_value, {}),
            )
        else:
            serialization_metadata = serializer.serialize(
                to_serialize=arg_value,
                gcs_path=os.path.join(remote_job_input_path, f"{arg_name}"),
                **serializer_args.get(arg_value, {}),
            )
        # serializer.get_dependencies() must be run after serializer.serialize()
        requirements += serialization_metadata[
            serializers_base.SERIALIZATION_METADATA_DEPENDENCIES_KEY
        ]

    # execute the method in CustomJob
    # set training configuration
    display_name = config.display_name or remote_job

    # get or generate worker_pool_specs
    # user can specify either worker_pool_specs OR machine_type etc.
    remote_specs._verify_specified_remote_config_values(
        config.worker_pool_specs,
        config.machine_type,
        config.accelerator_type,
        config.accelerator_count,
    )

    if not config.container_uri:
        container_uri = (
            supported_frameworks._get_cpu_container_uri()
            if not enable_cuda
            else supported_frameworks._get_gpu_container_uri(self)
        )
        requirements = _dedupe_requirements(
            vertex_requirements + config.requirements + requirements
        )
    else:
        container_uri = config.container_uri
        requirements = _dedupe_requirements(vertex_requirements + config.requirements)

    requirements = _add_indirect_dependency_versions(requirements)
    command = ["export PIP_ROOT_USER_ACTION=ignore &&"]

    # Combine user custom_commands and serializer custom_commands
    custom_commands += serialization_metadata[
        serializers_base.SERIALIZATION_METADATA_CUSTOM_COMMANDS_KEY
    ]
    custom_commands += config.custom_commands
    custom_commands = list(dict.fromkeys(custom_commands))

    if custom_commands:
        custom_commands = [f"{command} &&" for command in custom_commands]
        command.extend(custom_commands)
    if requirements:
        command.append("pip install --upgrade pip &&")
        requirements = [f"'{requirement}'" for requirement in requirements]
        command.append(f"pip install {' '.join(requirements)} &&")

    pass_through_bool_args_flag_value = ",".join(
        f"{key}={value}" for key, value in pass_through_bool_args.items()
    )
    pass_through_int_args_flag_value = ",".join(
        f"{key}={value}" for key, value in pass_through_int_args.items()
    )
    pass_through_float_args_flag_value = ",".join(
        f"{key}={value}" for key, value in pass_through_float_args.items()
    )
    pass_through_str_args_flag_value = ",".join(
        f"{key}={value}" for key, value in pass_through_str_args.items()
    )

    autolog_command = " --enable_autolog" if autolog else ""

    training_command = (
        "python3 -m "
        "vertexai.preview._workflow.executor.training_script "
        f"--pass_through_int_args={pass_through_int_args_flag_value} "
        f"--pass_through_float_args={pass_through_float_args_flag_value} "
        f"--pass_through_str_args={pass_through_str_args_flag_value} "
        f"--pass_through_bool_args={pass_through_bool_args_flag_value} "
        f"--input_path={remote_job_input_path.replace('gs://', '/gcs/', 1)} "
        f"--output_path={remote_job_output_path.replace('gs://', '/gcs/', 1)} "
        f"--method_name={method_name} "
        + f"--arg_names={','.join(list(serialized_args.keys()))} "
        + f"--enable_cuda={enable_cuda} "
        + f"--enable_distributed={config.enable_distributed} "
        # For distributed training. Use this to infer tf.distribute strategy for Keras training.
        # Keras single worker, multi-gpu needs to be compiled with tf.distribute.MirroredStrategy.
        # Keras multi-worker needs to be compiled with tf.distribute.MultiWorkerMirroredStrategy.
        + f"--accelerator_count={0 if not config.accelerator_count else config.accelerator_count}"
        + autolog_command
    )
    command.append(training_command)
    # Temporary fix for git not installed in pytorch cuda image
    # Remove it once SDK 2.0 is release and don't need to be installed from git
    if container_uri == "pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime":
        command = ["apt-get update && apt-get install -y git &&"] + command

    command = ["sh", "-c", " ".join(command)]

    labels = _set_job_labels(method_name)

    # serialize rewrapper, this is needed to load a model from a CustomJob
    filepath = os.path.join(
        remote_job_output_path,
        model_utils._REWRAPPER_NAME,
    )
    serializer.serialize(rewrapper, filepath, **serializer_args.get(rewrapper, {}))

    # Right before making the job, we save the serialization global metadata
    input_global_metadata_gcs_uri = os.path.join(
        remote_job_input_path, any_serializer.GLOBAL_SERIALIZATION_METADATA
    )
    serializer.save_global_metadata(input_global_metadata_gcs_uri)
    # create & run the CustomJob

    # disable CustomJob logs
    logging.getLogger("google.cloud.aiplatform.jobs").disabled = True
    logging.getLogger("google.cloud.aiplatform.preview.jobs").disabled = True
    cluster_name = (
        vertexai.preview.global_config.cluster.name
        if vertexai.preview.global_config.cluster is not None
        else None
    )
    try:
        job = jobs.CustomJob(
            display_name=display_name,
            project=vertexai.preview.global_config.project,
            location=vertexai.preview.global_config.location,
            worker_pool_specs=_get_worker_pool_specs(config, container_uri, command),
            base_output_dir=remote_job_base_path,
            staging_bucket=remote_job_base_path,
            labels=labels,
            persistent_resource_id=cluster_name,
        )

        job.submit(
            service_account=service_account,
            # TODO(jayceeli) Remove this check when manual logging is supported.
            experiment=metadata._experiment_tracker.experiment if autolog else None,
            experiment_run=metadata._experiment_tracker.experiment_run
            if autolog
            else None,
        )
        job.wait_for_resource_creation()

        _LOGGER.info(f"Remote job created. View the job: {job._dashboard_uri()}")

        _get_remote_logs_until_complete(
            job=job,
            system_logs=config.enable_full_logs,
        )
    except Exception as e:
        raise e
    finally:
        # enable CustomJob logs after remote training job is done
        logging.getLogger("google.cloud.aiplatform.jobs").disabled = False
        logging.getLogger("google.cloud.aiplatform.preview.jobs").disabled = False

    if job.state in jobs._JOB_ERROR_STATES:
        return job

    add_model_to_history_obj = False

    # retrieve the result from gcs to local
    # First, load the global metadata
    output_global_metadata_gcs_uri = os.path.join(
        remote_job_output_path, any_serializer.GLOBAL_SERIALIZATION_METADATA
    )
    serializer.load_global_metadata(output_global_metadata_gcs_uri)
    if method_name in supported_frameworks.REMOTE_TRAINING_STATEFUL_OVERRIDE_LIST:
        estimator = serializer.deserialize(
            os.path.join(remote_job_output_path, model_utils._OUTPUT_ESTIMATOR_DIR),
        )

        if supported_frameworks._is_sklearn(self):
            _update_sklearn_model_inplace(self, estimator)

        elif supported_frameworks._is_keras(self):
            add_model_to_history_obj = True
            _update_keras_model_inplace(self, estimator)

        elif supported_frameworks._is_torch(self):
            _update_torch_model_inplace(self, estimator)

        elif supported_frameworks._is_lightning(self):
            _update_lightning_trainer_inplace(self, estimator)
            # deserialize and update the trained model as well
            trained_model = serializer.deserialize(
                os.path.join(
                    remote_job_output_path, model_utils._OUTPUT_ESTIMATOR_DIR, "model"
                )
            )
            _update_torch_model_inplace(serialized_args["model"], trained_model)
        else:
            # if it's a custom model, update the model object by iterating its
            # attributes. A custom model is any class that has a method
            # decorated by @vertexai.preview.developer.mark.train (and optionally
            # another method decorated by @vertexai.preview.developer.mark.predict).
            _common_update_model_inplace(self, estimator)

    if method_name in supported_frameworks.REMOTE_PREDICTION_OVERRIDE_LIST:
        predictions = serializer.deserialize(
            os.path.join(remote_job_output_path, model_utils._OUTPUT_PREDICTIONS_DIR)
        )
        return predictions

    # Note: "output_data" refers to general output from the executed method, not
    # just a transformed data.
    try:
        # TODO b/296584472: figure out a general mechanism to populate
        # inter-object references.
        if add_model_to_history_obj:
            output_data = serializer.deserialize(
                os.path.join(remote_job_output_path, "output_data"), model=self
            )
        else:
            output_data = serializer.deserialize(
                os.path.join(remote_job_output_path, "output_data")
            )
        return output_data
    except Exception as e:
        _LOGGER.warning(
            f"Fail to deserialize the output due to error {e}, " "returning None."
        )
        return None
