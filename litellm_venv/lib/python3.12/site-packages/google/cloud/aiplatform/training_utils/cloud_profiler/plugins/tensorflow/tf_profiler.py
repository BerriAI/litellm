# -*- coding: utf-8 -*-

# Copyright 2020 Google LLC
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

"""A plugin to handle remote tensoflow profiler sessions for Vertex AI."""

from google.cloud.aiplatform.training_utils.cloud_profiler import (
    cloud_profiler_utils,
)

try:
    import tensorflow as tf
    from tensorboard_plugin_profile.profile_plugin import (
        ProfilePlugin,
    )
except ImportError as err:
    raise ImportError(cloud_profiler_utils.import_error_msg) from err

import argparse
from collections import namedtuple
import importlib.util
import json
import logging
from typing import Callable, Dict, Optional
from urllib import parse

import tensorboard.plugins.base_plugin as tensorboard_base_plugin
from werkzeug import Response

from google.cloud.aiplatform.tensorboard.plugins.tf_profiler import (
    profile_uploader,
)
from google.cloud.aiplatform.training_utils import environment_variables
from google.cloud.aiplatform.training_utils.cloud_profiler import wsgi_types
from google.cloud.aiplatform.training_utils.cloud_profiler.plugins import (
    base_plugin,
)
from google.cloud.aiplatform.training_utils.cloud_profiler.plugins.tensorflow import (
    tensorboard_api,
)


# TF verison information.
Version = namedtuple("Version", ["major", "minor", "patch"])

logger = logging.Logger("tf-profiler")

_BASE_TB_ENV_WARNING = (
    "To set this environment variable, run your training with the 'tensorboard' "
    "option. For more information on how to run with training with tensorboard, visit "
    "https://cloud.google.com/vertex-ai/docs/experiments/tensorboard-training"
)


def _get_tf_versioning() -> Optional[Version]:
    """Convert version string to a Version namedtuple for ease of parsing.

    Returns:
        A version object if finding the version was successful, None otherwise.
    """
    version = tf.__version__

    versioning = version.split(".")
    if len(versioning) != 3:
        return
    return Version(int(versioning[0]), int(versioning[1]), versioning[2])


def _is_compatible_version(version: Version) -> bool:
    """Check if version is compatible with tf profiling.

    Profiling plugin is available to be used for version >= 2.4.0.
    While the profiler is available in 2.2.0 >=, some additional dependencies
    that are included in 2.4.0 >= are also needed for the tensorboard-plugin-profile.

    Profiler:
        https://www.tensorflow.org/guide/profiler
    Required commit for tensorboard-plugin-profile:
        https://github.com/tensorflow/tensorflow/commit/8b9c207242db515daef033e74d69ea5d8e023dc6

    Args:
        version (Version):
            Required. `Verison` of tensorflow.

    Returns:
        Bool indicating wheter version is compatible with profiler.
    """
    return version.major >= 2 and version.minor >= 4


def _check_tf() -> bool:
    """Check whether all the tensorflow prereqs are met.

    Returns:
        True if all requirements met, False otherwise.
    """
    # Check tf is installed
    if importlib.util.find_spec("tensorflow") is None:
        logger.warning("Tensorflow not installed, cannot initialize profiling plugin")
        return False

    # Check tensorflow version
    version = _get_tf_versioning()
    if version is None:
        logger.warning(
            "Could not find major, minor, and patch versions of tensorflow. Version found: %s",
            version,
        )
        return False

    # Check compatibility, introduced in tensorflow >= 2.2.0
    if not _is_compatible_version(version):
        logger.warning(
            "Version %s is incompatible with tf profiler."
            "To use the profiler, choose a version >= 2.2.0",
            "%s.%s.%s" % (version.major, version.minor, version.patch),
        )
        return False

    # Check for the tf profiler plugin
    if importlib.util.find_spec("tensorboard_plugin_profile") is None:
        logger.warning(
            "Could not import tensorboard_plugin_profile, will not run tf profiling service"
        )
        return False

    return True


def _create_profiling_context() -> tensorboard_base_plugin.TBContext:
    """Creates the base context needed for TB Profiler.

    Returns:
        An initialized `TBContext`.
    """

    context_flags = argparse.Namespace(master_tpu_unsecure_channel=None)

    context = tensorboard_base_plugin.TBContext(
        logdir=environment_variables.tensorboard_log_dir,
        multiplexer=None,
        flags=context_flags,
    )

    return context


def _host_to_grpc(hostname: str) -> str:
    """Format a hostname to a grpc address.

    Args:
        hostname (str):
            Required. Address in form: `{hostname}:{port}`

    Returns:
        Address in form of: 'grpc://{hostname}:{port}'
    """
    return (
        "grpc://"
        + "".join(hostname.split(":")[:-1])
        + ":"
        + environment_variables.tf_profiler_port
    )


def _get_hostnames() -> Optional[str]:
    """Get the hostnames for all servers running.

    Returns:
        A host formatted by `_host_to_grpc` if obtaining the cluster spec
        is successful, None otherwise.
    """
    cluster_spec = environment_variables.cluster_spec
    if cluster_spec is None:
        return

    cluster = cluster_spec.get("cluster", "")
    if not cluster:
        return

    hostnames = []
    for value in cluster.values():
        hostnames.extend(value)

    return ",".join([_host_to_grpc(x) for x in hostnames])


def _update_environ(environ: wsgi_types.Environment) -> bool:
    """Add parameters to the query that are retrieved from training side.

    Args:
        environ (wsgi_types.Environment):
            Required. The WSGI Environment.

    Returns:
        Whether the environment was successfully updated.
    """
    hosts = _get_hostnames()

    if hosts is None:
        return False

    query_dict = {}
    query_dict["service_addr"] = hosts

    # Update service address and worker list
    # Use parse_qsl and then convert list to dictionary so we can update
    # attributes
    prev_query_string = dict(parse.parse_qsl(environ["QUERY_STRING"]))
    prev_query_string.update(query_dict)

    environ["QUERY_STRING"] = parse.urlencode(prev_query_string)

    return True


def warn_tensorboard_env_var(var_name: str):
    """Warns if a tensorboard related environment variable is missing.

    Args:
        var_name (str):
            Required. The name of the missing environment variable.
    """
    logging.warning(
        "Environment variable `%s` must be set. %s", var_name, _BASE_TB_ENV_WARNING
    )


def _check_env_vars() -> bool:
    """Determine whether the correct environment variables are set.

    Returns:
        bool indicating all necessary variables are set.
    """
    # The below are tensorboard specific environment variables.
    if environment_variables.tf_profiler_port is None:
        warn_tensorboard_env_var("AIP_TF_PROFILER_PORT")
        return False

    if environment_variables.tensorboard_log_dir is None:
        warn_tensorboard_env_var("AIP_TENSORBOARD_LOG_DIR")
        return False

    if environment_variables.tensorboard_api_uri is None:
        warn_tensorboard_env_var("AIP_TENSORBOARD_API_URI")
        return False

    if environment_variables.tensorboard_resource_name is None:
        warn_tensorboard_env_var("AIP_TENSORBOARD_RESOURCE_NAME")
        return False

    # These environment variables are not tensorboard related, they are
    # variables set for any Vertex training run.
    cluster_spec = environment_variables.cluster_spec
    if cluster_spec is None:
        logger.warning("Environment variable `CLUSTER_SPEC` is not set.")
        return False

    if environment_variables.cloud_ml_job_id is None:
        logger.warning("Environment variable `CLOUD_ML_JOB_ID` is not set")
        return False

    return True


class TFProfiler(base_plugin.BasePlugin):
    """Handler for Tensorflow Profiling."""

    PLUGIN_NAME = "profile"

    def __init__(self):
        """Build a TFProfiler object."""
        context = _create_profiling_context()
        self._profile_request_sender: profile_uploader.ProfileRequestSender = (
            tensorboard_api.create_profile_request_sender()
        )
        self._profile_plugin: ProfilePlugin = ProfilePlugin(context)

    def get_routes(
        self,
    ) -> Dict[str, Callable[[Dict[str, str], Callable[..., None]], Response]]:
        """List of routes to serve.

        Returns:
            A callable that takes an werkzeug env and start response and returns a response.
        """
        return {"/capture_profile": self.capture_profile_wrapper}

    # Define routes below
    def capture_profile_wrapper(
        self, environ: wsgi_types.Environment, start_response: wsgi_types.StartResponse
    ) -> Response:
        """Take a request from tensorboard.gcp and run the profiling for the available servers.

        Args:
            environ (wsgi_types.Environment):
                Required. The WSGI environment.
            start_response (wsgi_types.StartResponse):
                Required. The response callable provided by the WSGI server.

        Returns:
            A response iterable.
        """
        # The service address (localhost) and worker list are populated locally
        if not _update_environ(environ):
            err = {"error": "Could not parse the environ: %s"}
            return Response(
                json.dumps(err), content_type="application/json", status=500
            )

        response = self._profile_plugin.capture_route(environ, start_response)

        self._profile_request_sender.send_request("")

        return response

    # End routes

    @staticmethod
    def setup() -> None:
        """Sets up the plugin."""
        tf.profiler.experimental.server.start(
            int(environment_variables.tf_profiler_port)
        )

    @staticmethod
    def post_setup_check() -> bool:
        """Only chief and task 0 should run the webserver."""
        cluster_spec = environment_variables.cluster_spec
        task_type = cluster_spec.get("task", {}).get("type", "")
        task_index = cluster_spec.get("task", {}).get("index", -1)

        return task_type in {"workerpool0", "chief"} and task_index == 0

    @staticmethod
    def can_initialize() -> bool:
        """Check that we can use the TF Profiler plugin.

        This function checks a number of dependencies for the plugin to ensure we have the
        right packages installed, the necessary versions, and the correct environment variables set.

        Returns:
            True if can initialize, False otherwise.
        """

        return _check_env_vars() and _check_tf()
