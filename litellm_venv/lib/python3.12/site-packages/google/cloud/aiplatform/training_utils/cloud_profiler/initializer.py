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

import logging
import threading
from typing import Optional, Type

from google.cloud.aiplatform.training_utils.cloud_profiler import cloud_profiler_utils

try:
    from werkzeug import serving
except ImportError as err:
    raise ImportError(cloud_profiler_utils.import_error_msg) from err


from google.cloud.aiplatform.training_utils import environment_variables
from google.cloud.aiplatform.training_utils.cloud_profiler import webserver
from google.cloud.aiplatform.training_utils.cloud_profiler.plugins import base_plugin
from google.cloud.aiplatform.training_utils.cloud_profiler.plugins.tensorflow import (
    tf_profiler,
)


# Mapping of available plugins to use
_AVAILABLE_PLUGINS = {"tensorflow": tf_profiler.TFProfiler}


class MissingEnvironmentVariableException(Exception):
    pass


def _build_plugin(
    plugin: Type[base_plugin.BasePlugin],
) -> Optional[base_plugin.BasePlugin]:
    """Builds the plugin given the object.

    Args:
        plugin (Type[base_plugin]):
            Required. An uninitialized plugin class.

    Returns:
        An initialized plugin, or None if plugin cannot be
        initialized.
    """
    if not plugin.can_initialize():
        logging.warning("Cannot initialize the plugin")
        return

    plugin.setup()

    if not plugin.post_setup_check():
        return

    return plugin()


def _run_app_thread(server: webserver.WebServer, port: int):
    """Run the webserver in a separate thread.

    Args:
        server (webserver.WebServer):
            Required. A webserver to accept requests.
        port (int):
            Required. The port to run the webserver on.
    """
    daemon = threading.Thread(
        name="profile_server",
        target=serving.run_simple,
        args=(
            "0.0.0.0",
            port,
            server,
        ),
    )
    daemon.setDaemon(True)
    daemon.start()


def initialize(plugin: str = "tensorflow"):
    """Initializes the profiling SDK.

    Args:
        plugin (str):
            Required. Name of the plugin to initialize.
            Current options are ["tensorflow"]

    Raises:
        ValueError:
            The plugin does not exist.
        MissingEnvironmentVariableException:
            An environment variable that is needed is not set.
    """
    plugin_obj = _AVAILABLE_PLUGINS.get(plugin)

    if not plugin_obj:
        raise ValueError(
            "Plugin {} not available, must choose from {}".format(
                plugin, _AVAILABLE_PLUGINS.keys()
            )
        )

    prof_plugin = _build_plugin(plugin_obj)

    if prof_plugin is None:
        return

    server = webserver.WebServer([prof_plugin])

    if not environment_variables.http_handler_port:
        raise MissingEnvironmentVariableException(
            "'AIP_HTTP_HANDLER_PORT' must be set."
        )

    port = int(environment_variables.http_handler_port)

    _run_app_thread(server, port)
