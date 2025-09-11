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

import abc
from typing import Callable, Dict
from werkzeug import Response


class BasePlugin(abc.ABC):
    """Base plugin for cloud training tools endpoints.

    The plugins support registering http handlers to be used for
    AI Platform training jobs.
    """

    @staticmethod
    @abc.abstractmethod
    def setup() -> None:
        """Run any setup code for the plugin before webserver is launched."""
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def can_initialize() -> bool:
        """Check whether a plugin is able to be initialized.

        Used for checking if correct dependencies are installed, system requirements, etc.

        Returns:
            Bool indicating whether the plugin can be initialized.
        """
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def post_setup_check() -> bool:
        """Check if after initialization, we need to use the plugin.

        Example: Web server only needs to run for main node for training, others
        just need to have 'setup()' run to start the rpc server.

        Returns:
            A boolean indicating whether post setup checks pass.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_routes(self) -> Dict[str, Callable[..., Response]]:
        """Get the mapping from path to handler.

        This is the method in which plugins can assign different routes to
        different handlers.

        Returns:
            A mapping from a route to a handler.
        """
        raise NotImplementedError
