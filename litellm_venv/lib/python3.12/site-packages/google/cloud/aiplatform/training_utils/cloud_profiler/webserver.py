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

"""A basic webserver for hosting plugin routes."""

import os

from google.cloud.aiplatform.training_utils.cloud_profiler import wsgi_types
from google.cloud.aiplatform.training_utils.cloud_profiler.plugins import base_plugin
from typing import List
from werkzeug import wrappers, Response


class WebServer:
    """A basic web server for handling requests."""

    def __init__(self, plugins: List[base_plugin.BasePlugin]):
        """Creates a web server to host plugin routes.

        Args:
            plugins (List[base_plugin.BasePlugin]):
                Required. A list of `BasePlugin` objects.

        Raises:
            ValueError:
                When there is an invalid route passed from
                one of the plugins.
        """

        self._plugins = plugins
        self._routes = {}

        # Routes are in form {plugin_name}/{route}
        for plugin in self._plugins:
            for route, handler in plugin.get_routes().items():
                if not route.startswith("/"):
                    raise ValueError(
                        'Routes should start with a "/", '
                        "invalid route for plugin %s, route %s"
                        % (plugin.PLUGIN_NAME, route)
                    )

                app_route = os.path.join("/", plugin.PLUGIN_NAME)

                app_route += route
                self._routes[app_route] = handler

    def dispatch_request(
        self, environ: wsgi_types.Environment, start_response: wsgi_types.StartResponse
    ) -> Response:
        """Handles the routing of requests.

        Args:
            environ (wsgi_types.Environment):
                Required. The WSGI environment.
            start_response (wsgi_types.StartResponse):
                Required. The response callable provided by the WSGI server.

        Returns:
            A response iterable.
        """
        # Check for existince of route
        request = wrappers.Request(environ)

        if request.path in self._routes:
            return self._routes[request.path](environ, start_response)

        response = wrappers.Response("Not Found", status=404)
        return response(environ, start_response)

    def wsgi_app(
        self, environ: wsgi_types.Environment, start_response: wsgi_types.StartResponse
    ) -> Response:
        """Entrypoint for wsgi application.

        Args:
            environ (wsgi_types.Environment):
                Required. The WSGI environment.
            start_response (wsgi_types.StartResponse):
                Required. The response callable provided by the WSGI server.

        Returns:
            A response iterable.
        """
        response = self.dispatch_request(environ, start_response)
        return response

    def __call__(self, environ, start_response):
        """Entrypoint for wsgi application.

        Args:
            environ (wsgi_types.Environment):
                Required. The WSGI environment.
            start_response (wsgi_types.StartResponse):
                Required. The response callable provided by the WSGI server.

        Returns:
            A response iterable.
        """
        return self.wsgi_app(environ, start_response)
