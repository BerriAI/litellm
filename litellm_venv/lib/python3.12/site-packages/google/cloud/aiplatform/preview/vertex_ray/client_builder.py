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
import logging
from typing import Dict
from typing import Optional
from google.cloud import aiplatform
from ray import client_builder
from .render import VertexRayTemplate
from .util import _validation_utils
from .util import _gapic_utils


VERTEX_SDK_VERSION = aiplatform.__version__


class _VertexRayClientContext(client_builder.ClientContext):
    """Custom ClientContext."""

    def __init__(
        self,
        persistent_resource_id: str,
        ray_head_uris: Dict[str, str],
        ray_client_context: client_builder.ClientContext,
    ) -> None:
        dashboard_uri = ray_head_uris.get("RAY_DASHBOARD_URI")
        if dashboard_uri is None:
            raise ValueError(
                "Ray Cluster ",
                persistent_resource_id,
                " failed to start Head node properly.",
            )

        super().__init__(
            dashboard_url=dashboard_uri,
            python_version=ray_client_context.python_version,
            ray_version=ray_client_context.ray_version,
            ray_commit=ray_client_context.ray_commit,
            protocol_version=ray_client_context.protocol_version,
            _num_clients=ray_client_context._num_clients,
            _context_to_restore=ray_client_context._context_to_restore,
        )
        self.persistent_resource_id = persistent_resource_id
        self.vertex_sdk_version = str(VERTEX_SDK_VERSION)
        self.shell_uri = ray_head_uris.get("RAY_HEAD_NODE_INTERACTIVE_SHELL_URI")

    def _repr_html_(self):
        shell_uri_row = None
        if self.shell_uri is not None:
            shell_uri_row = VertexRayTemplate("context_shellurirow.html.j2").render(
                shell_uri=self.shell_uri
            )

        return VertexRayTemplate("context.html.j2").render(
            python_version=self.python_version,
            ray_version=self.ray_version,
            vertex_sdk_version=self.vertex_sdk_version,
            dashboard_url=self.dashboard_url,
            persistent_resource_id=self.persistent_resource_id,
            shell_uri_row=shell_uri_row,
        )


class VertexRayClientBuilder(client_builder.ClientBuilder):
    """Class to initialize a Ray client with vertex on ray capabilities."""

    def __init__(self, address: Optional[str]) -> None:
        address = _validation_utils.maybe_reconstruct_resource_name(address)
        _validation_utils.valid_resource_name(address)

        self.vertex_address = address
        logging.info(
            "[Ray on Vertex AI]: Using cluster resource name to access head address with GAPIC API"
        )

        self.resource_name = address

        self.response = _gapic_utils.get_persistent_resource(self.resource_name)
        address = self.response.resource_runtime.access_uris.get(
            "RAY_HEAD_NODE_INTERNAL_IP"
        )
        if address is None:
            persistent_resource_id = self.resource_name.split("/")[5]
            raise ValueError(
                "[Ray on Vertex AI]: Ray Cluster ",
                persistent_resource_id,
                " Head node is not reachable. Please ensure that a valid VPC network has been specified.",
            )
        # Handling service_account
        service_account = (
            self.response.resource_runtime_spec.service_account_spec.service_account
        )

        if service_account:
            raise ValueError(
                "[Ray on Vertex AI]: Ray Cluster ",
                address,
                " failed to start Head node properly because custom service account isn't supported.",
            )
        logging.debug("[Ray on Vertex AI]: Resolved head node ip: %s", address)
        cluster = _gapic_utils.persistent_resource_to_cluster(
            persistent_resource=self.response
        )
        if cluster is None:
            raise ValueError(
                "[Ray on Vertex AI]: Please delete and recreate the cluster (The cluster is not a Ray cluster or the cluster image is outdated)."
            )
        local_ray_verion = _validation_utils.get_local_ray_version()
        if cluster.ray_version != local_ray_verion:
            if cluster.head_node_type.custom_image is None:
                install_ray_version = _validation_utils.SUPPORTED_RAY_VERSIONS.get(
                    cluster.ray_version
                )
                logging.info(
                    "[Ray on Vertex]: Local runtime has Ray version %s"
                    ", but the requested cluster runtime has %s. Please "
                    "ensure that the Ray versions match for client connectivity. You may "
                    '"pip install --user --force-reinstall ray[default]==%s"'
                    " and restart runtime before cluster connection."
                    % (local_ray_verion, cluster.ray_version, install_ray_version)
                )
            else:
                logging.info(
                    "[Ray on Vertex]: Local runtime has Ray version %s."
                    "Please ensure that the Ray versions match for client connectivity."
                    % local_ray_verion
                )
        super().__init__(address)

    def connect(self) -> _VertexRayClientContext:
        # Can send any other params to ray cluster here
        logging.info("[Ray on Vertex AI]: Connecting...")
        ray_client_context = super().connect()
        ray_head_uris = self.response.resource_runtime.access_uris

        # Valid resource name (reference public doc for public release):
        # "projects/<project_num>/locations/<region>/persistentResources/<pr_id>"
        persistent_resource_id = self.resource_name.split("/")[5]

        return _VertexRayClientContext(
            persistent_resource_id=persistent_resource_id,
            ray_head_uris=ray_head_uris,
            ray_client_context=ray_client_context,
        )
