import json
import re
from typing import Dict, Optional

from litellm._logging import verbose_proxy_logger
from litellm.proxy.vertex_ai_endpoints.vertex_endpoints import (
    VertexPassThroughCredentials,
)


class VertexPassThroughRouter:
    """
    Vertex Pass Through Router for Vertex AI pass-through endpoints


    - if request specifies a project-id, location -> use credentials corresponding to the project-id, location
    - if request does not specify a project-id, location -> use credentials corresponding to the DEFAULT_VERTEXAI_PROJECT, DEFAULT_VERTEXAI_LOCATION
    """

    def __init__(self):
        """
        Initialize the VertexPassThroughRouter
        Stores the vertex credentials for each deployment key
        ```
        {
            "project_id-location": VertexPassThroughCredentials,
            "adroit-crow-us-central1": VertexPassThroughCredentials,
        }
        ```
        """
        self.deployment_key_to_vertex_credentials: Dict[
            str, VertexPassThroughCredentials
        ] = {}
        pass

    def get_vertex_credentials(
        self, project_id: Optional[str], location: Optional[str]
    ) -> VertexPassThroughCredentials:
        """
        Get the vertex credentials for the given project-id, location
        """
        from litellm.proxy.vertex_ai_endpoints.vertex_endpoints import (
            default_vertex_config,
        )

        deployment_key = self._get_deployment_key(
            project_id=project_id,
            location=location,
        )
        if deployment_key is None:
            return default_vertex_config
        if deployment_key in self.deployment_key_to_vertex_credentials:
            return self.deployment_key_to_vertex_credentials[deployment_key]
        else:
            return default_vertex_config

    def add_vertex_credentials(
        self,
        project_id: str,
        location: str,
        vertex_credentials: str,
    ):
        """
        Add the vertex credentials for the given project-id, location
        """
        from litellm.proxy.vertex_ai_endpoints.vertex_endpoints import (
            _set_default_vertex_config,
        )

        deployment_key = self._get_deployment_key(
            project_id=project_id,
            location=location,
        )
        if deployment_key is None:
            verbose_proxy_logger.debug(
                "No deployment key found for project-id, location"
            )
            return
        vertex_pass_through_credentials = VertexPassThroughCredentials(
            vertex_project=project_id,
            vertex_location=location,
            vertex_credentials=vertex_credentials,
        )
        self.deployment_key_to_vertex_credentials[deployment_key] = (
            vertex_pass_through_credentials
        )
        verbose_proxy_logger.debug(
            f"self.deployment_key_to_vertex_credentials: {json.dumps(self.deployment_key_to_vertex_credentials, indent=4, default=str)}"
        )
        _set_default_vertex_config(vertex_pass_through_credentials)

    def _get_deployment_key(
        self, project_id: Optional[str], location: Optional[str]
    ) -> Optional[str]:
        """
        Get the deployment key for the given project-id, location
        """
        if project_id is None or location is None:
            return None
        return f"{project_id}-{location}"

    @staticmethod
    def _get_vertex_project_id_from_url(url: str) -> Optional[str]:
        """
        Get the vertex project id from the url

        `https://${LOCATION}-aiplatform.googleapis.com/v1/projects/${PROJECT_ID}/locations/${LOCATION}/publishers/google/models/${MODEL_ID}:streamGenerateContent`
        """
        match = re.search(r"/projects/([^/]+)", url)
        return match.group(1) if match else None

    @staticmethod
    def _get_vertex_location_from_url(url: str) -> Optional[str]:
        """
        Get the vertex location from the url

        `https://${LOCATION}-aiplatform.googleapis.com/v1/projects/${PROJECT_ID}/locations/${LOCATION}/publishers/google/models/${MODEL_ID}:streamGenerateContent`
        """
        match = re.search(r"/locations/([^/]+)", url)
        return match.group(1) if match else None
