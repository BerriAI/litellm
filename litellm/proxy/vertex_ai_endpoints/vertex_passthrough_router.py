# import json
# import re
# from typing import Dict, Optional

# from litellm._logging import verbose_proxy_logger
# from litellm.proxy.vertex_ai_endpoints.vertex_endpoints import (
#     VertexPassThroughCredentials,
# )
# from litellm.types.llms.vertex_ai import VERTEX_CREDENTIALS_TYPES


# class VertexPassThroughRouter:
#     """
#     Vertex Pass Through Router for Vertex AI pass-through endpoints


#     - if request specifies a project-id, location -> use credentials corresponding to the project-id, location
#     - if request does not specify a project-id, location -> use credentials corresponding to the DEFAULT_VERTEXAI_PROJECT, DEFAULT_VERTEXAI_LOCATION
#     """

#     def __init__(self):
#         """
#         Initialize the VertexPassThroughRouter
#         Stores the vertex credentials for each deployment key
#         ```
#         {
#             "project_id-location": VertexPassThroughCredentials,
#             "adroit-crow-us-central1": VertexPassThroughCredentials,
#         }
#         ```
#         """
#         self.deployment_key_to_vertex_credentials: Dict[
#             str, VertexPassThroughCredentials
#         ] = {}
#         pass

#     def get_vertex_credentials(
#         self, project_id: Optional[str], location: Optional[str]
#     ) -> Optional[VertexPassThroughCredentials]:
#         """
#         Get the vertex credentials for the given project-id, location
#         """
#         from litellm.proxy.vertex_ai_endpoints.vertex_endpoints import (
#             default_vertex_config,
#         )

#         deployment_key = self._get_deployment_key(
#             project_id=project_id,
#             location=location,
#         )
#         if deployment_key is None:
#             return default_vertex_config
#         if deployment_key in self.deployment_key_to_vertex_credentials:
#             return self.deployment_key_to_vertex_credentials[deployment_key]
#         else:
#             return default_vertex_config


#     @staticmethod
#     def _get_vertex_project_id_from_url(url: str) -> Optional[str]:
#         """
#         Get the vertex project id from the url

#         `https://${LOCATION}-aiplatform.googleapis.com/v1/projects/${PROJECT_ID}/locations/${LOCATION}/publishers/google/models/${MODEL_ID}:streamGenerateContent`
#         """
#         match = re.search(r"/projects/([^/]+)", url)
#         return match.group(1) if match else None

#     @staticmethod
#     def _get_vertex_location_from_url(url: str) -> Optional[str]:
#         """
#         Get the vertex location from the url

#         `https://${LOCATION}-aiplatform.googleapis.com/v1/projects/${PROJECT_ID}/locations/${LOCATION}/publishers/google/models/${MODEL_ID}:streamGenerateContent`
#         """
#         match = re.search(r"/locations/([^/]+)", url)
#         return match.group(1) if match else None
