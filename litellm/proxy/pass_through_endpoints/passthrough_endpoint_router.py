from collections.abc import Callable
from typing import TYPE_CHECKING, Final

import litellm
from litellm._logging import verbose_router_logger
from litellm.integrations.vector_store_integrations.vector_store_pre_call_hook import (
    LiteLLM_ManagedVectorStore,
)
from litellm.litellm_core_utils.credential_accessor import CredentialAccessor
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.vertex_ai import VERTEX_CREDENTIALS_TYPES
from litellm.types.passthrough_endpoints.vertex_ai import VertexPassThroughCredentials
from litellm.types.router import LiteLLMParamsTypedDict

if TYPE_CHECKING:
    from litellm.router import Router


def _get_proxy_llm_router() -> "Router | None":
    from litellm.proxy.proxy_server import llm_router

    return llm_router


def _get_str_value(values: dict[str, object] | None, key: str) -> str | None:
    value: Final = values.get(key) if values is not None else None
    return value if isinstance(value, str) else None


class PassthroughEndpointRouter:
    """
    Use this class to Get credentials for pass-through endpoints
    """

    def __init__(
        self,
        llm_router_getter: "Callable[[], Router | None]" = _get_proxy_llm_router,
    ):
        self.llm_router_getter: Final = llm_router_getter
        self.deployment_key_to_vertex_credentials: dict[str, VertexPassThroughCredentials] = {}
        self.default_vertex_config: VertexPassThroughCredentials | None = None

    def get_credentials(
        self,
        custom_llm_provider: str,
        region_name: str | None,
    ) -> str | None:
        deployment_api_key: Final = self._get_deployment_api_key(
            custom_llm_provider=custom_llm_provider,
            region_name=region_name,
        )
        if deployment_api_key is not None:
            return deployment_api_key
        verbose_router_logger.debug(
            "No pass-through deployment credentials found for %s, looking for env variable", custom_llm_provider
        )
        _env_variable_name: Final = self._get_default_env_variable_name_passthrough_endpoint(
            custom_llm_provider=custom_llm_provider,
        )
        return get_secret_str(_env_variable_name)

    def _get_deployment_api_key(
        self,
        custom_llm_provider: str,
        region_name: str | None,
    ) -> str | None:
        llm_router: Final = self.llm_router_getter()
        if llm_router is None:
            return None
        deployments: Final = llm_router.get_model_list() or ()
        return next(
            (
                api_key
                for deployment in deployments
                if (
                    api_key := self._resolve_matching_deployment_api_key(
                        litellm_params=deployment["litellm_params"],
                        custom_llm_provider=custom_llm_provider,
                        region_name=region_name,
                    )
                )
                is not None
            ),
            None,
        )

    def _resolve_matching_deployment_api_key(
        self,
        litellm_params: LiteLLMParamsTypedDict,
        custom_llm_provider: str,
        region_name: str | None,
    ) -> str | None:
        if litellm_params.get("use_in_pass_through") is not True:
            return None
        if self._get_deployment_provider(litellm_params) != custom_llm_provider:
            return None
        credential_name: Final = litellm_params.get("litellm_credential_name")
        credential_values: Final = (
            CredentialAccessor.get_credential_values(credential_name) if credential_name is not None else None
        )
        api_base: Final = _get_str_value(credential_values, "api_base") or litellm_params.get("api_base")
        deployment_region: Final = self._get_region_name_from_api_base(
            custom_llm_provider=custom_llm_provider,
            api_base=api_base,
        )
        if deployment_region != region_name:
            return None
        return _get_str_value(credential_values, "api_key") or litellm_params.get("api_key")

    def _get_deployment_provider(self, litellm_params: LiteLLMParamsTypedDict) -> str | None:
        model: Final = litellm_params.get("model")
        if model is None:
            return None
        try:
            _, provider, _, _ = litellm.get_llm_provider(
                model=model,
                custom_llm_provider=litellm_params.get("custom_llm_provider"),
            )
        except litellm.exceptions.BadRequestError:
            return None
        return provider

    def _get_vertex_env_vars(self) -> VertexPassThroughCredentials:
        """
        Helper to get vertex pass through config from environment variables

        The following environment variables are used:
        - DEFAULT_VERTEXAI_PROJECT (project id)
        - DEFAULT_VERTEXAI_LOCATION (location)
        - DEFAULT_GOOGLE_APPLICATION_CREDENTIALS (path to credentials file)
        """
        return VertexPassThroughCredentials(
            vertex_project=get_secret_str("DEFAULT_VERTEXAI_PROJECT"),
            vertex_location=get_secret_str("DEFAULT_VERTEXAI_LOCATION"),
            vertex_credentials=get_secret_str("DEFAULT_GOOGLE_APPLICATION_CREDENTIALS"),
        )

    def set_default_vertex_config(self, config: dict | None = None):
        """Sets vertex configuration from provided config and/or environment variables

        Args:
            config (Optional[dict]): Configuration dictionary
            Example: {
                "vertex_project": "my-project-123",
                "vertex_location": "us-central1",
                "vertex_credentials": "os.environ/GOOGLE_CREDS"
            }
        """
        # Initialize config dictionary if None
        if config is None:
            self.default_vertex_config = self._get_vertex_env_vars()
            return

        if isinstance(config, dict):
            for key, value in config.items():
                if isinstance(value, str) and value.startswith("os.environ/"):
                    config[key] = get_secret_str(value)

        self.default_vertex_config = VertexPassThroughCredentials(**config)

    def add_vertex_credentials(
        self,
        project_id: str,
        location: str,
        vertex_credentials: VERTEX_CREDENTIALS_TYPES | None,
    ):
        """
        Add the vertex credentials for the given project-id, location
        """

        deployment_key: Final = self._get_deployment_key(
            project_id=project_id,
            location=location,
        )
        if deployment_key is None:
            verbose_router_logger.debug("No deployment key found for project-id, location")
            return
        vertex_pass_through_credentials: Final = VertexPassThroughCredentials(
            vertex_project=project_id,
            vertex_location=location,
            vertex_credentials=vertex_credentials,
        )
        self.deployment_key_to_vertex_credentials[deployment_key] = vertex_pass_through_credentials

    def _get_deployment_key(self, project_id: str | None, location: str | None) -> str | None:
        """
        Get the deployment key for the given project-id, location
        """
        if project_id is None or location is None:
            return None
        return f"{project_id}-{location}"

    def get_vector_store_credentials(self, vector_store_id: str) -> LiteLLM_ManagedVectorStore | None:
        """
        Get the vector store credentials for the given vector store id
        """
        if litellm.vector_store_registry is None:
            return None
        vector_store_to_run: Final[LiteLLM_ManagedVectorStore | None] = (
            litellm.vector_store_registry.get_litellm_managed_vector_store_from_registry(
                vector_store_id=vector_store_id
            )
        )
        return vector_store_to_run

    def get_vertex_credentials(
        self, project_id: str | None, location: str | None
    ) -> VertexPassThroughCredentials | None:
        """
        Get the vertex credentials for the given project-id, location
        """
        deployment_key: Final = self._get_deployment_key(
            project_id=project_id,
            location=location,
        )

        if deployment_key is None:
            return self.default_vertex_config
        if deployment_key in self.deployment_key_to_vertex_credentials:
            return self.deployment_key_to_vertex_credentials[deployment_key]
        else:
            return self.default_vertex_config

    def _get_region_name_from_api_base(
        self,
        custom_llm_provider: str,
        api_base: str | None,
    ) -> str | None:
        """
        Get the region name from the API base.

        Each provider might have a different way of specifying the region in the API base - this is where you can use conditional logic to handle that.
        """
        if custom_llm_provider == "assemblyai":
            if api_base and "eu" in api_base:
                return "eu"
        return None

    @staticmethod
    def _get_default_env_variable_name_passthrough_endpoint(
        custom_llm_provider: str,
    ) -> str:
        return f"{custom_llm_provider.upper()}_API_KEY"
