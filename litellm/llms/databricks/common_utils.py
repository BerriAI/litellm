from typing import Literal, Optional, Tuple

from litellm.llms.base_llm.chat.transformation import BaseLLMException


class DatabricksException(BaseLLMException):
    pass


class DatabricksBase:
    def _get_api_base(self, api_base: Optional[str]) -> str:
        if api_base is None:
            try:
                from databricks.sdk import WorkspaceClient

                databricks_client = WorkspaceClient()

                api_base = (
                    api_base or f"{databricks_client.config.host}/serving-endpoints"
                )

                return api_base
            except ImportError:
                raise DatabricksException(
                    status_code=400,
                    message=(
                        "Either set the DATABRICKS_API_BASE and DATABRICKS_API_KEY environment variables, "
                        "or install the databricks-sdk Python library."
                    ),
                )
        return api_base

    def _get_databricks_credentials(
        self, api_key: Optional[str], api_base: Optional[str], headers: Optional[dict]
    ) -> Tuple[str, dict]:
        headers = headers or {"Content-Type": "application/json"}
        try:
            from databricks.sdk import WorkspaceClient

            databricks_client = WorkspaceClient()

            api_base = api_base or f"{databricks_client.config.host}/serving-endpoints"

            if api_key is None:
                databricks_auth_headers: dict[
                    str, str
                ] = databricks_client.config.authenticate()
                headers = {**databricks_auth_headers, **headers}

            return api_base, headers
        except ImportError:
            raise DatabricksException(
                status_code=400,
                message=(
                    "If the Databricks base URL and API key are not set, the databricks-sdk "
                    "Python library must be installed. Please install the databricks-sdk, set "
                    "{LLM_PROVIDER}_API_BASE and {LLM_PROVIDER}_API_KEY environment variables, "
                    "or provide the base URL and API key as arguments."
                ),
            )

    def databricks_validate_environment(
        self,
        api_key: Optional[str],
        api_base: Optional[str],
        endpoint_type: Literal["chat_completions", "embeddings"],
        custom_endpoint: Optional[bool],
        headers: Optional[dict],
    ) -> Tuple[str, dict]:
        if api_key is None and not headers:  # handle empty headers
            if custom_endpoint is True:
                raise DatabricksException(
                    status_code=400,
                    message="Missing API Key - A call is being made to LLM Provider but no key is set either in the environment variables ({LLM_PROVIDER}_API_KEY) or via params",
                )
            else:
                api_base, headers = self._get_databricks_credentials(
                    api_base=api_base, api_key=api_key, headers=headers
                )

        if api_base is None:
            if custom_endpoint:
                raise DatabricksException(
                    status_code=400,
                    message="Missing API Base - A call is being made to LLM Provider but no api base is set either in the environment variables ({LLM_PROVIDER}_API_KEY) or via params",
                )
            else:
                api_base, headers = self._get_databricks_credentials(
                    api_base=api_base, api_key=api_key, headers=headers
                )

        if headers is None:
            headers = {
                "Authorization": "Bearer {}".format(api_key),
                "Content-Type": "application/json",
            }
        else:
            if api_key is not None:
                headers.update({"Authorization": "Bearer {}".format(api_key)})

        if api_key is not None:
            headers["Authorization"] = f"Bearer {api_key}"

        if endpoint_type == "chat_completions" and custom_endpoint is not True:
            api_base = "{}/chat/completions".format(api_base)
        elif endpoint_type == "embeddings" and custom_endpoint is not True:
            api_base = "{}/embeddings".format(api_base)
        return api_base, headers
