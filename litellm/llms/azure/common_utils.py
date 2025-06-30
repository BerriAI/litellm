import json
import os
from typing import Any, Callable, Dict, Literal, Optional, Union, cast

import httpx
from openai import AsyncAzureOpenAI, AzureOpenAI

import litellm
from litellm._logging import verbose_logger
from litellm.caching.caching import DualCache
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.openai.common_utils import BaseOpenAILLM
from litellm.secret_managers.get_azure_ad_token_provider import (
    get_azure_ad_token_provider,
)
from litellm.secret_managers.main import get_secret_str
from litellm.types.router import GenericLiteLLMParams
from litellm.utils import _add_path_to_api_base

azure_ad_cache = DualCache()


class AzureOpenAIError(BaseLLMException):
    def __init__(
        self,
        status_code,
        message,
        request: Optional[httpx.Request] = None,
        response: Optional[httpx.Response] = None,
        headers: Optional[Union[httpx.Headers, dict]] = None,
        body: Optional[dict] = None,
    ):
        super().__init__(
            status_code=status_code,
            message=message,
            request=request,
            response=response,
            headers=headers,
            body=body,
        )


def process_azure_headers(headers: Union[httpx.Headers, dict]) -> dict:
    openai_headers = {}
    if "x-ratelimit-limit-requests" in headers:
        openai_headers["x-ratelimit-limit-requests"] = headers[
            "x-ratelimit-limit-requests"
        ]
    if "x-ratelimit-remaining-requests" in headers:
        openai_headers["x-ratelimit-remaining-requests"] = headers[
            "x-ratelimit-remaining-requests"
        ]
    if "x-ratelimit-limit-tokens" in headers:
        openai_headers["x-ratelimit-limit-tokens"] = headers["x-ratelimit-limit-tokens"]
    if "x-ratelimit-remaining-tokens" in headers:
        openai_headers["x-ratelimit-remaining-tokens"] = headers[
            "x-ratelimit-remaining-tokens"
        ]
    llm_response_headers = {
        "{}-{}".format("llm_provider", k): v for k, v in headers.items()
    }

    return {**llm_response_headers, **openai_headers}


def get_azure_ad_token_from_entra_id(
    tenant_id: str,
    client_id: str,
    client_secret: str,
    scope: str = "https://cognitiveservices.azure.com/.default",
) -> Callable[[], str]:
    """
    Get Azure AD token provider from `client_id`, `client_secret`, and `tenant_id`

    Args:
        tenant_id: str
        client_id: str
        client_secret: str
        scope: str

    Returns:
        callable that returns a bearer token.
    """
    from azure.identity import ClientSecretCredential, get_bearer_token_provider

    verbose_logger.debug("Getting Azure AD Token from Entra ID")

    if tenant_id.startswith("os.environ/"):
        _tenant_id = get_secret_str(tenant_id)
    else:
        _tenant_id = tenant_id

    if client_id.startswith("os.environ/"):
        _client_id = get_secret_str(client_id)
    else:
        _client_id = client_id

    if client_secret.startswith("os.environ/"):
        _client_secret = get_secret_str(client_secret)
    else:
        _client_secret = client_secret

    verbose_logger.debug(
        "tenant_id %s, client_id %s, client_secret %s",
        _tenant_id,
        _client_id,
        _client_secret,
    )
    if _tenant_id is None or _client_id is None or _client_secret is None:
        raise ValueError("tenant_id, client_id, and client_secret must be provided")
    credential = ClientSecretCredential(_tenant_id, _client_id, _client_secret)

    verbose_logger.debug("credential %s", credential)

    token_provider = get_bearer_token_provider(credential, scope)

    verbose_logger.debug("token_provider %s", token_provider)

    return token_provider


def get_azure_ad_token_from_username_password(
    client_id: str,
    azure_username: str,
    azure_password: str,
    scope: str = "https://cognitiveservices.azure.com/.default",
) -> Callable[[], str]:
    """
    Get Azure AD token provider from `client_id`, `azure_username`, and `azure_password`

    Args:
        client_id: str
        azure_username: str
        azure_password: str
        scope: str

    Returns:
        callable that returns a bearer token.
    """
    from azure.identity import UsernamePasswordCredential, get_bearer_token_provider

    verbose_logger.debug(
        "client_id %s, azure_username %s, azure_password %s",
        client_id,
        azure_username,
        azure_password,
    )
    credential = UsernamePasswordCredential(
        client_id=client_id,
        username=azure_username,
        password=azure_password,
    )

    verbose_logger.debug("credential %s", credential)

    token_provider = get_bearer_token_provider(credential, scope)

    verbose_logger.debug("token_provider %s", token_provider)

    return token_provider


def get_azure_ad_token_from_oidc(
    azure_ad_token: str,
    azure_client_id: Optional[str],
    azure_tenant_id: Optional[str],
    scope: Optional[str] = None,
) -> str:
    """
    Get Azure AD token from OIDC token

    Args:
        azure_ad_token: str
        azure_client_id: Optional[str]
        azure_tenant_id: Optional[str]
        scope: str

    Returns:
        `azure_ad_token_access_token` - str
    """
    if scope is None:
        scope = "https://cognitiveservices.azure.com/.default"
    azure_authority_host = os.getenv(
        "AZURE_AUTHORITY_HOST", "https://login.microsoftonline.com"
    )
    azure_client_id = azure_client_id or os.getenv("AZURE_CLIENT_ID")
    azure_tenant_id = azure_tenant_id or os.getenv("AZURE_TENANT_ID")
    if azure_client_id is None or azure_tenant_id is None:
        raise AzureOpenAIError(
            status_code=422,
            message="AZURE_CLIENT_ID and AZURE_TENANT_ID must be set",
        )

    oidc_token = get_secret_str(azure_ad_token)

    if oidc_token is None:
        raise AzureOpenAIError(
            status_code=401,
            message="OIDC token could not be retrieved from secret manager.",
        )

    azure_ad_token_cache_key = json.dumps(
        {
            "azure_client_id": azure_client_id,
            "azure_tenant_id": azure_tenant_id,
            "azure_authority_host": azure_authority_host,
            "oidc_token": oidc_token,
        }
    )

    azure_ad_token_access_token = azure_ad_cache.get_cache(azure_ad_token_cache_key)
    if azure_ad_token_access_token is not None:
        return azure_ad_token_access_token

    client = litellm.module_level_client

    req_token = client.post(
        f"{azure_authority_host}/{azure_tenant_id}/oauth2/v2.0/token",
        data={
            "client_id": azure_client_id,
            "grant_type": "client_credentials",
            "scope": scope,
            "client_assertion_type": "urn:ietf:params:oauth:client-assertion-type:jwt-bearer",
            "client_assertion": oidc_token,
        },
    )

    if req_token.status_code != 200:
        raise AzureOpenAIError(
            status_code=req_token.status_code,
            message=req_token.text,
        )

    azure_ad_token_json = req_token.json()
    azure_ad_token_access_token = azure_ad_token_json.get("access_token", None)
    azure_ad_token_expires_in = azure_ad_token_json.get("expires_in", None)

    if azure_ad_token_access_token is None:
        raise AzureOpenAIError(
            status_code=422, message="Azure AD Token access_token not returned"
        )

    if azure_ad_token_expires_in is None:
        raise AzureOpenAIError(
            status_code=422, message="Azure AD Token expires_in not returned"
        )

    azure_ad_cache.set_cache(
        key=azure_ad_token_cache_key,
        value=azure_ad_token_access_token,
        ttl=azure_ad_token_expires_in,
    )

    return azure_ad_token_access_token


def select_azure_base_url_or_endpoint(azure_client_params: dict):
    azure_endpoint = azure_client_params.get("azure_endpoint", None)
    if azure_endpoint is not None:
        # see : https://github.com/openai/openai-python/blob/3d61ed42aba652b547029095a7eb269ad4e1e957/src/openai/lib/azure.py#L192
        if "/openai/deployments" in azure_endpoint:
            # this is base_url, not an azure_endpoint
            azure_client_params["base_url"] = azure_endpoint
            azure_client_params.pop("azure_endpoint")

    return azure_client_params


def get_azure_ad_token(
    litellm_params: GenericLiteLLMParams,
) -> Optional[str]:
    """
    Get Azure AD token from various authentication methods.

    This function tries different methods to obtain an Azure AD token:
    1. From an existing token provider
    2. From Entra ID using tenant_id, client_id, and client_secret
    3. From username and password
    4. From OIDC token
    5. From a service principal with secret workflow

    Args:
        litellm_params: Dictionary containing authentication parameters
            - azure_ad_token_provider: Optional callable that returns a token
            - azure_ad_token: Optional existing token
            - tenant_id: Optional Azure tenant ID
            - client_id: Optional Azure client ID
            - client_secret: Optional Azure client secret
            - azure_username: Optional Azure username
            - azure_password: Optional Azure password

    Returns:
        Azure AD token as string if successful, None otherwise
    """
    # Extract parameters
    azure_ad_token_provider = litellm_params.get("azure_ad_token_provider")
    azure_ad_token = litellm_params.get("azure_ad_token", None) or get_secret_str(
        "AZURE_AD_TOKEN"
    )
    tenant_id = litellm_params.get("tenant_id", os.getenv("AZURE_TENANT_ID"))
    client_id = litellm_params.get("client_id", os.getenv("AZURE_CLIENT_ID"))
    client_secret = litellm_params.get(
        "client_secret", os.getenv("AZURE_CLIENT_SECRET")
    )
    azure_username = litellm_params.get("azure_username", os.getenv("AZURE_USERNAME"))
    azure_password = litellm_params.get("azure_password", os.getenv("AZURE_PASSWORD"))
    scope = litellm_params.get(
        "azure_scope",
        os.getenv("AZURE_SCOPE", "https://cognitiveservices.azure.com/.default"),
    )
    if scope is None:
        scope = "https://cognitiveservices.azure.com/.default"

    # Try to get token provider from Entra ID
    if azure_ad_token_provider is None and tenant_id and client_id and client_secret:
        verbose_logger.debug(
            "Using Azure AD Token Provider from Entra ID for Azure Auth"
        )
        azure_ad_token_provider = get_azure_ad_token_from_entra_id(
            tenant_id=tenant_id,
            client_id=client_id,
            client_secret=client_secret,
            scope=scope,
        )

    # Try to get token provider from username and password
    if (
        azure_ad_token_provider is None
        and azure_username
        and azure_password
        and client_id
    ):
        verbose_logger.debug("Using Azure Username and Password for Azure Auth")
        azure_ad_token_provider = get_azure_ad_token_from_username_password(
            azure_username=azure_username,
            azure_password=azure_password,
            client_id=client_id,
            scope=scope,
        )

    # Try to get token from OIDC
    if (
        client_id
        and tenant_id
        and azure_ad_token
        and azure_ad_token.startswith("oidc/")
    ):
        verbose_logger.debug("Using Azure OIDC Token for Azure Auth")
        azure_ad_token = get_azure_ad_token_from_oidc(
            azure_ad_token=azure_ad_token,
            azure_client_id=client_id,
            azure_tenant_id=tenant_id,
            scope=scope,
        )
    # Try to get token provider from service principal
    elif (
        azure_ad_token_provider is None
        and litellm.enable_azure_ad_token_refresh is True
    ):
        verbose_logger.debug(
            "Using Azure AD token provider based on Service Principal with Secret workflow for Azure Auth"
        )
        try:
            azure_ad_token_provider = get_azure_ad_token_provider(azure_scope=scope)
        except ValueError:
            verbose_logger.debug("Azure AD Token Provider could not be used.")

    # Execute the token provider to get the token if available
    if azure_ad_token_provider and callable(azure_ad_token_provider):
        try:
            token = azure_ad_token_provider()
            if not isinstance(token, str):
                verbose_logger.error(
                    f"Azure AD token provider returned non-string value: {type(token)}"
                )
                raise TypeError(f"Azure AD token must be a string, got {type(token)}")
            else:
                azure_ad_token = token
        except TypeError:
            # Re-raise TypeError directly
            raise
        except Exception as e:
            verbose_logger.error(f"Error calling Azure AD token provider: {str(e)}")
            raise RuntimeError(f"Failed to get Azure AD token: {str(e)}") from e

    return azure_ad_token


class BaseAzureLLM(BaseOpenAILLM):
    def get_azure_openai_client(
        self,
        api_key: Optional[str],
        api_base: Optional[str],
        api_version: Optional[str] = None,
        client: Optional[Union[AzureOpenAI, AsyncAzureOpenAI]] = None,
        litellm_params: Optional[dict] = None,
        _is_async: bool = False,
        model: Optional[str] = None,
    ) -> Optional[Union[AzureOpenAI, AsyncAzureOpenAI]]:
        openai_client: Optional[Union[AzureOpenAI, AsyncAzureOpenAI]] = None
        client_initialization_params: dict = locals()
        client_initialization_params["is_async"] = _is_async
        if client is None:
            cached_client = self.get_cached_openai_client(
                client_initialization_params=client_initialization_params,
                client_type="azure",
            )
            if cached_client:
                if isinstance(cached_client, AzureOpenAI) or isinstance(
                    cached_client, AsyncAzureOpenAI
                ):
                    return cached_client

            azure_client_params = self.initialize_azure_sdk_client(
                litellm_params=litellm_params or {},
                api_key=api_key,
                api_base=api_base,
                model_name=model,
                api_version=api_version,
                is_async=_is_async,
            )
            if _is_async is True:
                openai_client = AsyncAzureOpenAI(**azure_client_params)
            else:
                openai_client = AzureOpenAI(**azure_client_params)  # type: ignore
        else:
            openai_client = client
            if api_version is not None and isinstance(
                openai_client._custom_query, dict
            ):
                # set api_version to version passed by user
                openai_client._custom_query.setdefault("api-version", api_version)

        # save client in-memory cache
        self.set_cached_openai_client(
            openai_client=openai_client,
            client_initialization_params=client_initialization_params,
            client_type="azure",
        )
        return openai_client

    def initialize_azure_sdk_client(
        self,
        litellm_params: dict,
        api_key: Optional[str],
        api_base: Optional[str],
        model_name: Optional[str],
        api_version: Optional[str],
        is_async: bool,
    ) -> dict:
        azure_ad_token_provider = litellm_params.get("azure_ad_token_provider")
        # If we have api_key, then we have higher priority
        azure_ad_token = litellm_params.get("azure_ad_token")
        tenant_id = litellm_params.get("tenant_id", os.getenv("AZURE_TENANT_ID"))
        client_id = litellm_params.get("client_id", os.getenv("AZURE_CLIENT_ID"))
        client_secret = litellm_params.get(
            "client_secret", os.getenv("AZURE_CLIENT_SECRET")
        )
        azure_username = litellm_params.get(
            "azure_username", os.getenv("AZURE_USERNAME")
        )
        azure_password = litellm_params.get(
            "azure_password", os.getenv("AZURE_PASSWORD")
        )
        scope = litellm_params.get(
            "azure_scope",
            os.getenv("AZURE_SCOPE", "https://cognitiveservices.azure.com/.default"),
        )
        if scope is None:
            scope = "https://cognitiveservices.azure.com/.default"
        max_retries = litellm_params.get("max_retries")
        timeout = litellm_params.get("timeout")
        if (
            not api_key
            and azure_ad_token_provider is None
            and tenant_id
            and client_id
            and client_secret
        ):
            verbose_logger.debug(
                "Using Azure AD Token Provider from Entra ID for Azure Auth"
            )
            azure_ad_token_provider = get_azure_ad_token_from_entra_id(
                tenant_id=tenant_id,
                client_id=client_id,
                client_secret=client_secret,
                scope=scope,
            )
        if (
            azure_ad_token_provider is None
            and azure_username
            and azure_password
            and client_id
        ):
            verbose_logger.debug("Using Azure Username and Password for Azure Auth")
            azure_ad_token_provider = get_azure_ad_token_from_username_password(
                azure_username=azure_username,
                azure_password=azure_password,
                client_id=client_id,
                scope=scope,
            )

        if azure_ad_token is not None and azure_ad_token.startswith("oidc/"):
            verbose_logger.debug("Using Azure OIDC Token for Azure Auth")
            azure_ad_token = get_azure_ad_token_from_oidc(
                azure_ad_token=azure_ad_token,
                azure_client_id=client_id,
                azure_tenant_id=tenant_id,
                scope=scope,
            )
        elif (
            not api_key
            and azure_ad_token_provider is None
            and litellm.enable_azure_ad_token_refresh is True
        ):
            verbose_logger.debug(
                "Using Azure AD token provider based on Service Principal with Secret workflow for Azure Auth"
            )
            try:
                azure_ad_token_provider = get_azure_ad_token_provider(azure_scope=scope)
            except ValueError:
                verbose_logger.debug("Azure AD Token Provider could not be used.")
        if api_version is None:
            api_version = os.getenv(
                "AZURE_API_VERSION", litellm.AZURE_DEFAULT_API_VERSION
            )

        _api_key = api_key
        if _api_key is not None and isinstance(_api_key, str):
            # only show first 5 chars of api_key
            _api_key = _api_key[:8] + "*" * 15
        verbose_logger.debug(
            f"Initializing Azure OpenAI Client for {model_name}, Api Base: {str(api_base)}, Api Key:{_api_key}"
        )
        azure_client_params = {
            "api_key": api_key,
            "azure_endpoint": api_base,
            "api_version": api_version,
            "azure_ad_token": azure_ad_token,
            "azure_ad_token_provider": azure_ad_token_provider,
        }
        # init http client + SSL Verification settings
        if is_async is True:
            azure_client_params["http_client"] = self._get_async_http_client()
        else:
            azure_client_params["http_client"] = self._get_sync_http_client()

        if max_retries is not None:
            azure_client_params["max_retries"] = max_retries
        if timeout is not None:
            azure_client_params["timeout"] = timeout

        if azure_ad_token_provider is not None:
            azure_client_params["azure_ad_token_provider"] = azure_ad_token_provider
        # this decides if we should set azure_endpoint or base_url on Azure OpenAI Client
        # required to support GPT-4 vision enhancements, since base_url needs to be set on Azure OpenAI Client

        azure_client_params = select_azure_base_url_or_endpoint(
            azure_client_params=azure_client_params
        )

        return azure_client_params

    def _init_azure_client_for_cloudflare_ai_gateway(
        self,
        api_base: str,
        model: str,
        api_version: str,
        max_retries: int,
        timeout: Union[float, httpx.Timeout],
        litellm_params: dict,
        api_key: Optional[str],
        azure_ad_token: Optional[str],
        azure_ad_token_provider: Optional[Callable[[], str]],
        acompletion: bool,
        client: Optional[Union[AzureOpenAI, AsyncAzureOpenAI]] = None,
    ) -> Union[AzureOpenAI, AsyncAzureOpenAI]:
        ## build base url - assume api base includes resource name
        tenant_id = litellm_params.get("tenant_id", os.getenv("AZURE_TENANT_ID"))
        client_id = litellm_params.get("client_id", os.getenv("AZURE_CLIENT_ID"))
        scope = litellm_params.get(
            "azure_scope",
            os.getenv("AZURE_SCOPE", "https://cognitiveservices.azure.com/.default"),
        )
        if client is None:
            if not api_base.endswith("/"):
                api_base += "/"
            api_base += f"{model}"

            azure_client_params: Dict[str, Any] = {
                "api_version": api_version,
                "base_url": f"{api_base}",
                "http_client": litellm.client_session,
                "max_retries": max_retries,
                "timeout": timeout,
            }
            if api_key is not None:
                azure_client_params["api_key"] = api_key
            elif azure_ad_token is not None:
                if azure_ad_token.startswith("oidc/"):
                    azure_ad_token = get_azure_ad_token_from_oidc(
                        azure_ad_token=azure_ad_token,
                        azure_client_id=client_id,
                        azure_tenant_id=tenant_id,
                        scope=scope,
                    )

                azure_client_params["azure_ad_token"] = azure_ad_token
            if azure_ad_token_provider is not None:
                azure_client_params["azure_ad_token_provider"] = azure_ad_token_provider

            if acompletion is True:
                client = AsyncAzureOpenAI(**azure_client_params)  # type: ignore
            else:
                client = AzureOpenAI(**azure_client_params)  # type: ignore
        return client
    
    @staticmethod
    def _base_validate_azure_environment(
        headers: dict,  litellm_params: Optional[GenericLiteLLMParams]
    ) -> dict:
        litellm_params = litellm_params or GenericLiteLLMParams()
        api_key = (
            litellm_params.api_key
            or litellm.api_key
            or litellm.azure_key
            or get_secret_str("AZURE_OPENAI_API_KEY")
            or get_secret_str("AZURE_API_KEY")
        )

        if api_key:
            headers["api-key"] = api_key
            return headers

        ### Fallback to Azure AD token-based authentication if no API key is available
        ### Retrieves Azure AD token and adds it to the Authorization header
        azure_ad_token = get_azure_ad_token(litellm_params)
        if azure_ad_token:
            headers["Authorization"] = f"Bearer {azure_ad_token}"

        return headers
    
    @staticmethod
    def _get_base_azure_url(
        api_base: Optional[str],
        litellm_params: Optional[Union[GenericLiteLLMParams, Dict[str, Any]]],
        route: Literal["/openai/responses", "/openai/vector_stores"]
    ) -> str:
        api_base = api_base or litellm.api_base or get_secret_str("AZURE_API_BASE")
        if api_base is None:
            raise ValueError(
                f"api_base is required for Azure AI Studio. Please set the api_base parameter. Passed `api_base={api_base}`"
            )
        original_url = httpx.URL(api_base)

        # Extract api_version or use default
        litellm_params = litellm_params or {}
        api_version = cast(Optional[str], litellm_params.get("api_version"))

        # Create a new dictionary with existing params
        query_params = dict(original_url.params)

        # Add api_version if needed
        if "api-version" not in query_params and api_version:
            query_params["api-version"] = api_version
        
        # Add the path to the base URL
        if route not in api_base:
            new_url = _add_path_to_api_base(
                api_base=api_base, ending_path=route
            )
        else:
            new_url = api_base
        
        if BaseAzureLLM._is_azure_v1_api_version(api_version):
            # ensure the request go to /openai/v1 and not just /openai
            if "/openai/v1" not in new_url:
                parsed_url = httpx.URL(new_url)
                new_url = str(parsed_url.copy_with(path=parsed_url.path.replace("/openai", "/openai/v1")))


        # Use the new query_params dictionary
        final_url = httpx.URL(new_url).copy_with(params=query_params)

        return str(final_url)
    
    @staticmethod
    def _is_azure_v1_api_version(api_version: Optional[str]) -> bool:
        if api_version is None:
            return False
        return api_version == "preview" or api_version == "latest"
