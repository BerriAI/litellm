import asyncio
import os
import traceback
from typing import TYPE_CHECKING, Any, Callable, Optional

import httpx
import openai

import litellm
from litellm._logging import verbose_router_logger
from litellm.llms.AzureOpenAI.azure import get_azure_ad_token_from_oidc
from litellm.secret_managers.get_azure_ad_token_provider import (
    get_azure_ad_token_provider,
)
from litellm.utils import calculate_max_parallel_requests

if TYPE_CHECKING:
    from litellm.router import Router as _Router

    LitellmRouter = _Router
else:
    LitellmRouter = Any


def should_initialize_sync_client(
    litellm_router_instance: LitellmRouter,
) -> bool:
    """
    Returns if Sync OpenAI, Azure Clients should be initialized.

    Do not init sync clients when router.router_general_settings.async_only_mode is True

    """
    if litellm_router_instance is None:
        return False

    if litellm_router_instance.router_general_settings is not None:
        if (
            hasattr(litellm_router_instance, "router_general_settings")
            and hasattr(
                litellm_router_instance.router_general_settings, "async_only_mode"
            )
            and litellm_router_instance.router_general_settings.async_only_mode is True
        ):
            return False

    return True


def set_client(litellm_router_instance: LitellmRouter, model: dict):
    """
    - Initializes Azure/OpenAI clients. Stores them in cache, b/c of this - https://github.com/BerriAI/litellm/issues/1278
    - Initializes Semaphore for client w/ rpm. Stores them in cache. b/c of this - https://github.com/BerriAI/litellm/issues/2994
    """
    client_ttl = litellm_router_instance.client_ttl
    litellm_params = model.get("litellm_params", {})
    model_name = litellm_params.get("model")
    model_id = model["model_info"]["id"]
    # ### IF RPM SET - initialize a semaphore ###
    rpm = litellm_params.get("rpm", None)
    tpm = litellm_params.get("tpm", None)
    max_parallel_requests = litellm_params.get("max_parallel_requests", None)
    calculated_max_parallel_requests = calculate_max_parallel_requests(
        rpm=rpm,
        max_parallel_requests=max_parallel_requests,
        tpm=tpm,
        default_max_parallel_requests=litellm_router_instance.default_max_parallel_requests,
    )
    if calculated_max_parallel_requests:
        semaphore = asyncio.Semaphore(calculated_max_parallel_requests)
        cache_key = f"{model_id}_max_parallel_requests_client"
        litellm_router_instance.cache.set_cache(
            key=cache_key,
            value=semaphore,
            local_only=True,
        )

    ####  for OpenAI / Azure we need to initalize the Client for High Traffic ########
    custom_llm_provider = litellm_params.get("custom_llm_provider")
    custom_llm_provider = custom_llm_provider or model_name.split("/", 1)[0] or ""
    default_api_base = None
    default_api_key = None
    if custom_llm_provider in litellm.openai_compatible_providers:
        _, custom_llm_provider, api_key, api_base = litellm.get_llm_provider(
            model=model_name
        )
        default_api_base = api_base
        default_api_key = api_key

    if (
        model_name in litellm.open_ai_chat_completion_models
        or custom_llm_provider in litellm.openai_compatible_providers
        or custom_llm_provider == "azure"
        or custom_llm_provider == "azure_text"
        or custom_llm_provider == "custom_openai"
        or custom_llm_provider == "openai"
        or custom_llm_provider == "text-completion-openai"
        or "ft:gpt-3.5-turbo" in model_name
        or model_name in litellm.open_ai_embedding_models
    ):
        is_azure_ai_studio_model: bool = False
        if custom_llm_provider == "azure":
            if litellm.utils._is_non_openai_azure_model(model_name):
                is_azure_ai_studio_model = True
                custom_llm_provider = "openai"
                # remove azure prefx from model_name
                model_name = model_name.replace("azure/", "")
        # glorified / complicated reading of configs
        # user can pass vars directly or they can pas os.environ/AZURE_API_KEY, in which case we will read the env
        # we do this here because we init clients for Azure, OpenAI and we need to set the right key
        api_key = litellm_params.get("api_key") or default_api_key
        if api_key and isinstance(api_key, str) and api_key.startswith("os.environ/"):
            api_key_env_name = api_key.replace("os.environ/", "")
            api_key = litellm.get_secret(api_key_env_name)
            litellm_params["api_key"] = api_key

        api_base = litellm_params.get("api_base")
        base_url = litellm_params.get("base_url")
        api_base = (
            api_base or base_url or default_api_base
        )  # allow users to pass in `api_base` or `base_url` for azure
        if api_base and api_base.startswith("os.environ/"):
            api_base_env_name = api_base.replace("os.environ/", "")
            api_base = litellm.get_secret(api_base_env_name)
            litellm_params["api_base"] = api_base

        ## AZURE AI STUDIO MISTRAL CHECK ##
        """
        Make sure api base ends in /v1/

        if not, add it - https://github.com/BerriAI/litellm/issues/2279
        """
        if (
            is_azure_ai_studio_model is True
            and api_base is not None
            and isinstance(api_base, str)
            and not api_base.endswith("/v1/")
        ):
            # check if it ends with a trailing slash
            if api_base.endswith("/"):
                api_base += "v1/"
            elif api_base.endswith("/v1"):
                api_base += "/"
            else:
                api_base += "/v1/"

        api_version = litellm_params.get("api_version")
        if api_version and api_version.startswith("os.environ/"):
            api_version_env_name = api_version.replace("os.environ/", "")
            api_version = litellm.get_secret(api_version_env_name)
            litellm_params["api_version"] = api_version

        timeout = litellm_params.pop("timeout", None) or litellm.request_timeout
        if isinstance(timeout, str) and timeout.startswith("os.environ/"):
            timeout_env_name = timeout.replace("os.environ/", "")
            timeout = litellm.get_secret(timeout_env_name)
            litellm_params["timeout"] = timeout

        stream_timeout = litellm_params.pop(
            "stream_timeout", timeout
        )  # if no stream_timeout is set, default to timeout
        if isinstance(stream_timeout, str) and stream_timeout.startswith("os.environ/"):
            stream_timeout_env_name = stream_timeout.replace("os.environ/", "")
            stream_timeout = litellm.get_secret(stream_timeout_env_name)
            litellm_params["stream_timeout"] = stream_timeout

        max_retries = litellm_params.pop("max_retries", 0)  # router handles retry logic
        if isinstance(max_retries, str) and max_retries.startswith("os.environ/"):
            max_retries_env_name = max_retries.replace("os.environ/", "")
            max_retries = litellm.get_secret(max_retries_env_name)
            litellm_params["max_retries"] = max_retries

        organization = litellm_params.get("organization", None)
        if isinstance(organization, str) and organization.startswith("os.environ/"):
            organization_env_name = organization.replace("os.environ/", "")
            organization = litellm.get_secret(organization_env_name)
            litellm_params["organization"] = organization
        azure_ad_token_provider: Optional[Callable[[], str]] = None
        if litellm_params.get("tenant_id"):
            verbose_router_logger.debug("Using Azure AD Token Provider for Azure Auth")
            azure_ad_token_provider = get_azure_ad_token_from_entrata_id(
                tenant_id=litellm_params.get("tenant_id"),
                client_id=litellm_params.get("client_id"),
                client_secret=litellm_params.get("client_secret"),
            )

        if custom_llm_provider == "azure" or custom_llm_provider == "azure_text":
            if api_base is None or not isinstance(api_base, str):
                filtered_litellm_params = {
                    k: v for k, v in model["litellm_params"].items() if k != "api_key"
                }
                _filtered_model = {
                    "model_name": model["model_name"],
                    "litellm_params": filtered_litellm_params,
                }
                raise ValueError(
                    f"api_base is required for Azure OpenAI. Set it on your config. Model - {_filtered_model}"
                )
            azure_ad_token = litellm_params.get("azure_ad_token")
            if azure_ad_token is not None:
                if azure_ad_token.startswith("oidc/"):
                    azure_ad_token = get_azure_ad_token_from_oidc(azure_ad_token)
            elif (
                azure_ad_token_provider is None
                and litellm.enable_azure_ad_token_refresh is True
            ):
                try:
                    azure_ad_token_provider = get_azure_ad_token_provider()
                except ValueError:
                    verbose_router_logger.debug(
                        "Azure AD Token Provider could not be used."
                    )
            if api_version is None:
                api_version = os.getenv(
                    "AZURE_API_VERSION", litellm.AZURE_DEFAULT_API_VERSION
                )

            if "gateway.ai.cloudflare.com" in api_base:
                if not api_base.endswith("/"):
                    api_base += "/"
                azure_model = model_name.replace("azure/", "")
                api_base += f"{azure_model}"
                cache_key = f"{model_id}_async_client"
                _client = openai.AsyncAzureOpenAI(
                    api_key=api_key,
                    azure_ad_token=azure_ad_token,
                    azure_ad_token_provider=azure_ad_token_provider,
                    base_url=api_base,
                    api_version=api_version,
                    timeout=timeout,
                    max_retries=max_retries,
                    http_client=httpx.AsyncClient(
                        limits=httpx.Limits(
                            max_connections=1000, max_keepalive_connections=100
                        ),
                        verify=litellm.ssl_verify,
                    ),  # type: ignore
                )
                litellm_router_instance.cache.set_cache(
                    key=cache_key,
                    value=_client,
                    ttl=client_ttl,
                    local_only=True,
                )  # cache for 1 hr

                if should_initialize_sync_client(
                    litellm_router_instance=litellm_router_instance
                ):
                    cache_key = f"{model_id}_client"
                    _client = openai.AzureOpenAI(  # type: ignore
                        api_key=api_key,
                        azure_ad_token=azure_ad_token,
                        azure_ad_token_provider=azure_ad_token_provider,
                        base_url=api_base,
                        api_version=api_version,
                        timeout=timeout,
                        max_retries=max_retries,
                        http_client=httpx.Client(
                            limits=httpx.Limits(
                                max_connections=1000, max_keepalive_connections=100
                            ),
                            verify=litellm.ssl_verify,
                        ),  # type: ignore
                    )
                    litellm_router_instance.cache.set_cache(
                        key=cache_key,
                        value=_client,
                        ttl=client_ttl,
                        local_only=True,
                    )  # cache for 1 hr
                # streaming clients can have diff timeouts
                cache_key = f"{model_id}_stream_async_client"
                _client = openai.AsyncAzureOpenAI(  # type: ignore
                    api_key=api_key,
                    azure_ad_token=azure_ad_token,
                    azure_ad_token_provider=azure_ad_token_provider,
                    base_url=api_base,
                    api_version=api_version,
                    timeout=stream_timeout,
                    max_retries=max_retries,
                    http_client=httpx.AsyncClient(
                        limits=httpx.Limits(
                            max_connections=1000, max_keepalive_connections=100
                        ),
                        verify=litellm.ssl_verify,
                    ),  # type: ignore
                )
                litellm_router_instance.cache.set_cache(
                    key=cache_key,
                    value=_client,
                    ttl=client_ttl,
                    local_only=True,
                )  # cache for 1 hr

                if should_initialize_sync_client(
                    litellm_router_instance=litellm_router_instance
                ):
                    cache_key = f"{model_id}_stream_client"
                    _client = openai.AzureOpenAI(  # type: ignore
                        api_key=api_key,
                        azure_ad_token=azure_ad_token,
                        azure_ad_token_provider=azure_ad_token_provider,
                        base_url=api_base,
                        api_version=api_version,
                        timeout=stream_timeout,
                        max_retries=max_retries,
                        http_client=httpx.Client(
                            limits=httpx.Limits(
                                max_connections=1000, max_keepalive_connections=100
                            ),
                            verify=litellm.ssl_verify,
                        ),  # type: ignore
                    )
                    litellm_router_instance.cache.set_cache(
                        key=cache_key,
                        value=_client,
                        ttl=client_ttl,
                        local_only=True,
                    )  # cache for 1 hr
            else:
                _api_key = api_key
                if _api_key is not None and isinstance(_api_key, str):
                    # only show first 5 chars of api_key
                    _api_key = _api_key[:8] + "*" * 15
                verbose_router_logger.debug(
                    f"Initializing Azure OpenAI Client for {model_name}, Api Base: {str(api_base)}, Api Key:{_api_key}"
                )
                azure_client_params = {
                    "api_key": api_key,
                    "azure_endpoint": api_base,
                    "api_version": api_version,
                    "azure_ad_token": azure_ad_token,
                    "azure_ad_token_provider": azure_ad_token_provider,
                }

                if azure_ad_token_provider is not None:
                    azure_client_params["azure_ad_token_provider"] = (
                        azure_ad_token_provider
                    )
                from litellm.llms.AzureOpenAI.azure import (
                    select_azure_base_url_or_endpoint,
                )

                # this decides if we should set azure_endpoint or base_url on Azure OpenAI Client
                # required to support GPT-4 vision enhancements, since base_url needs to be set on Azure OpenAI Client
                azure_client_params = select_azure_base_url_or_endpoint(
                    azure_client_params
                )

                cache_key = f"{model_id}_async_client"
                _client = openai.AsyncAzureOpenAI(  # type: ignore
                    **azure_client_params,
                    timeout=timeout,
                    max_retries=max_retries,
                    http_client=httpx.AsyncClient(
                        limits=httpx.Limits(
                            max_connections=1000, max_keepalive_connections=100
                        ),
                        verify=litellm.ssl_verify,
                    ),  # type: ignore
                )
                litellm_router_instance.cache.set_cache(
                    key=cache_key,
                    value=_client,
                    ttl=client_ttl,
                    local_only=True,
                )  # cache for 1 hr
                if should_initialize_sync_client(
                    litellm_router_instance=litellm_router_instance
                ):
                    cache_key = f"{model_id}_client"
                    _client = openai.AzureOpenAI(  # type: ignore
                        **azure_client_params,
                        timeout=timeout,
                        max_retries=max_retries,
                        http_client=httpx.Client(
                            limits=httpx.Limits(
                                max_connections=1000, max_keepalive_connections=100
                            ),
                            verify=litellm.ssl_verify,
                        ),  # type: ignore
                    )
                    litellm_router_instance.cache.set_cache(
                        key=cache_key,
                        value=_client,
                        ttl=client_ttl,
                        local_only=True,
                    )  # cache for 1 hr

                # streaming clients should have diff timeouts
                cache_key = f"{model_id}_stream_async_client"
                _client = openai.AsyncAzureOpenAI(  # type: ignore
                    **azure_client_params,
                    timeout=stream_timeout,
                    max_retries=max_retries,
                    http_client=httpx.AsyncClient(
                        limits=httpx.Limits(
                            max_connections=1000, max_keepalive_connections=100
                        ),
                        verify=litellm.ssl_verify,
                    ),
                )
                litellm_router_instance.cache.set_cache(
                    key=cache_key,
                    value=_client,
                    ttl=client_ttl,
                    local_only=True,
                )  # cache for 1 hr

                if should_initialize_sync_client(
                    litellm_router_instance=litellm_router_instance
                ):
                    cache_key = f"{model_id}_stream_client"
                    _client = openai.AzureOpenAI(  # type: ignore
                        **azure_client_params,
                        timeout=stream_timeout,
                        max_retries=max_retries,
                        http_client=httpx.Client(
                            limits=httpx.Limits(
                                max_connections=1000, max_keepalive_connections=100
                            ),
                            verify=litellm.ssl_verify,
                        ),
                    )
                    litellm_router_instance.cache.set_cache(
                        key=cache_key,
                        value=_client,
                        ttl=client_ttl,
                        local_only=True,
                    )  # cache for 1 hr

        else:
            _api_key = api_key  # type: ignore
            if _api_key is not None and isinstance(_api_key, str):
                # only show first 5 chars of api_key
                _api_key = _api_key[:8] + "*" * 15
            verbose_router_logger.debug(
                f"Initializing OpenAI Client for {model_name}, Api Base:{str(api_base)}, Api Key:{_api_key}"
            )
            cache_key = f"{model_id}_async_client"
            _client = openai.AsyncOpenAI(  # type: ignore
                api_key=api_key,
                base_url=api_base,
                timeout=timeout,
                max_retries=max_retries,
                organization=organization,
                http_client=httpx.AsyncClient(
                    limits=httpx.Limits(
                        max_connections=1000, max_keepalive_connections=100
                    ),
                    verify=litellm.ssl_verify,
                ),  # type: ignore
            )
            litellm_router_instance.cache.set_cache(
                key=cache_key,
                value=_client,
                ttl=client_ttl,
                local_only=True,
            )  # cache for 1 hr

            if should_initialize_sync_client(
                litellm_router_instance=litellm_router_instance
            ):
                cache_key = f"{model_id}_client"
                _client = openai.OpenAI(  # type: ignore
                    api_key=api_key,
                    base_url=api_base,
                    timeout=timeout,
                    max_retries=max_retries,
                    organization=organization,
                    http_client=httpx.Client(
                        limits=httpx.Limits(
                            max_connections=1000, max_keepalive_connections=100
                        ),
                        verify=litellm.ssl_verify,
                    ),  # type: ignore
                )
                litellm_router_instance.cache.set_cache(
                    key=cache_key,
                    value=_client,
                    ttl=client_ttl,
                    local_only=True,
                )  # cache for 1 hr

            # streaming clients should have diff timeouts
            cache_key = f"{model_id}_stream_async_client"
            _client = openai.AsyncOpenAI(  # type: ignore
                api_key=api_key,
                base_url=api_base,
                timeout=stream_timeout,
                max_retries=max_retries,
                organization=organization,
                http_client=httpx.AsyncClient(
                    limits=httpx.Limits(
                        max_connections=1000, max_keepalive_connections=100
                    ),
                    verify=litellm.ssl_verify,
                ),  # type: ignore
            )
            litellm_router_instance.cache.set_cache(
                key=cache_key,
                value=_client,
                ttl=client_ttl,
                local_only=True,
            )  # cache for 1 hr

            if should_initialize_sync_client(
                litellm_router_instance=litellm_router_instance
            ):
                # streaming clients should have diff timeouts
                cache_key = f"{model_id}_stream_client"
                _client = openai.OpenAI(  # type: ignore
                    api_key=api_key,
                    base_url=api_base,
                    timeout=stream_timeout,
                    max_retries=max_retries,
                    organization=organization,
                    http_client=httpx.Client(
                        limits=httpx.Limits(
                            max_connections=1000, max_keepalive_connections=100
                        ),
                        verify=litellm.ssl_verify,
                    ),  # type: ignore
                )
                litellm_router_instance.cache.set_cache(
                    key=cache_key,
                    value=_client,
                    ttl=client_ttl,
                    local_only=True,
                )  # cache for 1 hr


def get_azure_ad_token_from_entrata_id(
    tenant_id: str, client_id: str, client_secret: str
) -> Callable[[], str]:
    from azure.identity import (
        ClientSecretCredential,
        DefaultAzureCredential,
        get_bearer_token_provider,
    )

    verbose_router_logger.debug("Getting Azure AD Token from Entrata ID")

    if tenant_id.startswith("os.environ/"):
        tenant_id = litellm.get_secret(tenant_id)

    if client_id.startswith("os.environ/"):
        client_id = litellm.get_secret(client_id)

    if client_secret.startswith("os.environ/"):
        client_secret = litellm.get_secret(client_secret)
    verbose_router_logger.debug(
        "tenant_id %s, client_id %s, client_secret %s",
        tenant_id,
        client_id,
        client_secret,
    )
    credential = ClientSecretCredential(tenant_id, client_id, client_secret)

    verbose_router_logger.debug("credential %s", credential)

    token_provider = get_bearer_token_provider(
        credential, "https://cognitiveservices.azure.com/.default"
    )

    verbose_router_logger.debug("token_provider %s", token_provider)

    return token_provider
