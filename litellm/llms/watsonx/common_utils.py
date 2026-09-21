from typing import Final, cast

import httpx

import litellm
from litellm import verbose_logger
from litellm.caching import InMemoryCache
from litellm.litellm_core_utils.prompt_templates import factory as ptf
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues
from litellm.types.llms.watsonx import WatsonXAPIParams, WatsonXCredentials


class WatsonXAIError(BaseLLMException):
    def __init__(
        self,
        status_code: int,
        message: str,
        headers: dict | httpx.Headers | None = None,
    ):
        super().__init__(status_code=status_code, message=message, headers=headers)


iam_token_cache: Final = InMemoryCache()


def get_watsonx_iam_url():
    return get_secret_str("WATSONX_IAM_URL") or "https://iam.cloud.ibm.com/identity/token"


def generate_iam_token(api_key=None, **params) -> str:
    result: str | None = iam_token_cache.get_cache(api_key)

    if result is None:
        headers: Final = {}
        headers["Content-Type"] = "application/x-www-form-urlencoded"
        if api_key is None:
            api_key = (
                get_secret_str("WX_API_KEY")
                or get_secret_str("WATSONX_API_KEY")
                or get_secret_str("WATSONX_APIKEY")
                or get_secret_str("WATSONX_ZENAPIKEY")
            )
        if api_key is None:
            raise ValueError("API key is required")
        headers["Accept"] = "application/json"
        data: Final = {
            "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
            "apikey": api_key,
        }
        iam_token_url: Final = get_watsonx_iam_url()
        verbose_logger.debug(
            "calling ibm `/identity/token` to retrieve IAM token.\nURL=%s\nheaders=%s\ndata=%s",
            iam_token_url,
            headers,
            data,
        )
        response: Final = litellm.module_level_client.post(url=iam_token_url, data=data, headers=headers)
        response.raise_for_status()
        json_data: Final = response.json()

        result = json_data["access_token"]
        iam_token_cache.set_cache(
            key=api_key,
            value=result,
            ttl=json_data["expires_in"] - 10,  # leave some buffer
        )

    return cast(str, result)


def _generate_watsonx_token(api_key: str | None, token: str | None) -> str:
    if token is not None:
        return token
    token = generate_iam_token(api_key)
    return token


def _get_api_params(params: dict, model: str | None = None) -> WatsonXAPIParams:
    """
    Find watsonx.ai credentials in the params or environment variables and return the headers for authentication.
    """
    # Load auth variables from params
    project_id = params.pop(
        "project_id", params.pop("watsonx_project", None)
    )  # watsonx.ai project_id - allow 'watsonx_project' to be consistent with how vertex project implementation works -> reduce provider-specific params
    space_id = params.pop("space_id", None)  # watsonx.ai deployment space_id
    region_name = params.pop("region_name", params.pop("region", None))
    if region_name is None:
        region_name = params.pop(
            "watsonx_region_name", params.pop("watsonx_region", None)
        )  # consistent with how vertex ai + aws regions are accepted

    # Load auth variables from environment variables
    if project_id is None:
        project_id = (
            get_secret_str("WATSONX_PROJECT_ID") or get_secret_str("WX_PROJECT_ID") or get_secret_str("PROJECT_ID")
        )
    if region_name is None:
        region_name = get_secret_str("WATSONX_REGION") or get_secret_str("WX_REGION") or get_secret_str("REGION")
    if space_id is None:
        space_id = (
            get_secret_str("WATSONX_DEPLOYMENT_SPACE_ID")
            or get_secret_str("WATSONX_SPACE_ID")
            or get_secret_str("WX_SPACE_ID")
            or get_secret_str("SPACE_ID")
        )

    if project_id is None and space_id is None and model is not None and not model.startswith("deployment/"):
        raise WatsonXAIError(
            status_code=401,
            message="Error: Watsonx project_id and space_id not set. Set WX_PROJECT_ID or WX_SPACE_ID in environment variables or pass in as a parameter.",
        )

    return WatsonXAPIParams(
        project_id=project_id,
        space_id=space_id,
        region_name=region_name,
    )


async def _aconvert_watsonx_messages_core(
    model: str,
    messages: list[AllMessageValues],
    provider: str,
    custom_prompt_dict: dict,
    apply_template_fn,
) -> str:
    """Async core logic for converting watsonx messages to prompt"""
    from litellm.types.llms.watsonx import WatsonXModelPattern

    # handle anthropic prompts and amazon titan prompts
    if model in custom_prompt_dict:
        model_prompt_dict: Final = custom_prompt_dict[model]
        return ptf.custom_prompt(
            messages=messages,
            role_dict=model_prompt_dict.get("role_dict", model_prompt_dict.get("roles")),
            initial_prompt_value=model_prompt_dict.get("initial_prompt_value", ""),
            final_prompt_value=model_prompt_dict.get("final_prompt_value", ""),
            bos_token=model_prompt_dict.get("bos_token", ""),
            eos_token=model_prompt_dict.get("eos_token", ""),
        )
    elif provider == WatsonXModelPattern.IBM_MISTRALAI.value:
        return ptf.mistral_instruct_pt(messages=messages)
    else:
        # Try applying specific template first
        result: Final = await apply_template_fn(model=model, messages=messages)
        if result:
            return result
        # Fallback to default
        return ptf.prompt_factory(model=model, messages=messages, custom_llm_provider="watsonx")


def _convert_watsonx_messages_core(
    model: str,
    messages: list[AllMessageValues],
    provider: str,
    custom_prompt_dict: dict,
    apply_template_fn,
) -> str:
    """Sync core logic for converting watsonx messages to prompt"""
    from litellm.types.llms.watsonx import WatsonXModelPattern

    # handle anthropic prompts and amazon titan prompts
    if model in custom_prompt_dict:
        model_prompt_dict: Final = custom_prompt_dict[model]
        return ptf.custom_prompt(
            messages=messages,
            role_dict=model_prompt_dict.get("role_dict", model_prompt_dict.get("roles")),
            initial_prompt_value=model_prompt_dict.get("initial_prompt_value", ""),
            final_prompt_value=model_prompt_dict.get("final_prompt_value", ""),
            bos_token=model_prompt_dict.get("bos_token", ""),
            eos_token=model_prompt_dict.get("eos_token", ""),
        )
    elif provider == WatsonXModelPattern.IBM_MISTRALAI.value:
        return ptf.mistral_instruct_pt(messages=messages)
    else:
        # Try applying specific template first
        result: Final = apply_template_fn(model=model, messages=messages)
        if result:
            return result
        # Fallback to default
        return ptf.prompt_factory(model=model, messages=messages, custom_llm_provider="watsonx")


async def aconvert_watsonx_messages_to_prompt(
    model: str,
    messages: list[AllMessageValues],
    provider: str,
    custom_prompt_dict: dict,
) -> str:
    """Async version of convert_watsonx_messages_to_prompt"""
    from litellm.llms.watsonx.chat.transformation import IBMWatsonXChatConfig

    return await _aconvert_watsonx_messages_core(
        model=model,
        messages=messages,
        provider=provider,
        custom_prompt_dict=custom_prompt_dict,
        apply_template_fn=IBMWatsonXChatConfig.aapply_prompt_template,
    )


def convert_watsonx_messages_to_prompt(
    model: str,
    messages: list[AllMessageValues],
    provider: str,
    custom_prompt_dict: dict,
) -> str:
    """Sync version of convert_watsonx_messages_to_prompt"""
    from litellm.llms.watsonx.chat.transformation import IBMWatsonXChatConfig

    return _convert_watsonx_messages_core(
        model=model,
        messages=messages,
        provider=provider,
        custom_prompt_dict=custom_prompt_dict,
        apply_template_fn=IBMWatsonXChatConfig.apply_prompt_template,
    )


# Mixin class for shared IBM Watson X functionality
class IBMWatsonXMixin:
    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: list[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict:
        default_headers: Final = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        if "Authorization" in headers:
            return {**default_headers, **headers}
        token = cast(
            str | None,
            optional_params.get("token") or get_secret_str("WATSONX_TOKEN"),
        )
        zen_api_key: Final = cast(
            str | None,
            optional_params.pop("zen_api_key", None) or get_secret_str("WATSONX_ZENAPIKEY"),
        )
        if token:
            headers["Authorization"] = f"Bearer {token}"
        elif zen_api_key:
            headers["Authorization"] = f"ZenApiKey {zen_api_key}"
        else:
            token = _generate_watsonx_token(api_key=api_key, token=token)
            # build auth headers
            headers["Authorization"] = f"Bearer {token}"
        return {**default_headers, **headers}

    def _get_base_url(self, api_base: str | None) -> str:
        url: Final = (
            api_base
            or get_secret_str("WATSONX_API_BASE")  # consistent with 'AZURE_API_BASE'
            or get_secret_str("WATSONX_URL")
            or get_secret_str("WX_URL")
            or get_secret_str("WML_URL")
        )

        if url is None:
            raise WatsonXAIError(
                status_code=401,
                message="Error: Watsonx URL not set. Set WATSONX_API_BASE in environment variables or pass in as parameter - 'api_base='.",
            )
        return url

    def _add_api_version_to_url(self, url: str, api_version: str | None) -> str:
        api_version = api_version or litellm.WATSONX_DEFAULT_API_VERSION
        url = url + f"?version={api_version}"

        return url

    def get_error_class(self, error_message: str, status_code: int, headers: dict | httpx.Headers) -> BaseLLMException:
        return WatsonXAIError(status_code=status_code, message=error_message, headers=headers)

    @staticmethod
    def get_watsonx_credentials(optional_params: dict, api_key: str | None, api_base: str | None) -> WatsonXCredentials:
        api_key = (
            api_key
            or optional_params.pop("apikey", None)
            or get_secret_str("WATSONX_APIKEY")
            or get_secret_str("WATSONX_API_KEY")
            or get_secret_str("WX_API_KEY")
            or get_secret_str("WATSONX_ZENAPIKEY")
        )

        api_base = (
            api_base
            or optional_params.pop(
                "url",
                optional_params.pop("api_base", optional_params.pop("base_url", None)),
            )
            or get_secret_str("WATSONX_API_BASE")
            or get_secret_str("WATSONX_URL")
            or get_secret_str("WX_URL")
            or get_secret_str("WML_URL")
        )

        wx_credentials: Final = optional_params.pop(
            "wx_credentials",
            optional_params.pop("watsonx_credentials", None),  # follow {provider}_credentials, same as vertex ai
        )

        token: str | None = None

        if wx_credentials is not None:
            api_base = wx_credentials.get("url", api_base)
            api_key = wx_credentials.get("apikey", wx_credentials.get("api_key", api_key))
            token = wx_credentials.get(
                "token",
                wx_credentials.get(
                    "watsonx_token", None
                ),  # follow format of {provider}_token, same as azure - e.g. 'azure_ad_token=..'
            )
        if api_key is None or not isinstance(api_key, str):
            raise WatsonXAIError(
                status_code=401,
                message="Error: Watsonx API key not set. Set WATSONX_API_KEY in environment variables or pass in as parameter - 'api_key='.",
            )
        if api_base is None or not isinstance(api_base, str):
            raise WatsonXAIError(
                status_code=401,
                message="Error: Watsonx API base not set. Set WATSONX_API_BASE in environment variables or pass in as parameter - 'api_base='.",
            )
        return WatsonXCredentials(api_key=api_key, api_base=api_base, token=cast(str | None, token))

    def _prepare_payload(self, model: str, api_params: WatsonXAPIParams) -> dict:
        payload: Final[dict] = {}
        if model.startswith("deployment/"):
            return {}  # Deployment models do not support 'space_id' or 'project_id' in their payload
        payload["model_id"] = model
        if api_params["project_id"] is not None:
            payload["project_id"] = api_params["project_id"]
        else:
            payload["space_id"] = api_params["space_id"]
        return payload
