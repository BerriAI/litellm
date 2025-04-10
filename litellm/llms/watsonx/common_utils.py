from typing import Dict, List, Optional, Union, cast

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
        headers: Optional[Union[Dict, httpx.Headers]] = None,
    ):
        super().__init__(status_code=status_code, message=message, headers=headers)


iam_token_cache = InMemoryCache()


def get_watsonx_iam_url():
    return (
        get_secret_str("WATSONX_IAM_URL") or "https://iam.cloud.ibm.com/identity/token"
    )


def generate_iam_token(api_key=None, **params) -> str:
    result: Optional[str] = iam_token_cache.get_cache(api_key)  # type: ignore

    if result is None:
        headers = {}
        headers["Content-Type"] = "application/x-www-form-urlencoded"
        if api_key is None:
            api_key = get_secret_str("WX_API_KEY") or get_secret_str("WATSONX_API_KEY")
        if api_key is None:
            raise ValueError("API key is required")
        headers["Accept"] = "application/json"
        data = {
            "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
            "apikey": api_key,
        }
        iam_token_url = get_watsonx_iam_url()
        verbose_logger.debug(
            "calling ibm `/identity/token` to retrieve IAM token.\nURL=%s\nheaders=%s\ndata=%s",
            iam_token_url,
            headers,
            data,
        )
        response = litellm.module_level_client.post(
            url=iam_token_url, data=data, headers=headers
        )
        response.raise_for_status()
        json_data = response.json()

        result = json_data["access_token"]
        iam_token_cache.set_cache(
            key=api_key,
            value=result,
            ttl=json_data["expires_in"] - 10,  # leave some buffer
        )

    return cast(str, result)


def _generate_watsonx_token(api_key: Optional[str], token: Optional[str]) -> str:
    if token is not None:
        return token
    token = generate_iam_token(api_key)
    return token


def _get_api_params(
    params: dict,
) -> WatsonXAPIParams:
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
            get_secret_str("WATSONX_PROJECT_ID")
            or get_secret_str("WX_PROJECT_ID")
            or get_secret_str("PROJECT_ID")
        )
    if region_name is None:
        region_name = (
            get_secret_str("WATSONX_REGION")
            or get_secret_str("WX_REGION")
            or get_secret_str("REGION")
        )
    if space_id is None:
        space_id = (
            get_secret_str("WATSONX_DEPLOYMENT_SPACE_ID")
            or get_secret_str("WATSONX_SPACE_ID")
            or get_secret_str("WX_SPACE_ID")
            or get_secret_str("SPACE_ID")
        )

    if project_id is None:
        raise WatsonXAIError(
            status_code=401,
            message="Error: Watsonx project_id not set. Set WX_PROJECT_ID in environment variables or pass in as a parameter.",
        )

    return WatsonXAPIParams(
        project_id=project_id,
        space_id=space_id,
        region_name=region_name,
    )


def convert_watsonx_messages_to_prompt(
    model: str,
    messages: List[AllMessageValues],
    provider: str,
    custom_prompt_dict: Dict,
) -> str:
    # handle anthropic prompts and amazon titan prompts
    if model in custom_prompt_dict:
        # check if the model has a registered custom prompt
        model_prompt_dict = custom_prompt_dict[model]
        prompt = ptf.custom_prompt(
            messages=messages,
            role_dict=model_prompt_dict.get(
                "role_dict", model_prompt_dict.get("roles")
            ),
            initial_prompt_value=model_prompt_dict.get("initial_prompt_value", ""),
            final_prompt_value=model_prompt_dict.get("final_prompt_value", ""),
            bos_token=model_prompt_dict.get("bos_token", ""),
            eos_token=model_prompt_dict.get("eos_token", ""),
        )
        return prompt
    elif provider == "ibm-mistralai":
        prompt = ptf.mistral_instruct_pt(messages=messages)
    else:
        prompt: str = ptf.prompt_factory(  # type: ignore
            model=model, messages=messages, custom_llm_provider="watsonx"
        )
    return prompt


# Mixin class for shared IBM Watson X functionality
class IBMWatsonXMixin:
    def validate_environment(
        self,
        headers: Dict,
        model: str,
        messages: List[AllMessageValues],
        optional_params: Dict,
        litellm_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> Dict:
        default_headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        if "Authorization" in headers:
            return {**default_headers, **headers}
        token = cast(
            Optional[str],
            optional_params.get("token") or get_secret_str("WATSONX_TOKEN"),
        )
        if token:
            headers["Authorization"] = f"Bearer {token}"
        elif zen_api_key := get_secret_str("WATSONX_ZENAPIKEY"):
            headers["Authorization"] = f"ZenApiKey {zen_api_key}"
        else:
            token = _generate_watsonx_token(api_key=api_key, token=token)
            # build auth headers
            headers["Authorization"] = f"Bearer {token}"
        return {**default_headers, **headers}

    def _get_base_url(self, api_base: Optional[str]) -> str:
        url = (
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

    def _add_api_version_to_url(self, url: str, api_version: Optional[str]) -> str:
        api_version = api_version or litellm.WATSONX_DEFAULT_API_VERSION
        url = url + f"?version={api_version}"

        return url

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[Dict, httpx.Headers]
    ) -> BaseLLMException:
        return WatsonXAIError(
            status_code=status_code, message=error_message, headers=headers
        )

    @staticmethod
    def get_watsonx_credentials(
        optional_params: dict, api_key: Optional[str], api_base: Optional[str]
    ) -> WatsonXCredentials:
        api_key = (
            api_key
            or optional_params.pop("apikey", None)
            or get_secret_str("WATSONX_APIKEY")
            or get_secret_str("WATSONX_API_KEY")
            or get_secret_str("WX_API_KEY")
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

        wx_credentials = optional_params.pop(
            "wx_credentials",
            optional_params.pop(
                "watsonx_credentials", None
            ),  # follow {provider}_credentials, same as vertex ai
        )

        token: Optional[str] = None

        if wx_credentials is not None:
            api_base = wx_credentials.get("url", api_base)
            api_key = wx_credentials.get(
                "apikey", wx_credentials.get("api_key", api_key)
            )
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
        return WatsonXCredentials(
            api_key=api_key, api_base=api_base, token=cast(Optional[str], token)
        )

    def _prepare_payload(self, model: str, api_params: WatsonXAPIParams) -> dict:
        payload: dict = {}
        if model.startswith("deployment/"):
            if api_params["space_id"] is None:
                raise WatsonXAIError(
                    status_code=401,
                    message="Error: space_id is required for models called using the 'deployment/' endpoint. Pass in the space_id as a parameter or set it in the WX_SPACE_ID environment variable.",
                )
            payload["space_id"] = api_params["space_id"]
            return payload
        payload["model_id"] = model
        payload["project_id"] = api_params["project_id"]
        return payload
