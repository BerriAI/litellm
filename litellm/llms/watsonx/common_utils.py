from typing import Callable, Dict, List, Optional, Union, cast

import httpx

import litellm
from litellm import verbose_logger
from litellm.caching import InMemoryCache
from litellm.litellm_core_utils.prompt_templates import factory as ptf
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues
from litellm.types.llms.watsonx import WatsonXAPIParams


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
        response = httpx.post(iam_token_url, data=data, headers=headers)
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
    print_verbose: Optional[Callable] = None,
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
