from typing import Callable, Optional, cast

import httpx

import litellm
from litellm import verbose_logger
from litellm.caching import InMemoryCache
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.watsonx import WatsonXAPIParams


class WatsonXAIError(Exception):
    def __init__(self, status_code, message, url: Optional[str] = None):
        self.status_code = status_code
        self.message = message
        url = url or "https://https://us-south.ml.cloud.ibm.com"
        self.request = httpx.Request(method="POST", url=url)
        self.response = httpx.Response(status_code=status_code, request=self.request)
        super().__init__(
            self.message
        )  # Call the base class constructor with the parameters it needs


iam_token_cache = InMemoryCache()


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
        verbose_logger.debug(
            "calling ibm `/identity/token` to retrieve IAM token.\nURL=%s\nheaders=%s\ndata=%s",
            "https://iam.cloud.ibm.com/identity/token",
            headers,
            data,
        )
        response = httpx.post(
            "https://iam.cloud.ibm.com/identity/token", data=data, headers=headers
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


def _get_api_params(
    params: dict,
    print_verbose: Optional[Callable] = None,
    generate_token: Optional[bool] = True,
) -> WatsonXAPIParams:
    """
    Find watsonx.ai credentials in the params or environment variables and return the headers for authentication.
    """
    # Load auth variables from params
    url = params.pop("url", params.pop("api_base", params.pop("base_url", None)))
    api_key = params.pop("apikey", None)
    token = params.pop("token", None)
    project_id = params.pop(
        "project_id", params.pop("watsonx_project", None)
    )  # watsonx.ai project_id - allow 'watsonx_project' to be consistent with how vertex project implementation works -> reduce provider-specific params
    space_id = params.pop("space_id", None)  # watsonx.ai deployment space_id
    region_name = params.pop("region_name", params.pop("region", None))
    if region_name is None:
        region_name = params.pop(
            "watsonx_region_name", params.pop("watsonx_region", None)
        )  # consistent with how vertex ai + aws regions are accepted
    wx_credentials = params.pop(
        "wx_credentials",
        params.pop(
            "watsonx_credentials", None
        ),  # follow {provider}_credentials, same as vertex ai
    )
    api_version = params.pop("api_version", litellm.WATSONX_DEFAULT_API_VERSION)
    # Load auth variables from environment variables
    if url is None:
        url = (
            get_secret_str("WATSONX_API_BASE")  # consistent with 'AZURE_API_BASE'
            or get_secret_str("WATSONX_URL")
            or get_secret_str("WX_URL")
            or get_secret_str("WML_URL")
        )
    if api_key is None:
        api_key = (
            get_secret_str("WATSONX_APIKEY")
            or get_secret_str("WATSONX_API_KEY")
            or get_secret_str("WX_API_KEY")
        )
    if token is None:
        token = get_secret_str("WATSONX_TOKEN") or get_secret_str("WX_TOKEN")
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

    # credentials parsing
    if wx_credentials is not None:
        url = wx_credentials.get("url", url)
        api_key = wx_credentials.get("apikey", wx_credentials.get("api_key", api_key))
        token = wx_credentials.get(
            "token",
            wx_credentials.get(
                "watsonx_token", token
            ),  # follow format of {provider}_token, same as azure - e.g. 'azure_ad_token=..'
        )

    # verify that all required credentials are present
    if url is None:
        raise WatsonXAIError(
            status_code=401,
            message="Error: Watsonx URL not set. Set WX_URL in environment variables or pass in as a parameter.",
        )

    if token is None and api_key is not None and generate_token:
        # generate the auth token
        if print_verbose is not None:
            print_verbose("Generating IAM token for Watsonx.ai")
        token = generate_iam_token(api_key)
    elif token is None and api_key is None:
        raise WatsonXAIError(
            status_code=401,
            url=url,
            message="Error: API key or token not found. Set WX_API_KEY or WX_TOKEN in environment variables or pass in as a parameter.",
        )
    if project_id is None:
        raise WatsonXAIError(
            status_code=401,
            url=url,
            message="Error: Watsonx project_id not set. Set WX_PROJECT_ID in environment variables or pass in as a parameter.",
        )

    return WatsonXAPIParams(
        url=url,
        api_key=api_key,
        token=cast(str, token),
        project_id=project_id,
        space_id=space_id,
        region_name=region_name,
        api_version=api_version,
    )
