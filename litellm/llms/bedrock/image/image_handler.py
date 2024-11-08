import copy
import json
import os
from typing import Any, List, Optional

import httpx
from openai.types.image import Image

import litellm
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, _get_httpx_client
from litellm.types.utils import ImageResponse
from litellm.utils import print_verbose

from ...base_aws_llm import BaseAWSLLM
from ..common_utils import BedrockError


class BedrockImageGeneration(BaseAWSLLM):
    """
    Bedrock Image Generation handler
    """

    def image_generation(  # noqa: PLR0915
        self,
        model: str,
        prompt: str,
        model_response: ImageResponse,
        optional_params: dict,
        logging_obj: Any,
        timeout=None,
        aimg_generation: bool = False,
        api_base: Optional[str] = None,
        extra_headers: Optional[dict] = None,
        client: Optional[Any] = None,
    ):
        try:
            import boto3
            from botocore.auth import SigV4Auth
            from botocore.awsrequest import AWSRequest
            from botocore.credentials import Credentials
        except ImportError:
            raise ImportError("Missing boto3 to call bedrock. Run 'pip install boto3'.")
        boto3_credentials_info = self._get_boto_credentials_from_optional_params(
            optional_params
        )

        ### SET RUNTIME ENDPOINT ###
        modelId = model
        endpoint_url, proxy_endpoint_url = self.get_runtime_endpoint(
            api_base=api_base,
            aws_bedrock_runtime_endpoint=boto3_credentials_info.aws_bedrock_runtime_endpoint,
            aws_region_name=boto3_credentials_info.aws_region_name,
        )
        proxy_endpoint_url = f"{proxy_endpoint_url}/model/{modelId}/invoke"
        sigv4 = SigV4Auth(
            boto3_credentials_info.credentials,
            "bedrock",
            boto3_credentials_info.aws_region_name,
        )

        # transform request
        ### FORMAT IMAGE GENERATION INPUT ###
        provider = model.split(".")[0]
        inference_params = copy.deepcopy(optional_params)
        inference_params.pop(
            "user", None
        )  # make sure user is not passed in for bedrock call
        data = {}
        if provider == "stability":
            prompt = prompt.replace(os.linesep, " ")
            ## LOAD CONFIG
            config = litellm.AmazonStabilityConfig.get_config()
            for k, v in config.items():
                if (
                    k not in inference_params
                ):  # completion(top_k=3) > anthropic_config(top_k=3) <- allows for dynamic variables to be passed in
                    inference_params[k] = v
            data = {"text_prompts": [{"text": prompt, "weight": 1}], **inference_params}
        else:
            raise BedrockError(
                status_code=422, message=f"Unsupported model={model}, passed in"
            )

        # Make POST Request
        body = json.dumps(data).encode("utf-8")

        headers = {"Content-Type": "application/json"}
        if extra_headers is not None:
            headers = {"Content-Type": "application/json", **extra_headers}
        request = AWSRequest(
            method="POST", url=proxy_endpoint_url, data=body, headers=headers
        )
        sigv4.add_auth(request)
        if (
            extra_headers is not None and "Authorization" in extra_headers
        ):  # prevent sigv4 from overwriting the auth header
            request.headers["Authorization"] = extra_headers["Authorization"]
        prepped = request.prepare()

        ## LOGGING
        logging_obj.pre_call(
            input=prompt,
            api_key="",
            additional_args={
                "complete_input_dict": data,
                "api_base": proxy_endpoint_url,
                "headers": prepped.headers,
            },
        )

        if client is None or isinstance(client, AsyncHTTPHandler):
            _params = {}
            if timeout is not None:
                if isinstance(timeout, float) or isinstance(timeout, int):
                    timeout = httpx.Timeout(timeout)
                _params["timeout"] = timeout
            client = _get_httpx_client(_params)  # type: ignore
        else:
            client = client

        try:
            response = client.post(url=proxy_endpoint_url, headers=prepped.headers, data=body)  # type: ignore
            response.raise_for_status()
        except httpx.HTTPStatusError as err:
            error_code = err.response.status_code
            raise BedrockError(status_code=error_code, message=err.response.text)
        except httpx.TimeoutException:
            raise BedrockError(status_code=408, message="Timeout error occurred.")

        response_body = response.json()

        ## LOGGING
        if logging_obj is not None:
            logging_obj.post_call(
                input=prompt,
                api_key="",
                original_response=response.text,
                additional_args={"complete_input_dict": data},
            )
        print_verbose("raw model_response: %s", response.text)

        ### FORMAT RESPONSE TO OPENAI FORMAT ###
        if response_body is None:
            raise Exception("Error in response object format")

        if model_response is None:
            model_response = ImageResponse()

        image_list: List[Image] = []
        for artifact in response_body["artifacts"]:
            _image = Image(b64_json=artifact["base64"])
            image_list.append(_image)

        model_response.data = image_list
        return model_response

    async def async_image_generation(self):
        pass
