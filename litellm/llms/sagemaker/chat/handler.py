import json
from copy import deepcopy
from typing import Callable, Optional, Union

import httpx

from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM
from litellm.utils import ModelResponse, get_secret

from ..common_utils import AWSEventStreamDecoder
from .transformation import SagemakerChatConfig


class SagemakerChatHandler(BaseAWSLLM):

    def _load_credentials(
        self,
        optional_params: dict,
    ):
        try:
            from botocore.credentials import Credentials
        except ImportError:
            raise ImportError("Missing boto3 to call bedrock. Run 'pip install boto3'.")
        ## CREDENTIALS ##
        # pop aws_secret_access_key, aws_access_key_id, aws_session_token, aws_region_name from kwargs, since completion calls fail with them
        aws_secret_access_key = optional_params.pop("aws_secret_access_key", None)
        aws_access_key_id = optional_params.pop("aws_access_key_id", None)
        aws_session_token = optional_params.pop("aws_session_token", None)
        aws_region_name = optional_params.pop("aws_region_name", None)
        aws_role_name = optional_params.pop("aws_role_name", None)
        aws_session_name = optional_params.pop("aws_session_name", None)
        aws_profile_name = optional_params.pop("aws_profile_name", None)
        optional_params.pop(
            "aws_bedrock_runtime_endpoint", None
        )  # https://bedrock-runtime.{region_name}.amazonaws.com
        aws_web_identity_token = optional_params.pop("aws_web_identity_token", None)
        aws_sts_endpoint = optional_params.pop("aws_sts_endpoint", None)

        ### SET REGION NAME ###
        if aws_region_name is None:
            # check env #
            litellm_aws_region_name = get_secret("AWS_REGION_NAME", None)

            if litellm_aws_region_name is not None and isinstance(
                litellm_aws_region_name, str
            ):
                aws_region_name = litellm_aws_region_name

            standard_aws_region_name = get_secret("AWS_REGION", None)
            if standard_aws_region_name is not None and isinstance(
                standard_aws_region_name, str
            ):
                aws_region_name = standard_aws_region_name

            if aws_region_name is None:
                aws_region_name = "us-west-2"

        credentials: Credentials = self.get_credentials(
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token,
            aws_region_name=aws_region_name,
            aws_session_name=aws_session_name,
            aws_profile_name=aws_profile_name,
            aws_role_name=aws_role_name,
            aws_web_identity_token=aws_web_identity_token,
            aws_sts_endpoint=aws_sts_endpoint,
        )
        return credentials, aws_region_name

    def _prepare_request(
        self,
        credentials,
        model: str,
        data: dict,
        optional_params: dict,
        aws_region_name: str,
        extra_headers: Optional[dict] = None,
    ):
        try:
            from botocore.auth import SigV4Auth
            from botocore.awsrequest import AWSRequest
        except ImportError:
            raise ImportError("Missing boto3 to call bedrock. Run 'pip install boto3'.")

        sigv4 = SigV4Auth(credentials, "sagemaker", aws_region_name)
        if optional_params.get("stream") is True:
            api_base = f"https://runtime.sagemaker.{aws_region_name}.amazonaws.com/endpoints/{model}/invocations-response-stream"
        else:
            api_base = f"https://runtime.sagemaker.{aws_region_name}.amazonaws.com/endpoints/{model}/invocations"

        sagemaker_base_url = optional_params.get("sagemaker_base_url", None)
        if sagemaker_base_url is not None:
            api_base = sagemaker_base_url

        encoded_data = json.dumps(data).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if extra_headers is not None:
            headers = {"Content-Type": "application/json", **extra_headers}
        request = AWSRequest(
            method="POST", url=api_base, data=encoded_data, headers=headers
        )
        sigv4.add_auth(request)
        if (
            extra_headers is not None and "Authorization" in extra_headers
        ):  # prevent sigv4 from overwriting the auth header
            request.headers["Authorization"] = extra_headers["Authorization"]

        prepped_request = request.prepare()

        return prepped_request

    def completion(
        self,
        model: str,
        messages: list,
        model_response: ModelResponse,
        print_verbose: Callable,
        encoding,
        logging_obj,
        optional_params: dict,
        litellm_params: dict,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        custom_prompt_dict={},
        logger_fn=None,
        acompletion: bool = False,
        headers: dict = {},
    ):

        # pop streaming if it's in the optional params as 'stream' raises an error with sagemaker
        credentials, aws_region_name = self._load_credentials(optional_params)
        inference_params = deepcopy(optional_params)
        stream = inference_params.pop("stream", None)

        from litellm.llms.openai_like.chat.handler import OpenAILikeChatHandler

        openai_like_chat_completions = OpenAILikeChatHandler()
        inference_params["stream"] = True if stream is True else False
        _data = SagemakerChatConfig().transform_request(
            model=model,
            messages=messages,
            optional_params=inference_params,
            litellm_params=litellm_params,
            headers=headers,
        )

        prepared_request = self._prepare_request(
            model=model,
            data=_data,
            optional_params=optional_params,
            credentials=credentials,
            aws_region_name=aws_region_name,
        )

        custom_stream_decoder = AWSEventStreamDecoder(model="", is_messages_api=True)

        return openai_like_chat_completions.completion(
            model=model,
            messages=messages,
            api_base=prepared_request.url,
            api_key=None,
            custom_prompt_dict=custom_prompt_dict,
            model_response=model_response,
            print_verbose=print_verbose,
            logging_obj=logging_obj,
            optional_params=inference_params,
            acompletion=acompletion,
            litellm_params=litellm_params,
            logger_fn=logger_fn,
            timeout=timeout,
            encoding=encoding,
            headers=prepared_request.headers,  # type: ignore
            custom_endpoint=True,
            custom_llm_provider="sagemaker_chat",
            streaming_decoder=custom_stream_decoder,  # type: ignore
        )
