"""
Handles image gen calls to Bedrock's `/invoke` endpoint 
"""

import copy
import json
import os
from typing import Any, List

from openai.types.image import Image

import litellm
from litellm.types.utils import ImageResponse

from .common_utils import BedrockError, init_bedrock_client


def image_generation(
    model: str,
    prompt: str,
    model_response: ImageResponse,
    optional_params: dict,
    logging_obj: Any,
    timeout=None,
    aimg_generation=False,
):
    """
    Bedrock Image Gen endpoint support
    """
    ### BOTO3 INIT ###
    # pop aws_secret_access_key, aws_access_key_id, aws_region_name from kwargs, since completion calls fail with them
    aws_secret_access_key = optional_params.pop("aws_secret_access_key", None)
    aws_access_key_id = optional_params.pop("aws_access_key_id", None)
    aws_region_name = optional_params.pop("aws_region_name", None)
    aws_role_name = optional_params.pop("aws_role_name", None)
    aws_session_name = optional_params.pop("aws_session_name", None)
    aws_bedrock_runtime_endpoint = optional_params.pop(
        "aws_bedrock_runtime_endpoint", None
    )
    aws_web_identity_token = optional_params.pop("aws_web_identity_token", None)

    # use passed in BedrockRuntime.Client if provided, otherwise create a new one
    client = init_bedrock_client(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        aws_region_name=aws_region_name,
        aws_bedrock_runtime_endpoint=aws_bedrock_runtime_endpoint,
        aws_web_identity_token=aws_web_identity_token,
        aws_role_name=aws_role_name,
        aws_session_name=aws_session_name,
        timeout=timeout,
    )

    ### FORMAT IMAGE GENERATION INPUT ###
    modelId = model
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

    body = json.dumps(data).encode("utf-8")
    ## LOGGING
    request_str = f"""
    response = client.invoke_model(
        body={body}, # type: ignore
        modelId={modelId},
        accept="application/json",
        contentType="application/json",
    )"""  # type: ignore
    logging_obj.pre_call(
        input=prompt,
        api_key="",  # boto3 is used for init.
        additional_args={
            "complete_input_dict": {"model": modelId, "texts": prompt},
            "request_str": request_str,
        },
    )
    try:
        response = client.invoke_model(
            body=body,
            modelId=modelId,
            accept="application/json",
            contentType="application/json",
        )
        response_body = json.loads(response.get("body").read())
        ## LOGGING
        logging_obj.post_call(
            input=prompt,
            api_key="",
            additional_args={"complete_input_dict": data},
            original_response=json.dumps(response_body),
        )
    except Exception as e:
        raise BedrockError(
            message=f"Embedding Error with model {model}: {e}", status_code=500
        )

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
