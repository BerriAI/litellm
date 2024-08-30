"""
Handles embedding calls to Bedrock's `/invoke` endpoint 
"""

import copy
import json
import os
from typing import Any, Optional, Union

import litellm
from litellm.types.utils import Usage

from .common_utils import BedrockError, init_bedrock_client


def _embedding_func_single(
    model: str,
    input: str,
    client: Any,
    optional_params=None,
    encoding=None,
    logging_obj=None,
):
    if isinstance(input, str) is False:
        raise BedrockError(
            message="Bedrock Embedding API input must be type str | List[str]",
            status_code=400,
        )
    # logic for parsing in - calling - parsing out model embedding calls
    ## FORMAT EMBEDDING INPUT ##
    provider = model.split(".")[0]
    inference_params = copy.deepcopy(optional_params)
    inference_params.pop(
        "user", None
    )  # make sure user is not passed in for bedrock call
    modelId = (
        optional_params.pop("model_id", None) or model
    )  # default to model if not passed
    if provider == "amazon":
        input = input.replace(os.linesep, " ")
        data = {"inputText": input, **inference_params}
        # data = json.dumps(data)
    elif provider == "cohere":
        inference_params["input_type"] = inference_params.get(
            "input_type", "search_document"
        )  # aws bedrock example default - https://us-east-1.console.aws.amazon.com/bedrock/home?region=us-east-1#/providers?model=cohere.embed-english-v3
        data = {"texts": [input], **inference_params}  # type: ignore
    body = json.dumps(data).encode("utf-8")  # type: ignore
    ## LOGGING
    request_str = f"""
    response = client.invoke_model(
        body={body},
        modelId={modelId},
        accept="*/*",
        contentType="application/json",
    )"""  # type: ignore
    logging_obj.pre_call(
        input=input,
        api_key="",  # boto3 is used for init.
        additional_args={
            "complete_input_dict": {"model": modelId, "texts": input},
            "request_str": request_str,
        },
    )
    try:
        response = client.invoke_model(
            body=body,
            modelId=modelId,
            accept="*/*",
            contentType="application/json",
        )
        response_body = json.loads(response.get("body").read())
        ## LOGGING
        logging_obj.post_call(
            input=input,
            api_key="",
            additional_args={"complete_input_dict": data},
            original_response=json.dumps(response_body),
        )
        if provider == "cohere":
            response = response_body.get("embeddings")
            # flatten list
            response = [item for sublist in response for item in sublist]
            return response
        elif provider == "amazon":
            return response_body.get("embedding")
    except Exception as e:
        raise BedrockError(
            message=f"Embedding Error with model {model}: {e}", status_code=500
        )


def embedding(
    model: str,
    input: Union[list, str],
    model_response: litellm.EmbeddingResponse,
    api_key: Optional[str] = None,
    logging_obj=None,
    optional_params=None,
    encoding=None,
):
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
    )
    if isinstance(input, str):
        ## Embedding Call
        embeddings = [
            _embedding_func_single(
                model,
                input,
                optional_params=optional_params,
                client=client,
                logging_obj=logging_obj,
            )
        ]
    elif isinstance(input, list):
        ## Embedding Call - assuming this is a List[str]
        embeddings = [
            _embedding_func_single(
                model,
                i,
                optional_params=optional_params,
                client=client,
                logging_obj=logging_obj,
            )
            for i in input
        ]  # [TODO]: make these parallel calls
    else:
        # enters this branch if input = int, ex. input=2
        raise BedrockError(
            message="Bedrock Embedding API input must be type str | List[str]",
            status_code=400,
        )

    ## Populate OpenAI compliant dictionary
    embedding_response = []
    for idx, embedding in enumerate(embeddings):
        embedding_response.append(
            {
                "object": "embedding",
                "index": idx,
                "embedding": embedding,
            }
        )
    model_response.object = "list"
    model_response.data = embedding_response
    model_response.model = model
    input_tokens = 0

    input_str = "".join(input)

    input_tokens += len(encoding.encode(input_str))

    usage = Usage(
        prompt_tokens=input_tokens, completion_tokens=0, total_tokens=input_tokens + 0
    )
    model_response.usage = usage

    return model_response
