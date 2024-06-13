import os, types, traceback
from enum import Enum
import json
import requests  # type: ignore
import time
from typing import Callable, Optional, Any
import litellm
from litellm.utils import ModelResponse, EmbeddingResponse, get_secret, Usage
import sys
from copy import deepcopy
import httpx  # type: ignore
from .prompt_templates.factory import prompt_factory, custom_prompt


class SagemakerError(Exception):
    def __init__(self, status_code, message):
        self.status_code = status_code
        self.message = message
        self.request = httpx.Request(
            method="POST", url="https://us-west-2.console.aws.amazon.com/sagemaker"
        )
        self.response = httpx.Response(status_code=status_code, request=self.request)
        super().__init__(
            self.message
        )  # Call the base class constructor with the parameters it needs


import io
import json


class TokenIterator:
    def __init__(self, stream, acompletion: bool = False):
        if acompletion == False:
            self.byte_iterator = iter(stream)
        elif acompletion == True:
            self.byte_iterator = stream
        self.buffer = io.BytesIO()
        self.read_pos = 0
        self.end_of_data = False

    def __iter__(self):
        return self

    def __next__(self):
        try:
            while True:
                self.buffer.seek(self.read_pos)
                line = self.buffer.readline()
                if line and line[-1] == ord("\n"):
                    response_obj = {"text": "", "is_finished": False}
                    self.read_pos += len(line) + 1
                    full_line = line[:-1].decode("utf-8")
                    line_data = json.loads(full_line.lstrip("data:").rstrip("/n"))
                    if line_data.get("generated_text", None) is not None:
                        self.end_of_data = True
                        response_obj["is_finished"] = True
                    response_obj["text"] = line_data["token"]["text"]
                    return response_obj
                chunk = next(self.byte_iterator)
                self.buffer.seek(0, io.SEEK_END)
                self.buffer.write(chunk["PayloadPart"]["Bytes"])
        except StopIteration as e:
            if self.end_of_data == True:
                raise e  # Re-raise StopIteration
            else:
                self.end_of_data = True
                return "data: [DONE]"

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            while True:
                self.buffer.seek(self.read_pos)
                line = self.buffer.readline()
                if line and line[-1] == ord("\n"):
                    response_obj = {"text": "", "is_finished": False}
                    self.read_pos += len(line) + 1
                    full_line = line[:-1].decode("utf-8")
                    line_data = json.loads(full_line.lstrip("data:").rstrip("/n"))
                    if line_data.get("generated_text", None) is not None:
                        self.end_of_data = True
                        response_obj["is_finished"] = True
                    response_obj["text"] = line_data["token"]["text"]
                    return response_obj
                chunk = await self.byte_iterator.__anext__()
                self.buffer.seek(0, io.SEEK_END)
                self.buffer.write(chunk["PayloadPart"]["Bytes"])
        except StopAsyncIteration as e:
            if self.end_of_data == True:
                raise e  # Re-raise StopIteration
            else:
                self.end_of_data = True
                return "data: [DONE]"


class SagemakerConfig:
    """
    Reference: https://d-uuwbxj1u4cnu.studio.us-west-2.sagemaker.aws/jupyter/default/lab/workspaces/auto-q/tree/DemoNotebooks/meta-textgeneration-llama-2-7b-SDK_1.ipynb
    """

    max_new_tokens: Optional[int] = None
    top_p: Optional[float] = None
    temperature: Optional[float] = None
    return_full_text: Optional[bool] = None

    def __init__(
        self,
        max_new_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        temperature: Optional[float] = None,
        return_full_text: Optional[bool] = None,
    ) -> None:
        locals_ = locals()
        for key, value in locals_.items():
            if key != "self" and value is not None:
                setattr(self.__class__, key, value)

    @classmethod
    def get_config(cls):
        return {
            k: v
            for k, v in cls.__dict__.items()
            if not k.startswith("__")
            and not isinstance(
                v,
                (
                    types.FunctionType,
                    types.BuiltinFunctionType,
                    classmethod,
                    staticmethod,
                ),
            )
            and v is not None
        }


"""
SAGEMAKER AUTH Keys/Vars
os.environ['AWS_ACCESS_KEY_ID'] = ""
os.environ['AWS_SECRET_ACCESS_KEY'] = ""
"""

# set os.environ['AWS_REGION_NAME'] = <your-region_name>


def completion(
    model: str,
    messages: list,
    model_response: ModelResponse,
    print_verbose: Callable,
    encoding,
    logging_obj,
    custom_prompt_dict={},
    hf_model_name=None,
    optional_params=None,
    litellm_params=None,
    logger_fn=None,
    acompletion: bool = False,
):
    import boto3

    # pop aws_secret_access_key, aws_access_key_id, aws_region_name from kwargs, since completion calls fail with them
    aws_secret_access_key = optional_params.pop("aws_secret_access_key", None)
    aws_access_key_id = optional_params.pop("aws_access_key_id", None)
    aws_region_name = optional_params.pop("aws_region_name", None)
    model_id = optional_params.pop("model_id", None)

    if aws_access_key_id != None:
        # uses auth params passed to completion
        # aws_access_key_id is not None, assume user is trying to auth using litellm.completion
        client = boto3.client(
            service_name="sagemaker-runtime",
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=aws_region_name,
        )
    else:
        # aws_access_key_id is None, assume user is trying to auth using env variables
        # boto3 automaticaly reads env variables

        # we need to read region name from env
        # I assume majority of users use .env for auth
        region_name = (
            get_secret("AWS_REGION_NAME")
            or "us-west-2"  # default to us-west-2 if user not specified
        )
        client = boto3.client(
            service_name="sagemaker-runtime",
            region_name=region_name,
        )

    # pop streaming if it's in the optional params as 'stream' raises an error with sagemaker
    inference_params = deepcopy(optional_params)

    ## Load Config
    config = litellm.SagemakerConfig.get_config()
    for k, v in config.items():
        if (
            k not in inference_params
        ):  # completion(top_k=3) > sagemaker_config(top_k=3) <- allows for dynamic variables to be passed in
            inference_params[k] = v

    model = model
    if model in custom_prompt_dict:
        # check if the model has a registered custom prompt
        model_prompt_details = custom_prompt_dict[model]
        prompt = custom_prompt(
            role_dict=model_prompt_details.get("roles", None),
            initial_prompt_value=model_prompt_details.get("initial_prompt_value", ""),
            final_prompt_value=model_prompt_details.get("final_prompt_value", ""),
            messages=messages,
        )
    elif hf_model_name in custom_prompt_dict:
        # check if the base huggingface model has a registered custom prompt
        model_prompt_details = custom_prompt_dict[hf_model_name]
        prompt = custom_prompt(
            role_dict=model_prompt_details.get("roles", None),
            initial_prompt_value=model_prompt_details.get("initial_prompt_value", ""),
            final_prompt_value=model_prompt_details.get("final_prompt_value", ""),
            messages=messages,
        )
    else:
        if hf_model_name is None:
            if "llama-2" in model.lower():  # llama-2 model
                if "chat" in model.lower():  # apply llama2 chat template
                    hf_model_name = "meta-llama/Llama-2-7b-chat-hf"
                else:  # apply regular llama2 template
                    hf_model_name = "meta-llama/Llama-2-7b"
        hf_model_name = (
            hf_model_name or model
        )  # pass in hf model name for pulling it's prompt template - (e.g. `hf_model_name="meta-llama/Llama-2-7b-chat-hf` applies the llama2 chat template to the prompt)
        prompt = prompt_factory(model=hf_model_name, messages=messages)
    stream = inference_params.pop("stream", None)
    if stream == True:
        data = json.dumps(
            {"inputs": prompt, "parameters": inference_params, "stream": True}
        ).encode("utf-8")
        if acompletion == True:
            response = async_streaming(
                optional_params=optional_params,
                encoding=encoding,
                model_response=model_response,
                model=model,
                logging_obj=logging_obj,
                data=data,
                model_id=model_id,
                aws_secret_access_key=aws_secret_access_key,
                aws_access_key_id=aws_access_key_id,
                aws_region_name=aws_region_name,
            )
            return response

        if model_id is not None:
            response = client.invoke_endpoint_with_response_stream(
                EndpointName=model,
                InferenceComponentName=model_id,
                ContentType="application/json",
                Body=data,
                CustomAttributes="accept_eula=true",
            )
        else:
            response = client.invoke_endpoint_with_response_stream(
                EndpointName=model,
                ContentType="application/json",
                Body=data,
                CustomAttributes="accept_eula=true",
            )
        return response["Body"]
    elif acompletion == True:
        _data = {"inputs": prompt, "parameters": inference_params}
        return async_completion(
            optional_params=optional_params,
            encoding=encoding,
            model_response=model_response,
            model=model,
            logging_obj=logging_obj,
            data=_data,
            model_id=model_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_access_key_id=aws_access_key_id,
            aws_region_name=aws_region_name,
        )
    data = json.dumps({"inputs": prompt, "parameters": inference_params}).encode(
        "utf-8"
    )
    ## COMPLETION CALL
    try:
        if model_id is not None:
            ## LOGGING
            request_str = f"""
            response = client.invoke_endpoint(
                EndpointName={model},
                InferenceComponentName={model_id},
                ContentType="application/json",
                Body={data}, # type: ignore
                CustomAttributes="accept_eula=true",
            )
            """  # type: ignore
            logging_obj.pre_call(
                input=prompt,
                api_key="",
                additional_args={
                    "complete_input_dict": data,
                    "request_str": request_str,
                    "hf_model_name": hf_model_name,
                },
            )
            response = client.invoke_endpoint(
                EndpointName=model,
                InferenceComponentName=model_id,
                ContentType="application/json",
                Body=data,
                CustomAttributes="accept_eula=true",
            )
        else:
            ## LOGGING
            request_str = f"""
            response = client.invoke_endpoint(
                EndpointName={model},
                ContentType="application/json",
                Body={data}, # type: ignore
                CustomAttributes="accept_eula=true",
            )
            """  # type: ignore
            logging_obj.pre_call(
                input=prompt,
                api_key="",
                additional_args={
                    "complete_input_dict": data,
                    "request_str": request_str,
                    "hf_model_name": hf_model_name,
                },
            )
            response = client.invoke_endpoint(
                EndpointName=model,
                ContentType="application/json",
                Body=data,
                CustomAttributes="accept_eula=true",
            )
    except Exception as e:
        status_code = (
            getattr(e, "response", {})
            .get("ResponseMetadata", {})
            .get("HTTPStatusCode", 500)
        )
        error_message = (
            getattr(e, "response", {}).get("Error", {}).get("Message", str(e))
        )
        if "Inference Component Name header is required" in error_message:
            error_message += "\n pass in via `litellm.completion(..., model_id={InferenceComponentName})`"
        raise SagemakerError(status_code=status_code, message=error_message)

    response = response["Body"].read().decode("utf8")
    ## LOGGING
    logging_obj.post_call(
        input=prompt,
        api_key="",
        original_response=response,
        additional_args={"complete_input_dict": data},
    )
    print_verbose(f"raw model_response: {response}")
    ## RESPONSE OBJECT
    completion_response = json.loads(response)
    try:
        if isinstance(completion_response, list):
            completion_response_choices = completion_response[0]
        else:
            completion_response_choices = completion_response
        completion_output = ""
        if "generation" in completion_response_choices:
            completion_output += completion_response_choices["generation"]
        elif "generated_text" in completion_response_choices:
            completion_output += completion_response_choices["generated_text"]

        # check if the prompt template is part of output, if so - filter it out
        if completion_output.startswith(prompt) and "<s>" in prompt:
            completion_output = completion_output.replace(prompt, "", 1)

        model_response["choices"][0]["message"]["content"] = completion_output
    except:
        raise SagemakerError(
            message=f"LiteLLM Error: Unable to parse sagemaker RAW RESPONSE {json.dumps(completion_response)}",
            status_code=500,
        )

    ## CALCULATING USAGE - baseten charges on time, not tokens - have some mapping of cost here.
    prompt_tokens = len(encoding.encode(prompt))
    completion_tokens = len(
        encoding.encode(model_response["choices"][0]["message"].get("content", ""))
    )

    model_response["created"] = int(time.time())
    model_response["model"] = model
    usage = Usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )
    setattr(model_response, "usage", usage)
    return model_response


async def async_streaming(
    optional_params,
    encoding,
    model_response: ModelResponse,
    model: str,
    model_id: Optional[str],
    logging_obj: Any,
    data,
    aws_secret_access_key: Optional[str],
    aws_access_key_id: Optional[str],
    aws_region_name: Optional[str],
):
    """
    Use aioboto3
    """
    import aioboto3

    session = aioboto3.Session()

    if aws_access_key_id != None:
        # uses auth params passed to completion
        # aws_access_key_id is not None, assume user is trying to auth using litellm.completion
        _client = session.client(
            service_name="sagemaker-runtime",
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=aws_region_name,
        )
    else:
        # aws_access_key_id is None, assume user is trying to auth using env variables
        # boto3 automaticaly reads env variables

        # we need to read region name from env
        # I assume majority of users use .env for auth
        region_name = (
            get_secret("AWS_REGION_NAME")
            or "us-west-2"  # default to us-west-2 if user not specified
        )
        _client = session.client(
            service_name="sagemaker-runtime",
            region_name=region_name,
        )

    async with _client as client:
        try:
            if model_id is not None:
                response = await client.invoke_endpoint_with_response_stream(
                    EndpointName=model,
                    InferenceComponentName=model_id,
                    ContentType="application/json",
                    Body=data,
                    CustomAttributes="accept_eula=true",
                )
            else:
                response = await client.invoke_endpoint_with_response_stream(
                    EndpointName=model,
                    ContentType="application/json",
                    Body=data,
                    CustomAttributes="accept_eula=true",
                )
        except Exception as e:
            raise SagemakerError(status_code=500, message=f"{str(e)}")
        response = response["Body"]
        async for chunk in response:
            yield chunk


async def async_completion(
    optional_params,
    encoding,
    model_response: ModelResponse,
    model: str,
    logging_obj: Any,
    data: dict,
    model_id: Optional[str],
    aws_secret_access_key: Optional[str],
    aws_access_key_id: Optional[str],
    aws_region_name: Optional[str],
):
    """
    Use aioboto3
    """
    import aioboto3

    session = aioboto3.Session()

    if aws_access_key_id != None:
        # uses auth params passed to completion
        # aws_access_key_id is not None, assume user is trying to auth using litellm.completion
        _client = session.client(
            service_name="sagemaker-runtime",
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=aws_region_name,
        )
    else:
        # aws_access_key_id is None, assume user is trying to auth using env variables
        # boto3 automaticaly reads env variables

        # we need to read region name from env
        # I assume majority of users use .env for auth
        region_name = (
            get_secret("AWS_REGION_NAME")
            or "us-west-2"  # default to us-west-2 if user not specified
        )
        _client = session.client(
            service_name="sagemaker-runtime",
            region_name=region_name,
        )

    async with _client as client:
        encoded_data = json.dumps(data).encode("utf-8")
        try:
            if model_id is not None:
                ## LOGGING
                request_str = f"""
                response = client.invoke_endpoint(
                    EndpointName={model},
                    InferenceComponentName={model_id},
                    ContentType="application/json",
                    Body={data},
                    CustomAttributes="accept_eula=true",
                )
                """  # type: ignore
                logging_obj.pre_call(
                    input=data["inputs"],
                    api_key="",
                    additional_args={
                        "complete_input_dict": data,
                        "request_str": request_str,
                    },
                )
                response = await client.invoke_endpoint(
                    EndpointName=model,
                    InferenceComponentName=model_id,
                    ContentType="application/json",
                    Body=encoded_data,
                    CustomAttributes="accept_eula=true",
                )
            else:
                ## LOGGING
                request_str = f"""
                response = client.invoke_endpoint(
                    EndpointName={model},
                    ContentType="application/json",
                    Body={data},
                    CustomAttributes="accept_eula=true",
                )
                """  # type: ignore
                logging_obj.pre_call(
                    input=data["inputs"],
                    api_key="",
                    additional_args={
                        "complete_input_dict": data,
                        "request_str": request_str,
                    },
                )
                response = await client.invoke_endpoint(
                    EndpointName=model,
                    ContentType="application/json",
                    Body=encoded_data,
                    CustomAttributes="accept_eula=true",
                )
        except Exception as e:
            error_message = f"{str(e)}"
            if "Inference Component Name header is required" in error_message:
                error_message += "\n pass in via `litellm.completion(..., model_id={InferenceComponentName})`"
            raise SagemakerError(status_code=500, message=error_message)
        response = await response["Body"].read()
        response = response.decode("utf8")
        ## LOGGING
        logging_obj.post_call(
            input=data["inputs"],
            api_key="",
            original_response=response,
            additional_args={"complete_input_dict": data},
        )
        ## RESPONSE OBJECT
        completion_response = json.loads(response)
        try:
            if isinstance(completion_response, list):
                completion_response_choices = completion_response[0]
            else:
                completion_response_choices = completion_response
            completion_output = ""
            if "generation" in completion_response_choices:
                completion_output += completion_response_choices["generation"]
            elif "generated_text" in completion_response_choices:
                completion_output += completion_response_choices["generated_text"]

            # check if the prompt template is part of output, if so - filter it out
            if completion_output.startswith(data["inputs"]) and "<s>" in data["inputs"]:
                completion_output = completion_output.replace(data["inputs"], "", 1)

            model_response["choices"][0]["message"]["content"] = completion_output
        except:
            raise SagemakerError(
                message=f"LiteLLM Error: Unable to parse sagemaker RAW RESPONSE {json.dumps(completion_response)}",
                status_code=500,
            )

        ## CALCULATING USAGE - baseten charges on time, not tokens - have some mapping of cost here.
        prompt_tokens = len(encoding.encode(data["inputs"]))
        completion_tokens = len(
            encoding.encode(model_response["choices"][0]["message"].get("content", ""))
        )

        model_response["created"] = int(time.time())
        model_response["model"] = model
        usage = Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )
        setattr(model_response, "usage", usage)
        return model_response


def embedding(
    model: str,
    input: list,
    model_response: EmbeddingResponse,
    print_verbose: Callable,
    encoding,
    logging_obj,
    custom_prompt_dict={},
    optional_params=None,
    litellm_params=None,
    logger_fn=None,
):
    """
    Supports Huggingface Jumpstart embeddings like GPT-6B
    """
    ### BOTO3 INIT
    import boto3

    # pop aws_secret_access_key, aws_access_key_id, aws_region_name from kwargs, since completion calls fail with them
    aws_secret_access_key = optional_params.pop("aws_secret_access_key", None)
    aws_access_key_id = optional_params.pop("aws_access_key_id", None)
    aws_region_name = optional_params.pop("aws_region_name", None)

    if aws_access_key_id is not None:
        # uses auth params passed to completion
        # aws_access_key_id is not None, assume user is trying to auth using litellm.completion
        client = boto3.client(
            service_name="sagemaker-runtime",
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=aws_region_name,
        )
    else:
        # aws_access_key_id is None, assume user is trying to auth using env variables
        # boto3 automaticaly reads env variables

        # we need to read region name from env
        # I assume majority of users use .env for auth
        region_name = (
            get_secret("AWS_REGION_NAME")
            or "us-west-2"  # default to us-west-2 if user not specified
        )
        client = boto3.client(
            service_name="sagemaker-runtime",
            region_name=region_name,
        )

    # pop streaming if it's in the optional params as 'stream' raises an error with sagemaker
    inference_params = deepcopy(optional_params)
    inference_params.pop("stream", None)

    ## Load Config
    config = litellm.SagemakerConfig.get_config()
    for k, v in config.items():
        if (
            k not in inference_params
        ):  # completion(top_k=3) > sagemaker_config(top_k=3) <- allows for dynamic variables to be passed in
            inference_params[k] = v

    #### HF EMBEDDING LOGIC
    data = json.dumps({"text_inputs": input}).encode("utf-8")

    ## LOGGING
    request_str = f"""
    response = client.invoke_endpoint(
        EndpointName={model},
        ContentType="application/json",
        Body={data}, # type: ignore
        CustomAttributes="accept_eula=true",
    )"""  # type: ignore
    logging_obj.pre_call(
        input=input,
        api_key="",
        additional_args={"complete_input_dict": data, "request_str": request_str},
    )
    ## EMBEDDING CALL
    try:
        response = client.invoke_endpoint(
            EndpointName=model,
            ContentType="application/json",
            Body=data,
            CustomAttributes="accept_eula=true",
        )
    except Exception as e:
        status_code = (
            getattr(e, "response", {})
            .get("ResponseMetadata", {})
            .get("HTTPStatusCode", 500)
        )
        error_message = (
            getattr(e, "response", {}).get("Error", {}).get("Message", str(e))
        )
        raise SagemakerError(status_code=status_code, message=error_message)

    response = json.loads(response["Body"].read().decode("utf8"))
    ## LOGGING
    logging_obj.post_call(
        input=input,
        api_key="",
        original_response=response,
        additional_args={"complete_input_dict": data},
    )

    print_verbose(f"raw model_response: {response}")
    if "embedding" not in response:
        raise SagemakerError(status_code=500, message="embedding not found in response")
    embeddings = response["embedding"]

    if not isinstance(embeddings, list):
        raise SagemakerError(
            status_code=422, message=f"Response not in expected format - {embeddings}"
        )

    output_data = []
    for idx, embedding in enumerate(embeddings):
        output_data.append(
            {"object": "embedding", "index": idx, "embedding": embedding}
        )

    model_response["object"] = "list"
    model_response["data"] = output_data
    model_response["model"] = model

    input_tokens = 0
    for text in input:
        input_tokens += len(encoding.encode(text))

    model_response["usage"] = Usage(
        prompt_tokens=input_tokens, completion_tokens=0, total_tokens=input_tokens
    )

    return model_response
