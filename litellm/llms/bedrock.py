import json, copy, types
import os
from enum import Enum
import time
from typing import Callable, Optional, Any, Union, List
import litellm
from litellm.utils import ModelResponse, get_secret, Usage, ImageResponse
from .prompt_templates.factory import prompt_factory, custom_prompt
import httpx


class BedrockError(Exception):
    def __init__(self, status_code, message):
        self.status_code = status_code
        self.message = message
        self.request = httpx.Request(
            method="POST", url="https://us-west-2.console.aws.amazon.com/bedrock"
        )
        self.response = httpx.Response(status_code=status_code, request=self.request)
        super().__init__(
            self.message
        )  # Call the base class constructor with the parameters it needs


class AmazonTitanConfig:
    """
    Reference: https://us-west-2.console.aws.amazon.com/bedrock/home?region=us-west-2#/providers?model=titan-text-express-v1

    Supported Params for the Amazon Titan models:

    - `maxTokenCount` (integer) max tokens,
    - `stopSequences` (string[]) list of stop sequence strings
    - `temperature` (float) temperature for model,
    - `topP` (int) top p for model
    """

    maxTokenCount: Optional[int] = None
    stopSequences: Optional[list] = None
    temperature: Optional[float] = None
    topP: Optional[int] = None

    def __init__(
        self,
        maxTokenCount: Optional[int] = None,
        stopSequences: Optional[list] = None,
        temperature: Optional[float] = None,
        topP: Optional[int] = None,
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


class AmazonAnthropicConfig:
    """
    Reference: https://us-west-2.console.aws.amazon.com/bedrock/home?region=us-west-2#/providers?model=claude

    Supported Params for the Amazon / Anthropic models:

    - `max_tokens_to_sample` (integer) max tokens,
    - `temperature` (float) model temperature,
    - `top_k` (integer) top k,
    - `top_p` (integer) top p,
    - `stop_sequences` (string[]) list of stop sequences - e.g. ["\\n\\nHuman:"],
    - `anthropic_version` (string) version of anthropic for bedrock - e.g. "bedrock-2023-05-31"
    """

    max_tokens_to_sample: Optional[int] = litellm.max_tokens
    stop_sequences: Optional[list] = None
    temperature: Optional[float] = None
    top_k: Optional[int] = None
    top_p: Optional[int] = None
    anthropic_version: Optional[str] = None

    def __init__(
        self,
        max_tokens_to_sample: Optional[int] = None,
        stop_sequences: Optional[list] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[int] = None,
        anthropic_version: Optional[str] = None,
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


class AmazonCohereConfig:
    """
    Reference: https://us-west-2.console.aws.amazon.com/bedrock/home?region=us-west-2#/providers?model=command

    Supported Params for the Amazon / Cohere models:

    - `max_tokens` (integer) max tokens,
    - `temperature` (float) model temperature,
    - `return_likelihood` (string) n/a
    """

    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    return_likelihood: Optional[str] = None

    def __init__(
        self,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        return_likelihood: Optional[str] = None,
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


class AmazonAI21Config:
    """
    Reference: https://us-west-2.console.aws.amazon.com/bedrock/home?region=us-west-2#/providers?model=j2-ultra

    Supported Params for the Amazon / AI21 models:

    - `maxTokens` (int32): The maximum number of tokens to generate per result. Optional, default is 16. If no `stopSequences` are given, generation stops after producing `maxTokens`.

    - `temperature` (float): Modifies the distribution from which tokens are sampled. Optional, default is 0.7. A value of 0 essentially disables sampling and results in greedy decoding.

    - `topP` (float): Used for sampling tokens from the corresponding top percentile of probability mass. Optional, default is 1. For instance, a value of 0.9 considers only tokens comprising the top 90% probability mass.

    - `stopSequences` (array of strings): Stops decoding if any of the input strings is generated. Optional.

    - `frequencyPenalty` (object): Placeholder for frequency penalty object.

    - `presencePenalty` (object): Placeholder for presence penalty object.

    - `countPenalty` (object): Placeholder for count penalty object.
    """

    maxTokens: Optional[int] = None
    temperature: Optional[float] = None
    topP: Optional[float] = None
    stopSequences: Optional[list] = None
    frequencePenalty: Optional[dict] = None
    presencePenalty: Optional[dict] = None
    countPenalty: Optional[dict] = None

    def __init__(
        self,
        maxTokens: Optional[int] = None,
        temperature: Optional[float] = None,
        topP: Optional[float] = None,
        stopSequences: Optional[list] = None,
        frequencePenalty: Optional[dict] = None,
        presencePenalty: Optional[dict] = None,
        countPenalty: Optional[dict] = None,
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


class AnthropicConstants(Enum):
    HUMAN_PROMPT = "\n\nHuman: "
    AI_PROMPT = "\n\nAssistant: "


class AmazonLlamaConfig:
    """
    Reference: https://us-west-2.console.aws.amazon.com/bedrock/home?region=us-west-2#/providers?model=meta.llama2-13b-chat-v1

    Supported Params for the Amazon / Meta Llama models:

    - `max_gen_len` (integer) max tokens,
    - `temperature` (float) temperature for model,
    - `top_p` (float) top p for model
    """

    max_gen_len: Optional[int] = None
    temperature: Optional[float] = None
    topP: Optional[float] = None

    def __init__(
        self,
        maxTokenCount: Optional[int] = None,
        temperature: Optional[float] = None,
        topP: Optional[int] = None,
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


class AmazonStabilityConfig:
    """
    Reference: https://us-west-2.console.aws.amazon.com/bedrock/home?region=us-west-2#/providers?model=stability.stable-diffusion-xl-v0

    Supported Params for the Amazon / Stable Diffusion models:

    - `cfg_scale` (integer): Default `7`. Between [ 0 .. 35 ]. How strictly the diffusion process adheres to the prompt text (higher values keep your image closer to your prompt)

    - `seed` (float): Default: `0`. Between [ 0 .. 4294967295 ]. Random noise seed (omit this option or use 0 for a random seed)

    - `steps` (array of strings): Default `30`. Between [ 10 .. 50 ]. Number of diffusion steps to run.

    - `width` (integer): Default: `512`. multiple of 64 >= 128. Width of the image to generate, in pixels, in an increment divible by 64.
        Engine-specific dimension validation:

        - SDXL Beta: must be between 128x128 and 512x896 (or 896x512); only one dimension can be greater than 512.
        - SDXL v0.9: must be one of 1024x1024, 1152x896, 1216x832, 1344x768, 1536x640, 640x1536, 768x1344, 832x1216, or 896x1152
        - SDXL v1.0: same as SDXL v0.9
        - SD v1.6: must be between 320x320 and 1536x1536

    - `height` (integer): Default: `512`. multiple of 64 >= 128. Height of the image to generate, in pixels, in an increment divible by 64.
        Engine-specific dimension validation:

        - SDXL Beta: must be between 128x128 and 512x896 (or 896x512); only one dimension can be greater than 512.
        - SDXL v0.9: must be one of 1024x1024, 1152x896, 1216x832, 1344x768, 1536x640, 640x1536, 768x1344, 832x1216, or 896x1152
        - SDXL v1.0: same as SDXL v0.9
        - SD v1.6: must be between 320x320 and 1536x1536
    """

    cfg_scale: Optional[int] = None
    seed: Optional[float] = None
    steps: Optional[List[str]] = None
    width: Optional[int] = None
    height: Optional[int] = None

    def __init__(
        self,
        cfg_scale: Optional[int] = None,
        seed: Optional[float] = None,
        steps: Optional[List[str]] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
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


def init_bedrock_client(
    region_name=None,
    aws_access_key_id: Optional[str] = None,
    aws_secret_access_key: Optional[str] = None,
    aws_region_name: Optional[str] = None,
    aws_bedrock_runtime_endpoint: Optional[str] = None,
    aws_session_name: Optional[str] = None,
    aws_profile_name: Optional[str] = None,
    aws_role_name: Optional[str] = None,
    timeout: Optional[int] = None,
):
    # check for custom AWS_REGION_NAME and use it if not passed to init_bedrock_client
    litellm_aws_region_name = get_secret("AWS_REGION_NAME", None)
    standard_aws_region_name = get_secret("AWS_REGION", None)

    ## CHECK IS  'os.environ/' passed in
    # Define the list of parameters to check
    params_to_check = [
        aws_access_key_id,
        aws_secret_access_key,
        aws_region_name,
        aws_bedrock_runtime_endpoint,
        aws_session_name,
        aws_profile_name,
        aws_role_name,
    ]

    # Iterate over parameters and update if needed
    for i, param in enumerate(params_to_check):
        if param and param.startswith("os.environ/"):
            params_to_check[i] = get_secret(param)
    # Assign updated values back to parameters
    (
        aws_access_key_id,
        aws_secret_access_key,
        aws_region_name,
        aws_bedrock_runtime_endpoint,
        aws_session_name,
        aws_profile_name,
        aws_role_name,
    ) = params_to_check

    ### SET REGION NAME
    if region_name:
        pass
    elif aws_region_name:
        region_name = aws_region_name
    elif litellm_aws_region_name:
        region_name = litellm_aws_region_name
    elif standard_aws_region_name:
        region_name = standard_aws_region_name
    else:
        raise BedrockError(
            message="AWS region not set: set AWS_REGION_NAME or AWS_REGION env variable or in .env file",
            status_code=401,
        )

    # check for custom AWS_BEDROCK_RUNTIME_ENDPOINT and use it if not passed to init_bedrock_client
    env_aws_bedrock_runtime_endpoint = get_secret("AWS_BEDROCK_RUNTIME_ENDPOINT")
    if aws_bedrock_runtime_endpoint:
        endpoint_url = aws_bedrock_runtime_endpoint
    elif env_aws_bedrock_runtime_endpoint:
        endpoint_url = env_aws_bedrock_runtime_endpoint
    else:
        endpoint_url = f"https://bedrock-runtime.{region_name}.amazonaws.com"

    import boto3

    config = boto3.session.Config(connect_timeout=timeout, read_timeout=timeout)

    ### CHECK STS ###
    if aws_role_name is not None and aws_session_name is not None:
        # use sts if role name passed in
        sts_client = boto3.client(
            "sts",
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
        )

        sts_response = sts_client.assume_role(
            RoleArn=aws_role_name, RoleSessionName=aws_session_name
        )

        client = boto3.client(
            service_name="bedrock-runtime",
            aws_access_key_id=sts_response["Credentials"]["AccessKeyId"],
            aws_secret_access_key=sts_response["Credentials"]["SecretAccessKey"],
            aws_session_token=sts_response["Credentials"]["SessionToken"],
            region_name=region_name,
            endpoint_url=endpoint_url,
            config=config,
        )
    elif aws_access_key_id is not None:
        # uses auth params passed to completion
        # aws_access_key_id is not None, assume user is trying to auth using litellm.completion

        client = boto3.client(
            service_name="bedrock-runtime",
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=region_name,
            endpoint_url=endpoint_url,
            config=config,
        )
    elif aws_profile_name is not None:
        # uses auth values from AWS profile usually stored in ~/.aws/credentials

        client = boto3.Session(profile_name=aws_profile_name).client(
            service_name="bedrock-runtime",
            region_name=region_name,
            endpoint_url=endpoint_url,
            config=config,
        )
    else:
        # aws_access_key_id is None, assume user is trying to auth using env variables
        # boto3 automatically reads env variables

        client = boto3.client(
            service_name="bedrock-runtime",
            region_name=region_name,
            endpoint_url=endpoint_url,
            config=config,
        )

    return client


def convert_messages_to_prompt(model, messages, provider, custom_prompt_dict):
    # handle anthropic prompts and amazon titan prompts
    if provider == "anthropic" or provider == "amazon":
        if model in custom_prompt_dict:
            # check if the model has a registered custom prompt
            model_prompt_details = custom_prompt_dict[model]
            prompt = custom_prompt(
                role_dict=model_prompt_details["roles"],
                initial_prompt_value=model_prompt_details["initial_prompt_value"],
                final_prompt_value=model_prompt_details["final_prompt_value"],
                messages=messages,
            )
        else:
            prompt = prompt_factory(
                model=model, messages=messages, custom_llm_provider="bedrock"
            )
    else:
        prompt = ""
        for message in messages:
            if "role" in message:
                if message["role"] == "user":
                    prompt += f"{message['content']}"
                else:
                    prompt += f"{message['content']}"
            else:
                prompt += f"{message['content']}"
    return prompt


"""
BEDROCK AUTH Keys/Vars
os.environ['AWS_ACCESS_KEY_ID'] = ""
os.environ['AWS_SECRET_ACCESS_KEY'] = ""
"""


# set os.environ['AWS_REGION_NAME'] = <your-region_name>


def completion(
    model: str,
    messages: list,
    custom_prompt_dict: dict,
    model_response: ModelResponse,
    print_verbose: Callable,
    encoding,
    logging_obj,
    optional_params=None,
    litellm_params=None,
    logger_fn=None,
    timeout=None,
):
    exception_mapping_worked = False
    try:
        # pop aws_secret_access_key, aws_access_key_id, aws_region_name from kwargs, since completion calls fail with them
        aws_secret_access_key = optional_params.pop("aws_secret_access_key", None)
        aws_access_key_id = optional_params.pop("aws_access_key_id", None)
        aws_region_name = optional_params.pop("aws_region_name", None)
        aws_role_name = optional_params.pop("aws_role_name", None)
        aws_session_name = optional_params.pop("aws_session_name", None)
        aws_profile_name = optional_params.pop("aws_profile_name", None)
        aws_bedrock_runtime_endpoint = optional_params.pop(
            "aws_bedrock_runtime_endpoint", None
        )

        # use passed in BedrockRuntime.Client if provided, otherwise create a new one
        client = optional_params.pop("aws_bedrock_client", None)

        # only init client, if user did not pass one
        if client is None:
            client = init_bedrock_client(
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                aws_region_name=aws_region_name,
                aws_bedrock_runtime_endpoint=aws_bedrock_runtime_endpoint,
                aws_role_name=aws_role_name,
                aws_session_name=aws_session_name,
                aws_profile_name=aws_profile_name,
                timeout=timeout,
            )

        model = model
        modelId = (
            optional_params.pop("model_id", None) or model
        )  # default to model if not passed
        provider = model.split(".")[0]
        prompt = convert_messages_to_prompt(
            model, messages, provider, custom_prompt_dict
        )
        inference_params = copy.deepcopy(optional_params)
        stream = inference_params.pop("stream", False)
        if provider == "anthropic":
            ## LOAD CONFIG
            config = litellm.AmazonAnthropicConfig.get_config()
            for k, v in config.items():
                if (
                    k not in inference_params
                ):  # completion(top_k=3) > anthropic_config(top_k=3) <- allows for dynamic variables to be passed in
                    inference_params[k] = v
            data = json.dumps({"prompt": prompt, **inference_params})
        elif provider == "ai21":
            ## LOAD CONFIG
            config = litellm.AmazonAI21Config.get_config()
            for k, v in config.items():
                if (
                    k not in inference_params
                ):  # completion(top_k=3) > anthropic_config(top_k=3) <- allows for dynamic variables to be passed in
                    inference_params[k] = v

            data = json.dumps({"prompt": prompt, **inference_params})
        elif provider == "cohere":
            ## LOAD CONFIG
            config = litellm.AmazonCohereConfig.get_config()
            for k, v in config.items():
                if (
                    k not in inference_params
                ):  # completion(top_k=3) > anthropic_config(top_k=3) <- allows for dynamic variables to be passed in
                    inference_params[k] = v
            if optional_params.get("stream", False) == True:
                inference_params[
                    "stream"
                ] = True  # cohere requires stream = True in inference params
            data = json.dumps({"prompt": prompt, **inference_params})
        elif provider == "meta":
            ## LOAD CONFIG
            config = litellm.AmazonLlamaConfig.get_config()
            for k, v in config.items():
                if (
                    k not in inference_params
                ):  # completion(top_k=3) > anthropic_config(top_k=3) <- allows for dynamic variables to be passed in
                    inference_params[k] = v
            data = json.dumps({"prompt": prompt, **inference_params})
        elif provider == "amazon":  # amazon titan
            ## LOAD CONFIG
            config = litellm.AmazonTitanConfig.get_config()
            for k, v in config.items():
                if (
                    k not in inference_params
                ):  # completion(top_k=3) > amazon_config(top_k=3) <- allows for dynamic variables to be passed in
                    inference_params[k] = v

            data = json.dumps(
                {
                    "inputText": prompt,
                    "textGenerationConfig": inference_params,
                }
            )

        else:
            data = json.dumps({})

        ## COMPLETION CALL
        accept = "application/json"
        contentType = "application/json"
        if stream == True:
            if provider == "ai21":
                ## LOGGING
                request_str = f"""
                response = client.invoke_model(
                    body={data},
                    modelId={modelId},
                    accept=accept,
                    contentType=contentType
                )
                """
                logging_obj.pre_call(
                    input=prompt,
                    api_key="",
                    additional_args={
                        "complete_input_dict": data,
                        "request_str": request_str,
                    },
                )

                response = client.invoke_model(
                    body=data, modelId=modelId, accept=accept, contentType=contentType
                )

                response = response.get("body").read()
                return response
            else:
                ## LOGGING
                request_str = f"""
                response = client.invoke_model_with_response_stream(
                    body={data},
                    modelId={modelId},
                    accept=accept,
                    contentType=contentType
                )
                """
                logging_obj.pre_call(
                    input=prompt,
                    api_key="",
                    additional_args={
                        "complete_input_dict": data,
                        "request_str": request_str,
                    },
                )

                response = client.invoke_model_with_response_stream(
                    body=data, modelId=modelId, accept=accept, contentType=contentType
                )
                response = response.get("body")
                return response
        try:
            ## LOGGING
            request_str = f"""
            response = client.invoke_model(
                body={data},
                modelId={modelId},
                accept=accept,
                contentType=contentType
            )
            """
            logging_obj.pre_call(
                input=prompt,
                api_key="",
                additional_args={
                    "complete_input_dict": data,
                    "request_str": request_str,
                },
            )
            response = client.invoke_model(
                body=data, modelId=modelId, accept=accept, contentType=contentType
            )
        except client.exceptions.ValidationException as e:
            if "The provided model identifier is invalid" in str(e):
                raise BedrockError(status_code=404, message=str(e))
            raise BedrockError(status_code=400, message=str(e))
        except Exception as e:
            raise BedrockError(status_code=500, message=str(e))

        response_body = json.loads(response.get("body").read())

        ## LOGGING
        logging_obj.post_call(
            input=prompt,
            api_key="",
            original_response=json.dumps(response_body),
            additional_args={"complete_input_dict": data},
        )
        print_verbose(f"raw model_response: {response}")
        ## RESPONSE OBJECT
        outputText = "default"
        if provider == "ai21":
            outputText = response_body.get("completions")[0].get("data").get("text")
        elif provider == "anthropic":
            outputText = response_body["completion"]
            model_response["finish_reason"] = response_body["stop_reason"]
        elif provider == "cohere":
            outputText = response_body["generations"][0]["text"]
        elif provider == "meta":
            outputText = response_body["generation"]
        else:  # amazon titan
            outputText = response_body.get("results")[0].get("outputText")

        response_metadata = response.get("ResponseMetadata", {})
        if response_metadata.get("HTTPStatusCode", 500) >= 400:
            raise BedrockError(
                message=outputText,
                status_code=response_metadata.get("HTTPStatusCode", 500),
            )
        else:
            try:
                if len(outputText) > 0:
                    model_response["choices"][0]["message"]["content"] = outputText
                else:
                    raise Exception()
            except:
                raise BedrockError(
                    message=json.dumps(outputText),
                    status_code=response_metadata.get("HTTPStatusCode", 500),
                )

        ## CALCULATING USAGE - baseten charges on time, not tokens - have some mapping of cost here.
        prompt_tokens = response_metadata.get(
            "x-amzn-bedrock-input-token-count", len(encoding.encode(prompt))
        )
        completion_tokens = response_metadata.get(
            "x-amzn-bedrock-output-token-count",
            len(
                encoding.encode(
                    model_response["choices"][0]["message"].get("content", "")
                )
            ),
        )

        model_response["created"] = int(time.time())
        model_response["model"] = model
        usage = Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )
        model_response.usage = usage
        model_response._hidden_params["region_name"] = client.meta.region_name
        print_verbose(f"model_response._hidden_params: {model_response._hidden_params}")
        return model_response
    except BedrockError as e:
        exception_mapping_worked = True
        raise e
    except Exception as e:
        if exception_mapping_worked:
            raise e
        else:
            import traceback

            raise BedrockError(status_code=500, message=traceback.format_exc())


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
    body = json.dumps(data).encode("utf-8")
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
    api_key: Optional[str] = None,
    logging_obj=None,
    model_response=None,
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

    # use passed in BedrockRuntime.Client if provided, otherwise create a new one
    client = init_bedrock_client(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        aws_region_name=aws_region_name,
        aws_bedrock_runtime_endpoint=aws_bedrock_runtime_endpoint,
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
    model_response["object"] = "list"
    model_response["data"] = embedding_response
    model_response["model"] = model
    input_tokens = 0

    input_str = "".join(input)

    input_tokens += len(encoding.encode(input_str))

    usage = Usage(
        prompt_tokens=input_tokens, completion_tokens=0, total_tokens=input_tokens + 0
    )
    model_response.usage = usage

    return model_response


def image_generation(
    model: str,
    prompt: str,
    timeout=None,
    logging_obj=None,
    model_response=None,
    optional_params=None,
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

    # use passed in BedrockRuntime.Client if provided, otherwise create a new one
    client = init_bedrock_client(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        aws_region_name=aws_region_name,
        aws_bedrock_runtime_endpoint=aws_bedrock_runtime_endpoint,
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
        body={body},
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

    image_list: List = []
    for artifact in response_body["artifacts"]:
        image_dict = {"url": artifact["base64"]}

    model_response.data = image_dict
    return model_response
