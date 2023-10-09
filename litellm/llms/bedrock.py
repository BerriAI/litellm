import json, copy, types
from enum import Enum
import time
from typing import Callable, Optional
import litellm
from litellm.utils import ModelResponse, get_secret

class BedrockError(Exception):
    def __init__(self, status_code, message):
        self.status_code = status_code
        self.message = message
        super().__init__(
            self.message
        )  # Call the base class constructor with the parameters it needs

class AmazonTitanConfig(): 
    """
    Reference: https://us-west-2.console.aws.amazon.com/bedrock/home?region=us-west-2#/providers?model=titan-text-express-v1

    Supported Params for the Amazon Titan models:

    - `maxTokenCount` (integer) max tokens,
    - `stopSequences` (string[]) list of stop sequence strings
    - `temperature` (float) temperature for model,
    - `topP` (int) top p for model
    """
    maxTokenCount: Optional[int]=None
    stopSequences: Optional[list]=None
    temperature: Optional[float]=None
    topP: Optional[int]=None

    def __init__(self, 
                 maxTokenCount: Optional[int]=None,
                 stopSequences: Optional[list]=None,
                 temperature: Optional[float]=None,
                 topP: Optional[int]=None) -> None:
        locals_ = locals()
        for key, value in locals_.items():
            if key != 'self' and value is not None:
                setattr(self.__class__, key, value)
    
    @classmethod
    def get_config(cls):
        return {k: v for k, v in cls.__dict__.items() 
                if not k.startswith('__') 
                and not isinstance(v, (types.FunctionType, types.BuiltinFunctionType, classmethod, staticmethod)) 
                and v is not None}

class AmazonAnthropicConfig(): 
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
    max_tokens_to_sample: Optional[int]=litellm.max_tokens
    stop_sequences: Optional[list]=None
    temperature: Optional[float]=None
    top_k: Optional[int]=None
    top_p: Optional[int]=None
    anthropic_version: Optional[str]=None

    def __init__(self, 
                 max_tokens_to_sample: Optional[int]=None,
                 stop_sequences: Optional[list]=None,
                 temperature: Optional[float]=None,
                 top_k: Optional[int]=None,
                 top_p: Optional[int]=None,
                 anthropic_version: Optional[str]=None) -> None:
        locals_ = locals()
        for key, value in locals_.items():
            if key != 'self' and value is not None:
                setattr(self.__class__, key, value)
    
    @classmethod
    def get_config(cls):
        return {k: v for k, v in cls.__dict__.items() 
                if not k.startswith('__') 
                and not isinstance(v, (types.FunctionType, types.BuiltinFunctionType, classmethod, staticmethod)) 
                and v is not None}

class AmazonCohereConfig(): 
    """
    Reference: https://us-west-2.console.aws.amazon.com/bedrock/home?region=us-west-2#/providers?model=command

    Supported Params for the Amazon / Cohere models:

    - `max_tokens` (integer) max tokens,
    - `temperature` (float) model temperature,
    - `return_likelihood` (string) n/a
    """
    max_tokens: Optional[int]=None
    temperature: Optional[float]=None
    return_likelihood: Optional[str]=None

    def __init__(self, 
                 max_tokens: Optional[int]=None,
                 temperature: Optional[float]=None,
                 return_likelihood: Optional[str]=None) -> None:
        locals_ = locals()
        for key, value in locals_.items():
            if key != 'self' and value is not None:
                setattr(self.__class__, key, value)
    
    @classmethod
    def get_config(cls):
        return {k: v for k, v in cls.__dict__.items() 
                if not k.startswith('__') 
                and not isinstance(v, (types.FunctionType, types.BuiltinFunctionType, classmethod, staticmethod)) 
                and v is not None}

class AmazonAI21Config(): 
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
    maxTokens: Optional[int]=None
    temperature: Optional[float]=None
    topP: Optional[float]=None
    stopSequences: Optional[list]=None
    frequencePenalty: Optional[dict]=None
    presencePenalty: Optional[dict]=None
    countPenalty: Optional[dict]=None

    def __init__(self,
                 maxTokens: Optional[int]=None,
                 temperature: Optional[float]=None,
                 topP: Optional[float]=None,
                 stopSequences: Optional[list]=None,
                 frequencePenalty: Optional[dict]=None,
                 presencePenalty: Optional[dict]=None,
                 countPenalty: Optional[dict]=None) -> None:
        locals_ = locals()
        for key, value in locals_.items():
            if key != 'self' and value is not None:
                setattr(self.__class__, key, value)
    
    @classmethod
    def get_config(cls):
        return {k: v for k, v in cls.__dict__.items() 
                if not k.startswith('__') 
                and not isinstance(v, (types.FunctionType, types.BuiltinFunctionType, classmethod, staticmethod)) 
                and v is not None}

class AnthropicConstants(Enum):
    HUMAN_PROMPT = "\n\nHuman:"
    AI_PROMPT = "\n\nAssistant:"


def init_bedrock_client(
        region_name = None,
        aws_access_key_id = None,
        aws_secret_access_key = None,
        aws_region_name=None,
    ):

    import boto3
    if aws_access_key_id != None:
        # uses auth params passed to completion
        # aws_access_key_id is not None, assume user is trying to auth using litellm.completion

        client = boto3.client(
            service_name="bedrock-runtime",
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=aws_region_name,
            endpoint_url=f'https://bedrock-runtime.{aws_region_name}.amazonaws.com'
        )
    else:
        # aws_access_key_id is None, assume user is trying to auth using env variables 
        # boto3 automaticaly reads env variables

        # we need to read region name from env
        # I assume majority of users use .env for auth 
        region_name = (
            get_secret("AWS_REGION_NAME") or
            "us-west-2"  # default to us-west-2 if user not specified
        )
        client = boto3.client(
            service_name="bedrock-runtime",
            region_name=region_name,
            endpoint_url=f'https://bedrock-runtime.{region_name}.amazonaws.com'
        )

    return client


def convert_messages_to_prompt(messages, provider):
    # handle anthropic prompts using anthropic constants
    if provider == "anthropic":
        prompt = ""
        for message in messages:
            if "role" in message:
                if message["role"] == "user":
                    prompt += (
                        f"{AnthropicConstants.HUMAN_PROMPT.value}{message['content']}"
                    )
                elif message["role"] == "system":
                    prompt += (
                        f"{AnthropicConstants.HUMAN_PROMPT.value}<admin>{message['content']}</admin>"
                    )
                else:
                    prompt += (
                        f"{AnthropicConstants.AI_PROMPT.value}{message['content']}"
                    )
            else:
                prompt += f"{AnthropicConstants.HUMAN_PROMPT.value}{message['content']}"
        prompt += f"{AnthropicConstants.AI_PROMPT.value}"
    else:
        prompt = ""
        for message in messages:
            if "role" in message:
                if message["role"] == "user":
                    prompt += (
                        f"{message['content']}"
                    )
                else:
                    prompt += (
                        f"{message['content']}"
                    )
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
        model_response: ModelResponse,
        print_verbose: Callable,
        encoding,
        logging_obj,
        optional_params=None,
        litellm_params=None,
        logger_fn=None,
):
    # pop aws_secret_access_key, aws_access_key_id, aws_region_name from kwargs, since completion calls fail with them
    aws_secret_access_key = optional_params.pop("aws_secret_access_key", None)
    aws_access_key_id = optional_params.pop("aws_access_key_id", None)
    aws_region_name = optional_params.pop("aws_region_name", None)

    # use passed in BedrockRuntime.Client if provided, otherwise create a new one
    client = optional_params.pop(
        "aws_bedrock_client",
        # only pass variables that are not None
        init_bedrock_client(
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_region_name=aws_region_name,
        ),
    )

    model = model
    provider = model.split(".")[0]
    prompt = convert_messages_to_prompt(messages, provider)
    inference_params = copy.deepcopy(optional_params)
    stream = inference_params.pop("stream", False)

    print(f"bedrock provider: {provider}")
    if provider == "anthropic":
        ## LOAD CONFIG
        config = litellm.AmazonAnthropicConfig.get_config() 
        for k, v in config.items(): 
            if k not in inference_params: # completion(top_k=3) > anthropic_config(top_k=3) <- allows for dynamic variables to be passed in
                inference_params[k] = v
        data = json.dumps({
            "prompt": prompt,
            **inference_params
        })
    elif provider == "ai21":
        ## LOAD CONFIG
        config = litellm.AmazonAI21Config.get_config() 
        for k, v in config.items(): 
            if k not in inference_params: # completion(top_k=3) > anthropic_config(top_k=3) <- allows for dynamic variables to be passed in
                inference_params[k] = v

        data = json.dumps({
            "prompt": prompt,
            **inference_params
        })
    elif provider == "cohere":
        ## LOAD CONFIG
        config = litellm.AmazonCohereConfig.get_config() 
        for k, v in config.items(): 
            if k not in inference_params: # completion(top_k=3) > anthropic_config(top_k=3) <- allows for dynamic variables to be passed in
                inference_params[k] = v
        data = json.dumps({
            "prompt": prompt,
            **inference_params
        })
    elif provider == "amazon":  # amazon titan
        ## LOAD CONFIG
        config = litellm.AmazonTitanConfig.get_config() 
        for k, v in config.items(): 
            if k not in inference_params: # completion(top_k=3) > amazon_config(top_k=3) <- allows for dynamic variables to be passed in
                inference_params[k] = v

        data = json.dumps({
            "inputText": prompt,
            "textGenerationConfig": inference_params,
        })
    
    ## LOGGING
    logging_obj.pre_call(
        input=prompt,
        api_key="",
        additional_args={"complete_input_dict": data},
    )

    ## COMPLETION CALL
    accept = 'application/json'
    contentType = 'application/json'
    if stream == True:
        response = client.invoke_model_with_response_stream(
            body=data,
            modelId=model,
            accept=accept,
            contentType=contentType
        )
        response = response.get('body')
        return response

    response = client.invoke_model(
        body=data,
        modelId=model,
        accept=accept,
        contentType=contentType
    )
    response_body = json.loads(response.get('body').read())

    ## LOGGING
    logging_obj.post_call(
        input=prompt,
        api_key="",
        original_response=response_body,
        additional_args={"complete_input_dict": data},
    )
    print_verbose(f"raw model_response: {response}")
    ## RESPONSE OBJECT
    outputText = "default"
    if provider == "ai21":
        outputText = response_body.get('completions')[0].get('data').get('text')
    elif provider == "anthropic":
        outputText = response_body['completion']
        model_response["finish_reason"] = response_body["stop_reason"]
    elif provider == "cohere": 
        outputText = response_body["generations"][0]["text"]
    else:  # amazon titan
        outputText = response_body.get('results')[0].get('outputText')
    if "error" in outputText:
        raise BedrockError(
            message=outputText,
            status_code=response.status_code,
        )
    else:
        try:
            model_response["choices"][0]["message"]["content"] = outputText
        except:
            raise BedrockError(message=json.dumps(outputText), status_code=response.status_code)

    ## CALCULATING USAGE - baseten charges on time, not tokens - have some mapping of cost here. 
    prompt_tokens = len(
        encoding.encode(prompt)
    )
    completion_tokens = len(
        encoding.encode(model_response["choices"][0]["message"]["content"])
    )

    model_response["created"] = time.time()
    model_response["model"] = model
    model_response["usage"] = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }
    return model_response


def embedding():
    # logic for parsing in - calling - parsing out model embedding calls
    pass
