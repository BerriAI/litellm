import json, copy
from enum import Enum
import time
from typing import Callable
from litellm.utils import ModelResponse, get_secret

class BedrockError(Exception):
    def __init__(self, status_code, message):
        self.status_code = status_code
        self.message = message
        super().__init__(
            self.message
        )  # Call the base class constructor with the parameters it needs


class AnthropicConstants(Enum):
    HUMAN_PROMPT = "\n\nHuman:"
    AI_PROMPT = "\n\nAssistant:"


def init_bedrock_client(region_name):
    import boto3

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
    region_name = (
            get_secret("AWS_REGION_NAME") or
            "us-west-2"  # default to us-west-2 if user not specified
    )

    client = init_bedrock_client(region_name)

    model = model
    provider = model.split(".")[0]
    prompt = convert_messages_to_prompt(messages, provider)
    inference_params = copy.deepcopy(optional_params)
    stream = inference_params.pop("stream", False)
    if provider == "anthropic":
        data = json.dumps({
            "prompt": prompt,
            **inference_params
        })
    elif provider == "ai21":
        data = json.dumps({
            "prompt": prompt,
        })

    else:  # amazon titan
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
        original_response=response,
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
