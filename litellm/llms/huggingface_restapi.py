## Uses the huggingface text generation inference API
import os, copy, types
import json
from enum import Enum
import requests
import time
import litellm
from typing import Callable
from litellm.utils import ModelResponse, Choices, Message
from typing import Optional
from .prompt_templates.factory import prompt_factory, custom_prompt

class HuggingfaceError(Exception):
    def __init__(self, status_code, message):
        self.status_code = status_code
        self.message = message
        super().__init__(
            self.message
        )  # Call the base class constructor with the parameters it needs

class HuggingfaceConfig(): 
    """
    Reference: https://huggingface.github.io/text-generation-inference/#/Text%20Generation%20Inference/compat_generate 
    """
    best_of: Optional[int] = None
    decoder_input_details: Optional[bool] = None
    details: Optional[bool] = True # enables returning logprobs + best of
    max_new_tokens: Optional[int] = None
    repetition_penalty: Optional[float] = None
    return_full_text: Optional[bool] = False # by default don't return the input as part of the output
    seed: Optional[int] = None
    temperature: Optional[float] = None
    top_k: Optional[int] = None
    top_n_tokens: Optional[int] = None
    top_p: Optional[int] = None
    truncate: Optional[int] = None
    typical_p: Optional[float] = None
    watermark: Optional[bool] = None

    def __init__(self, 
                 best_of: Optional[int] = None, 
                 decoder_input_details: Optional[bool] = None,
                 details: Optional[bool] = None,
                 max_new_tokens: Optional[int] = None,
                 repetition_penalty: Optional[float] = None,
                 return_full_text: Optional[bool] = None,
                 seed: Optional[int] = None,
                 temperature: Optional[float] = None, 
                 top_k: Optional[int] = None,
                 top_n_tokens: Optional[int] = None, 
                 top_p: Optional[int] = None,
                 truncate: Optional[int] = None,
                 typical_p: Optional[float] = None,
                 watermark: Optional[bool] = None
                 ) -> None:
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

def validate_environment(api_key):
    headers = {
        "content-type": "application/json",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers

tgi_models_cache = None
conv_models_cache = None
def read_tgi_conv_models():
    try:
        global tgi_models_cache, conv_models_cache
        # Check if the cache is already populated
        # so we don't keep on reading txt file if there are 1k requests
        if (tgi_models_cache is not None) and (conv_models_cache is not None):
            return tgi_models_cache, conv_models_cache
        # If not, read the file and populate the cache
        tgi_models = set()
        script_directory = os.path.dirname(os.path.abspath(__file__))
        # Construct the file path relative to the script's directory
        file_path = os.path.join(script_directory, "huggingface_llms_metadata", "hf_text_generation_models.txt")

        with open(file_path, 'r') as file:
            for line in file:
                tgi_models.add(line.strip())
        
        # Cache the set for future use
        tgi_models_cache = tgi_models
        
        # If not, read the file and populate the cache
        file_path = os.path.join(script_directory, "huggingface_llms_metadata", "hf_conversational_models.txt")
        conv_models = set()
        with open(file_path, 'r') as file:
            for line in file:
                conv_models.add(line.strip())
        # Cache the set for future use
        conv_models_cache = conv_models        
        return tgi_models, conv_models
    except:
        return set(), set()


def get_hf_task_for_model(model):
    # read text file, cast it to set 
    # read the file called "huggingface_llms_metadata/hf_text_generation_models.txt"
    tgi_models, conversational_models = read_tgi_conv_models()
    if model in tgi_models:
        return "text-generation-inference"
    elif model in conversational_models:
        return "conversational"
    elif "roneneldan/TinyStories" in model:
        return None
    else:
        return "text-generation-inference" # default to tgi

def completion(
    model: str,
    messages: list,
    api_base: Optional[str],
    model_response: ModelResponse,
    print_verbose: Callable,
    encoding,
    api_key,
    logging_obj,
    custom_prompt_dict={},
    optional_params=None,
    litellm_params=None,
    logger_fn=None,
):
    headers = validate_environment(api_key)
    task = get_hf_task_for_model(model)
    print_verbose(f"{model}, {task}")
    completion_url = ""
    input_text = None
    if "https" in model:
        completion_url = model
    elif api_base:
        completion_url = api_base
    elif "HF_API_BASE" in os.environ:
        completion_url = os.getenv("HF_API_BASE", "")
    elif "HUGGINGFACE_API_BASE" in os.environ:
        completion_url = os.getenv("HUGGINGFACE_API_BASE", "")
    else:
        completion_url = f"https://api-inference.huggingface.co/models/{model}"

    ## Load Config
    config=litellm.HuggingfaceConfig.get_config()
    for k, v in config.items():
        if k not in optional_params: # completion(top_k=3) > huggingfaceConfig(top_k=3) <- allows for dynamic variables to be passed in
            optional_params[k] = v

    ### MAP INPUT PARAMS
    if task == "conversational":
        inference_params = copy.deepcopy(optional_params)
        inference_params.pop("details")
        inference_params.pop("return_full_text")
        past_user_inputs = []
        generated_responses = []
        text = ""
        for message in messages:
            if message["role"] == "user":
                if text != "":
                    past_user_inputs.append(text)
                text = message["content"]
            elif message["role"] == "assistant" or message["role"] == "system":
                generated_responses.append(message["content"])
        data = {
            "inputs": {
                "text": text, 
                "past_user_inputs": past_user_inputs, 
                "generated_responses": generated_responses
            },
            "parameters": inference_params
        }
        input_text = "".join(message["content"] for message in messages)
    elif task == "text-generation-inference":
        # always send "details" and "return_full_text" as params
        if model in custom_prompt_dict:
            # check if the model has a registered custom prompt
            model_prompt_details = custom_prompt_dict[model]
            prompt = custom_prompt(
                role_dict=model_prompt_details["roles"], 
                initial_prompt_value=model_prompt_details["initial_prompt_value"],  
                final_prompt_value=model_prompt_details["final_prompt_value"], 
                messages=messages
            )
        else:
            prompt = prompt_factory(model=model, messages=messages)
        data = {
            "inputs": prompt,
            "parameters": optional_params,
            "stream": True if "stream" in optional_params and optional_params["stream"] == True else False,
        }
        input_text = prompt
    else:
        # Non TGI and Conversational llms
        # We need this branch, it removes 'details' and 'return_full_text' from params 
        if model in custom_prompt_dict:
            # check if the model has a registered custom prompt
            model_prompt_details = custom_prompt_dict[model]
            prompt = custom_prompt(
                role_dict=model_prompt_details["roles"], 
                initial_prompt_value=model_prompt_details["initial_prompt_value"],  
                final_prompt_value=model_prompt_details["final_prompt_value"], 
                messages=messages
            )
        else:
            prompt = prompt_factory(model=model, messages=messages)
        inference_params = copy.deepcopy(optional_params)
        inference_params.pop("details")
        inference_params.pop("return_full_text")
        data = {
            "inputs": prompt,
            "parameters": inference_params,
            "stream": True if "stream" in optional_params and optional_params["stream"] == True else False,
        }
        input_text = prompt
    ## LOGGING
    logging_obj.pre_call(
            input=input_text,
            api_key=api_key,
            additional_args={"complete_input_dict": data, "task": task},
        )
    ## COMPLETION CALL
    if "stream" in optional_params and optional_params["stream"] == True:
        response = requests.post(
            completion_url, 
            headers=headers, 
            data=json.dumps(data), 
            stream=optional_params["stream"]
        )
        return response.iter_lines()
    else:
        response = requests.post(
            completion_url, 
            headers=headers, 
            data=json.dumps(data)
        )
        ## LOGGING
        logging_obj.post_call(
            input=input_text,
            api_key=api_key,
            original_response=response.text,
            additional_args={"complete_input_dict": data, "task": task},
        )
        ## RESPONSE OBJECT
        try:
            completion_response = response.json()
        except:
            raise HuggingfaceError(
                message=response.text, status_code=response.status_code
            )
        print_verbose(f"response: {completion_response}")
        if isinstance(completion_response, dict) and "error" in completion_response:
            print_verbose(f"completion error: {completion_response['error']}")
            print_verbose(f"response.status_code: {response.status_code}")
            raise HuggingfaceError(
                message=completion_response["error"],
                status_code=response.status_code,
            )
        else:
            if task == "conversational": 
                model_response["choices"][0]["message"][
                    "content"
                ] = completion_response["generated_text"]
            elif task == "text-generation-inference": 
                model_response["choices"][0]["message"][
                    "content"
                ] = completion_response[0]["generated_text"]   
                ## GETTING LOGPROBS + FINISH REASON 
                if "details" in completion_response[0] and "tokens" in completion_response[0]["details"]:
                    model_response.choices[0].finish_reason = completion_response[0]["details"]["finish_reason"]
                    sum_logprob = 0
                    for token in completion_response[0]["details"]["tokens"]:
                        sum_logprob += token["logprob"]
                    model_response["choices"][0]["message"]["logprobs"] = sum_logprob
                if "best_of" in optional_params and optional_params["best_of"] > 1: 
                    if "details" in completion_response[0] and "best_of_sequences" in completion_response[0]["details"]:
                        choices_list = []
                        for idx, item in enumerate(completion_response[0]["details"]["best_of_sequences"]):
                            sum_logprob = 0
                            for token in item["tokens"]:
                                sum_logprob += token["logprob"]
                            message_obj = Message(content=item["generated_text"], logprobs=sum_logprob)
                            choice_obj = Choices(finish_reason=item["finish_reason"], index=idx+1, message=message_obj)
                            choices_list.append(choice_obj)
                        model_response["choices"].extend(choices_list)
            else:
                model_response["choices"][0]["message"]["content"] = completion_response[0]["generated_text"]
        ## CALCULATING USAGE
        prompt_tokens = len(
            encoding.encode(input_text)
        )  ##[TODO] use the llama2 tokenizer here
        completion_tokens = len(
            encoding.encode(model_response["choices"][0]["message"]["content"])
        )  ##[TODO] use the llama2 tokenizer here

        model_response["created"] = time.time()
        model_response["model"] = model
        model_response["usage"] = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
        return model_response


def embedding(
    model: str,
    input: list,
    api_key: str,
    api_base: str,
    logging_obj=None,
    model_response=None,
    encoding=None,
):
    headers = validate_environment(api_key)
    # print_verbose(f"{model}, {task}")
    embed_url = ""
    if "https" in model:
        embed_url = model
    elif api_base:
        embed_url = api_base
    elif "HF_API_BASE" in os.environ:
        embed_url = os.getenv("HF_API_BASE", "")
    elif "HUGGINGFACE_API_BASE" in os.environ:
        embed_url = os.getenv("HUGGINGFACE_API_BASE", "")
    else:
        embed_url = f"https://api-inference.huggingface.co/models/{model}"
    
    data = {
        "inputs": input
    }
    
    ## LOGGING
    logging_obj.pre_call(
            input=input,
            api_key=api_key,
            additional_args={"complete_input_dict": data},
        )
    ## COMPLETION CALL
    response = requests.post(
        embed_url, headers=headers, data=json.dumps(data)
    )

  
    ## LOGGING
    logging_obj.post_call(
            input=input,
            api_key=api_key,
            additional_args={"complete_input_dict": data},
            original_response=response,
        )


    embeddings = response.json()

    output_data = []
    for idx, embedding in enumerate(embeddings):
        output_data.append(
            {
                "object": "embedding",
                "index": idx,
                "embedding": embedding[0][0] # flatten list returned from hf
            }
        )
    model_response["object"] = "list"
    model_response["data"] = output_data
    model_response["model"] = model
    input_tokens = 0
    for text in input:
        input_tokens+=len(encoding.encode(text)) 

    model_response["usage"] = { 
        "prompt_tokens": input_tokens, 
        "total_tokens": input_tokens,
    }
    return model_response


    
