import json, types, time
from typing import Callable, Optional, Any, Union, List

import httpx
import litellm
from litellm.utils import ModelResponse, get_secret, Usage, ImageResponse

from .prompt_templates import factory as ptf

class WatsonxError(Exception):
    def __init__(self, status_code, message):
        self.status_code = status_code
        self.message = message
        self.request = httpx.Request(
            method="POST", url="https://https://us-south.ml.cloud.ibm.com"
        )
        self.response = httpx.Response(status_code=status_code, request=self.request)
        super().__init__(
            self.message
        )  # Call the base class constructor with the parameters it needs

class IBMWatsonXConfig:
    """
    Reference: https://cloud.ibm.com/apidocs/watsonx-ai#deployments-text-generation
    (See ibm_watsonx_ai.metanames.GenTextParamsMetaNames for a list of all available params)

    Supported params for all available watsonx.ai foundational models.

    - `decoding_method` (str): One of "greedy" or "sample"

    - `temperature` (float): Sets the model temperature for sampling - not available when decoding_method='greedy'.

    - `max_new_tokens` (integer): Maximum length of the generated tokens.

    - `min_new_tokens` (integer): Maximum length of input tokens. Any more than this will be truncated.

    - `stop_sequences` (string[]): list of strings to use as stop sequences.

    - `time_limit` (integer): time limit in milliseconds. If the generation is not completed within the time limit, the model will return the generated text up to that point.

    - `top_p` (integer): top p for sampling - not available when decoding_method='greedy'.

    - `top_k` (integer): top k for sampling - not available when decoding_method='greedy'.

    - `repetition_penalty` (float): token repetition penalty during text generation.

    - `stream` (bool): If True, the model will return a stream of responses.

    - `return_options` (dict): A dictionary of options to return. Options include "input_text", "generated_tokens", "input_tokens", "token_ranks".

    - `truncate_input_tokens` (integer): Truncate input tokens to this length.

    - `length_penalty` (dict): A dictionary with keys "decay_factor" and "start_index".

    - `random_seed` (integer): Random seed for text generation.

    - `guardrails` (bool): Enable guardrails for harmful content.

    - `guardrails_hap_params` (dict): Guardrails for harmful content.

    - `guardrails_pii_params` (dict): Guardrails for Personally Identifiable Information.

    - `concurrency_limit` (integer): Maximum number of concurrent requests.

    - `async_mode` (bool): Enable async mode.

    - `verify` (bool): Verify the SSL certificate of calls to the watsonx url.

    - `validate` (bool): Validate the model_id at initialization.

    - `model_inference` (ibm_watsonx_ai.ModelInference): An instance of an ibm_watsonx_ai.ModelInference class to use instead of creating a new model instance.

    - `watsonx_client` (ibm_watsonx_ai.APIClient): An instance of an ibm_watsonx_ai.APIClient class to initialize the watsonx model with.
    """
    decoding_method: Optional[str] = "sample" # 'sample' or 'greedy'. "sample" follows the default openai API behavior
    temperature: Optional[float] = None # 
    min_new_tokens: Optional[int] = None
    max_new_tokens: Optional[int] = litellm.max_tokens
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    random_seed: Optional[int] = None # e.g 42
    repetition_penalty: Optional[float] = None
    stop_sequences: Optional[List[str]] = None # e.g ["}", ")", "."]
    time_limit: Optional[int] = None # e.g 10000 (timeout in milliseconds)
    return_options: Optional[dict] = None # e.g {"input_text": True, "generated_tokens": True, "input_tokens": True, "token_ranks": False}
    truncate_input_tokens: Optional[int] = None # e.g 512
    length_penalty: Optional[dict] = None # e.g {"decay_factor": 2.5, "start_index": 5}
    stream: Optional[bool] = False
    # other inference params
    guardrails: Optional[bool] = False # enable guardrails
    guardrails_hap_params: Optional[dict] = None  # guardrails for harmful content
    guardrails_pii_params: Optional[dict] = None # guardrails for Personally Identifiable Information
    concurrency_limit: Optional[int] = 10 # max number of concurrent requests
    async_mode: Optional[bool] = False # enable async mode
    verify: Optional[Union[bool,str]] = None # verify the SSL certificate of calls to the watsonx url
    validate: Optional[bool] = False # validate the model_id at initialization
    model_inference: Optional[object] = None # an instance of an ibm_watsonx_ai.ModelInference class to use instead of creating a new model instance
    watsonx_client: Optional[object] = None # an instance of an ibm_watsonx_ai.APIClient class to initialize the watsonx model with

    def __init__(
        self,
        decoding_method: Optional[str] = None,
        temperature: Optional[float] = None,
        min_new_tokens: Optional[int] = None,
        max_new_tokens: Optional[
            int
        ] = litellm.max_tokens,  # petals requires max tokens to be set
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        random_seed: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None,
        time_limit: Optional[int] = None,
        return_options: Optional[dict] = None,
        truncate_input_tokens: Optional[int] = None,
        length_penalty: Optional[dict] = None,
        stream: Optional[bool] = False,
        guardrails: Optional[bool] = False,
        guardrails_hap_params: Optional[dict] = None,
        guardrails_pii_params: Optional[dict] = None,
        concurrency_limit: Optional[int] = 10,
        async_mode: Optional[bool] = False,
        verify: Optional[Union[bool,str]] = None,
        validate: Optional[bool] = False,
        model_inference: Optional[object] = None,
        watsonx_client: Optional[object] = None,
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

    def get_supported_openai_params(self):
        return [
            "temperature", # equivalent to temperature
            "max_tokens", # equivalent to max_new_tokens
            "top_p", # equivalent to top_p
            "frequency_penalty", # equivalent to repetition_penalty
            "stop", # equivalent to stop_sequences
            "seed", # equivalent to random_seed
            "stream", # equivalent to stream
        ]


def init_watsonx_model(
    model_id: str,
    url: Optional[str] = None,
    api_key: Optional[str] = None,
    project_id: Optional[str] = None,
    space_id: Optional[str] = None,
    wx_credentials: Optional[dict] = None,
    region_name: Optional[str] = None,
    verify: Optional[Union[bool,str]] = None,
    validate: Optional[bool] = False,
    watsonx_client: Optional[object] = None,
    model_params: Optional[dict] = None,
):
    """
    Initialize a watsonx.ai model for inference.

    Args:

    model_id (str): The model ID to use for inference. If this is a model deployed in a deployment space, the model_id should be in the format 'deployment/<deployment_id>' and the space_id to the deploymend space should be provided.
    url (str): The URL of the watsonx.ai instance.
    api_key (str): The API key for the watsonx.ai instance.
    project_id (str): The project ID for the watsonx.ai instance.
    space_id (str): The space ID for the deployment space.
    wx_credentials (dict): A dictionary containing 'apikey' and 'url' keys for the watsonx.ai instance.
    region_name (str): The region name for the watsonx.ai instance (e.g. 'us-south').
    verify (bool): Whether to verify the SSL certificate of calls to the watsonx url.
    validate (bool): Whether to validate the model_id at initialization.
    watsonx_client (object): An instance of the ibm_watsonx_ai.APIClient class. If this is provided, the model will be initialized using the provided client.
    model_params (dict): A dictionary containing additional parameters to pass to the model (see IBMWatsonXConfig for a list of supported parameters).
    """

    from ibm_watsonx_ai import APIClient
    from ibm_watsonx_ai.foundation_models import ModelInference
        

    if wx_credentials is not None:
        if 'apikey' not in wx_credentials and 'api_key' in wx_credentials:
            wx_credentials['apikey'] = wx_credentials.pop('api_key')
        if 'apikey' not in wx_credentials:
            raise WatsonxError(500, "Error: key 'apikey' expected in wx_credentials")

    if url is None:
        url = get_secret("WX_URL") or get_secret("WATSONX_URL") or get_secret("WML_URL")
    if api_key is None:
        api_key = get_secret("WX_API_KEY") or get_secret("WML_API_KEY")
    if project_id is None:
        project_id = get_secret("WX_PROJECT_ID") or get_secret("PROJECT_ID")
    if region_name is None:
        region_name = get_secret("WML_REGION_NAME") or get_secret("WX_REGION_NAME") or get_secret("REGION_NAME")
    if space_id is None:
        space_id = get_secret("WX_SPACE_ID")  or get_secret("WML_DEPLOYMENT_SPACE_ID") or get_secret("SPACE_ID")
    

    ## CHECK IS  'os.environ/' passed in
    # Define the list of parameters to check
    params_to_check = (url, api_key, project_id, space_id, region_name)
    # Iterate over parameters and update if needed
    for i, param in enumerate(params_to_check):
        if param and param.startswith("os.environ/"):
            params_to_check[i] = get_secret(param)
    # Assign updated values back to parameters
    url, api_key, project_id, space_id, region_name = params_to_check

    ### SET WATSONX URL
    if url is not None or watsonx_client is not None or wx_credentials is not None:
        pass
    elif region_name is not None:
        url = f"https://{region_name}.ml.cloud.ibm.com"
    else:
        raise WatsonxError(
            message="Watsonx URL not set: set WX_URL env variable or in .env file",
            status_code=401,
        )
    if watsonx_client is not None and project_id is None:
        project_id = watsonx_client.project_id
    
    if model_id.startswith("deployment/"):
        # deployment models are passed in as 'deployment/<deployment_id>'
        assert space_id is not None, "space_id is required for deployment models"
        deployment_id = '/'.join(model_id.split("/")[1:])
        model_id = None
    else:
        deployment_id = None

    if watsonx_client is not None:
        model = ModelInference(
            model_id=model_id,
            params=model_params,
            api_client=watsonx_client,
            project_id=project_id,
            deployment_id=deployment_id,
            verify=verify,
            validate=validate,
            space_id=space_id,
        )
    elif wx_credentials is not None:
        model = ModelInference(
            model_id=model_id,
            params=model_params,
            credentials=wx_credentials,
            project_id=project_id,
            deployment_id=deployment_id,
            verify=verify,
            validate=validate,
            space_id=space_id,
        )
    elif api_key is not None:
        model = ModelInference(
            model_id=model_id,
            params=model_params,
            credentials={
                "apikey": api_key,
                "url": url,
            },
            project_id=project_id,
            deployment_id=deployment_id,
            verify=verify,
            validate=validate,
            space_id=space_id,
        )
    else:
        raise WatsonxError(500, "WatsonX credentials not passed or could not be found.")


    return model


def convert_messages_to_prompt(model, messages, provider, custom_prompt_dict):
    # handle anthropic prompts and amazon titan prompts
    if model in custom_prompt_dict:
        # check if the model has a registered custom prompt
        model_prompt_dict = custom_prompt_dict[model]
        prompt = ptf.custom_prompt(
            messages=messages,
            role_dict=model_prompt_dict.get("role_dict", model_prompt_dict.get("roles")),
            initial_prompt_value=model_prompt_dict.get("initial_prompt_value",""),
            final_prompt_value=model_prompt_dict.get("final_prompt_value", ""),
            bos_token=model_prompt_dict.get("bos_token", ""),
            eos_token=model_prompt_dict.get("eos_token", ""),
        )
        return prompt
    elif provider == "ibm":
        prompt = ptf.prompt_factory(
            model=model, messages=messages, custom_llm_provider="watsonx"
        )
    elif provider == "ibm-mistralai":
        prompt = ptf.mistral_instruct_pt(messages=messages)
    else:
        prompt = ptf.prompt_factory(model=model, messages=messages, custom_llm_provider='watsonx')
    return prompt


"""
IBM watsonx.ai AUTH Keys/Vars
os.environ['WX_URL'] = ""
os.environ['WX_API_KEY'] = ""
os.environ['WX_PROJECT_ID'] = ""
"""

def completion(
    model: str,
    messages: list,
    custom_prompt_dict: dict,
    model_response: ModelResponse,
    print_verbose: Callable,
    encoding,
    logging_obj,
    optional_params:Optional[dict]=None,
    litellm_params:Optional[dict]=None,
    logger_fn=None,
    timeout:float=None,
):
    from ibm_watsonx_ai.foundation_models import Model, ModelInference   

    try:
        stream = optional_params.pop("stream", False)
        extra_generate_params = dict(
            guardrails=optional_params.pop("guardrails", False),
            guardrails_hap_params=optional_params.pop("guardrails_hap_params", None),
            guardrails_pii_params=optional_params.pop("guardrails_pii_params", None),
            concurrency_limit=optional_params.pop("concurrency_limit", 10),
            async_mode=optional_params.pop("async_mode", False),
        )
        if timeout is not None and optional_params.get("time_limit") is None:
            # the time_limit in watsonx.ai is in milliseconds (as opposed to OpenAI which is in seconds)
            optional_params['time_limit'] = max(0, int(timeout*1000))
        extra_body_params = optional_params.pop("extra_body", {})
        optional_params.update(extra_body_params)
        # LOAD CONFIG
        config = IBMWatsonXConfig.get_config()
        for k, v in config.items():
            if k not in optional_params:
                optional_params[k] = v

        model_inference = optional_params.pop("model_inference", None)
        if model_inference is None:
            # INIT MODEL
            model_client:ModelInference = init_watsonx_model(
                model_id=model,
                url=optional_params.pop("url", None),
                api_key=optional_params.pop("api_key", None),
                project_id=optional_params.pop("project_id", None),
                space_id=optional_params.pop("space_id", None),
                wx_credentials=optional_params.pop("wx_credentials", None),
                region_name=optional_params.pop("region_name", None),
                verify=optional_params.pop("verify", None),
                validate=optional_params.pop("validate", False),
                watsonx_client=optional_params.pop("watsonx_client", None),
                model_params=optional_params,
            )
        else:
            model_client:ModelInference = model_inference
            model = model_client.model_id

        # MAKE PROMPT
        provider = model.split("/")[0]
        model_name = '/'.join(model.split("/")[1:])
        prompt = convert_messages_to_prompt(
            model, messages, provider, custom_prompt_dict
        )
        ## COMPLETION CALL
        if stream is True:
            request_str = (
                "response = model.generate_text_stream(\n"
                f"\tprompt={prompt},\n"
                "\traw_response=True\n)"
            )
            logging_obj.pre_call(
                input=prompt,
                api_key="",
                additional_args={
                    "complete_input_dict": optional_params,
                    "request_str": request_str,
                },
            )
            # remove params that are not needed for streaming
            del extra_generate_params["async_mode"]
            del extra_generate_params["concurrency_limit"]
            # make generate call
            response = model_client.generate_text_stream(
                prompt=prompt,
                raw_response=True,
                **extra_generate_params
            )
            return litellm.CustomStreamWrapper(
                response,
                model=model,
                custom_llm_provider="watsonx",
                logging_obj=logging_obj,
            )
        else:
            try:
                ## LOGGING
                request_str = (
                "response = model.generate(\n"
                    f"\tprompt={prompt},\n"
                    "\traw_response=True\n)"
                )
                logging_obj.pre_call(
                    input=prompt,
                    api_key="",
                    additional_args={
                        "complete_input_dict": optional_params,
                        "request_str": request_str,
                    },
                )
                response = model_client.generate(
                    prompt=prompt,
                    **extra_generate_params
                )
            except Exception as e:
                raise WatsonxError(status_code=500, message=str(e))

        ## LOGGING
        logging_obj.post_call(
            input=prompt,
            api_key="",
            original_response=json.dumps(response),
            additional_args={"complete_input_dict": optional_params},
        )
        print_verbose(f"raw model_response: {response}")
        ## BUILD RESPONSE OBJECT
        output_text = response['results'][0]['generated_text']

        try:
            if (
                len(output_text) > 0
                and hasattr(model_response.choices[0], "message")
            ):
                model_response["choices"][0]["message"]["content"] = output_text
                model_response["finish_reason"] = response['results'][0]['stop_reason']
                prompt_tokens = response['results'][0]['input_token_count']
                completion_tokens = response['results'][0]['generated_token_count']
            else:
                raise Exception()
        except:
            raise WatsonxError(
                message=json.dumps(output_text),
                status_code=500,
            )
        model_response['created'] = int(time.time())
        model_response['model'] = model_name
        usage = Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )
        model_response.usage = usage
        return model_response
    except WatsonxError as e:
        raise e
    except Exception as e:
        raise WatsonxError(status_code=500, message=str(e))


def embedding():
    # logic for parsing in - calling - parsing out model embedding calls
    pass