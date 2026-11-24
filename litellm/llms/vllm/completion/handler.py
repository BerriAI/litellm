import time
from collections.abc import Callable, Mapping, Sequence, Sized
from typing import TYPE_CHECKING, Final, Protocol

import httpx

from litellm.litellm_core_utils.prompt_templates.factory import (
    prompt_factory,
    render_custom_prompt_template,
)
from litellm.utils import ModelResponse, Usage

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj

llm = None


class _VLLMCompletionOutput(Protocol):
    @property
    def text(self) -> str | None: ...
    @property
    def token_ids(self) -> Sized: ...


class _VLLMRequestOutput(Protocol):
    @property
    def outputs(self) -> Sequence[_VLLMCompletionOutput]: ...
    @property
    def prompt_token_ids(self) -> Sized: ...


class _VLLMEngine(Protocol):
    def generate(self, prompts: object, sampling_params: object, /) -> Sequence[_VLLMRequestOutput]: ...


class VLLMError(Exception):
    def __init__(self, status_code: int, message: str) -> None:
        self.status_code = status_code
        self.message = message
        self.request = httpx.Request(method="POST", url="http://0.0.0.0:8000")
        self.response = httpx.Response(status_code=status_code, request=self.request)
        super().__init__(self.message)  # Call the base class constructor with the parameters it needs


# check if vllm is installed
def validate_environment(model: str) -> tuple[_VLLMEngine, Callable[..., object]]:
    global llm
    try:
        from vllm import LLM, SamplingParams

        if llm is None:
            llm = LLM(model=model)
        return llm, SamplingParams
    except Exception as e:
        raise VLLMError(status_code=0, message=str(e))


def completion(
    model: str,
    messages: list,
    model_response: ModelResponse,
    print_verbose: Callable[[str], object],
    encoding: object,
    logging_obj: "LiteLLMLoggingObj",
    optional_params: Mapping[str, object],
    custom_prompt_dict: Mapping[str, Mapping[str, object]] = {},
    litellm_params: object = None,
    logger_fn: object = None,
):
    global llm
    try:
        llm, SamplingParams = validate_environment(model=model)
    except Exception as e:
        raise VLLMError(status_code=0, message=str(e))
    sampling_params: Final = SamplingParams(**optional_params)
    if model in custom_prompt_dict:
        # check if the model has a registered custom prompt
        model_prompt_details: Final = custom_prompt_dict[model]
        prompt = render_custom_prompt_template(
            template=model_prompt_details, messages=messages, model=model, llm_provider="vllm"
        )
    else:
        prompt = prompt_factory(model=model, messages=messages)

    ## LOGGING
    logging_obj.pre_call(
        input=prompt,
        api_key="",
        additional_args={"complete_input_dict": sampling_params},
    )

    if llm:
        outputs: Final = llm.generate(prompt, sampling_params)
    else:
        raise VLLMError(status_code=0, message="Need to pass in a model name to initialize vllm")

    ## COMPLETION CALL
    if "stream" in optional_params and optional_params["stream"] is True:
        return iter(outputs)
    else:
        ## LOGGING
        logging_obj.post_call(
            input=prompt,
            api_key="",
            original_response=outputs,
            additional_args={"complete_input_dict": sampling_params},
        )
        print_verbose(f"raw model_response: {outputs}")
        ## RESPONSE OBJECT
        model_response.choices[0].message.content = outputs[0].outputs[0].text

        ## CALCULATING USAGE
        prompt_tokens: Final = len(outputs[0].prompt_token_ids)
        completion_tokens: Final = len(outputs[0].outputs[0].token_ids)

        model_response.created = int(time.time())
        model_response.model = model
        usage: Final = Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )
        setattr(model_response, "usage", usage)
        return model_response


def batch_completions(
    model: str,
    messages: list,
    optional_params: Mapping[str, object] | None = None,
    custom_prompt_dict: Mapping[str, Mapping[str, object]] = {},
) -> list[ModelResponse]:
    """
    Example usage:
    import litellm
    import os
    from litellm import batch_completion


    responses = batch_completion(
        model="vllm/facebook/opt-125m",
        messages = [
            [
                {
                    "role": "user",
                    "content": "good morning? "
                }
            ],
            [
                {
                    "role": "user",
                    "content": "what's the time? "
                }
            ]
        ]
    )
    """
    try:
        llm, SamplingParams = validate_environment(model=model)
    except Exception as e:
        error_str: Final = str(e)
        raise VLLMError(status_code=0, message=error_str)
    sampling_params: Final = SamplingParams(**optional_params)
    prompts: Final[list[object]] = []
    if model in custom_prompt_dict:
        # check if the model has a registered custom prompt
        model_prompt_details: Final = custom_prompt_dict[model]
        for message in messages:
            prompt = render_custom_prompt_template(
                template=model_prompt_details, messages=message, model=model, llm_provider="vllm"
            )
            prompts.append(prompt)
    else:
        for message in messages:
            prompt = prompt_factory(model=model, messages=message)
            prompts.append(prompt)

    if llm:
        outputs: Final = llm.generate(prompts, sampling_params)
    else:
        raise VLLMError(status_code=0, message="Need to pass in a model name to initialize vllm")

    final_outputs: Final[list[ModelResponse]] = []
    for output in outputs:
        model_response = ModelResponse()
        ## RESPONSE OBJECT
        model_response.choices[0].message.content = output.outputs[0].text

        ## CALCULATING USAGE
        prompt_tokens = len(output.prompt_token_ids)
        completion_tokens = len(output.outputs[0].token_ids)

        model_response.created = int(time.time())
        model_response.model = model
        usage = Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )
        setattr(model_response, "usage", usage)
        final_outputs.append(model_response)
    return final_outputs


def embedding():
    # logic for parsing in - calling - parsing out model embedding calls
    pass
