from collections.abc import Coroutine, Iterator, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from concurrent.futures import Future as _Future
from types import MappingProxyType
from typing import Final

import litellm
from litellm._logging import print_verbose
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
from litellm.types.utils import ModelResponse
from litellm.utils import get_optional_params

from ..llms.vllm.completion import handler as vllm_handler


def batch_completion(
    model: str,
    # Optional OpenAI params: see https://platform.openai.com/docs/api-reference/chat/create
    messages: list = [],
    functions: Sequence[object] | None = None,
    function_call: str | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    n: int | None = None,
    stream: bool | None = None,
    stop: object = None,
    max_tokens: int | None = None,
    presence_penalty: float | None = None,
    frequency_penalty: float | None = None,
    logit_bias: Mapping[str, object] | None = None,
    user: str | None = None,
    deployment_id: object = None,
    request_timeout: int | None = None,
    timeout: int | None = 600,
    max_workers: int | None = 100,
    # Optional liteLLM function params
    **kwargs: object,
):
    """
    Batch litellm.completion function for a given model.

    Args:
        model (str): The model to use for generating completions.
        messages (List, optional): List of messages to use as input for generating completions. Defaults to [].
        functions (List, optional): List of functions to use as input for generating completions. Defaults to [].
        function_call (str, optional): The function call to use as input for generating completions. Defaults to "".
        temperature (float, optional): The temperature parameter for generating completions. Defaults to None.
        top_p (float, optional): The top-p parameter for generating completions. Defaults to None.
        n (int, optional): The number of completions to generate. Defaults to None.
        stream (bool, optional): Whether to stream completions or not. Defaults to None.
        stop (optional): The stop parameter for generating completions. Defaults to None.
        max_tokens (float, optional): The maximum number of tokens to generate. Defaults to None.
        presence_penalty (float, optional): The presence penalty for generating completions. Defaults to None.
        frequency_penalty (float, optional): The frequency penalty for generating completions. Defaults to None.
        logit_bias (dict, optional): The logit bias for generating completions. Defaults to {}.
        user (str, optional): The user string for generating completions. Defaults to "".
        deployment_id (optional): The deployment ID for generating completions. Defaults to None.
        request_timeout (int, optional): The request timeout for generating completions. Defaults to None.
        max_workers (int,optional): The maximum number of threads to use for parallel processing.

    Returns:
        list: A list of completion results.
    """
    args: Final = MappingProxyType[str, object](locals())

    batch_messages: Final = messages
    completions: Final[
        list[
            _Future[
                ModelResponse | CustomStreamWrapper | Coroutine[object, object, ModelResponse | CustomStreamWrapper]
            ]
        ]
    ] = []
    model = model
    custom_llm_provider = None
    if model.split("/", 1)[0] in litellm.provider_list:
        custom_llm_provider = model.split("/", 1)[0]
        model = model.split("/", 1)[1]
    if custom_llm_provider == "vllm":
        optional_params: Final = get_optional_params(
            functions=functions,
            function_call=function_call,
            temperature=temperature,
            top_p=top_p,
            n=n,
            stream=stream or False,
            stop=stop,
            max_tokens=max_tokens,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            logit_bias=logit_bias,
            user=user,
            # params to identify the model
            model=model,
            custom_llm_provider=custom_llm_provider,
        )
        results = vllm_handler.batch_completions(
            model=model,
            messages=batch_messages,
            custom_prompt_dict=litellm.custom_prompt_dict,
            optional_params=optional_params,
        )
    # all non VLLM models for batch completion models
    else:

        def chunks(lst: Sequence[object], n: int) -> Iterator[Sequence[object]]:
            """Yield successive n-sized chunks from lst."""
            for i in range(0, len(lst), n):
                yield lst[i : i + n]

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for sub_batch in chunks(batch_messages, 100):
                for message_list in sub_batch:
                    kwargs_modified = dict(args)
                    kwargs_modified.pop("max_workers")
                    kwargs_modified["messages"] = message_list
                    del kwargs_modified["kwargs"]
                    future = executor.submit(litellm.completion, **kwargs_modified, **kwargs)
                    completions.append(future)

        # Retrieve the results from the futures
        # results = [future.result() for future in completions]
        # return exceptions if any
        collected: Final[
            list[
                ModelResponse
                | CustomStreamWrapper
                | Coroutine[object, object, ModelResponse | CustomStreamWrapper]
                | Exception
            ]
        ] = []
        for future in completions:
            try:
                collected.append(future.result())
            except Exception as exc:
                collected.append(exc)
        results = collected

    return results


# send one request to multiple models
# return as soon as one of the llms responds
def batch_completion_models(*args: object, **kwargs):
    """
    Send a request to multiple language models concurrently and return the response
    as soon as one of the models responds.

    Args:
        *args: Variable-length positional arguments passed to the completion function.
        **kwargs: Additional keyword arguments:
            - models (str or list of str): The language models to send requests to.
            - Other keyword arguments to be passed to the completion function.

    Returns:
        str or None: The response from one of the language models, or None if no response is received.

    Note:
        This function utilizes a ThreadPoolExecutor to parallelize requests to multiple models.
        It sends requests concurrently and returns the response from the first model that responds.
    """

    if "model" in kwargs:
        kwargs.pop("model")
    if "models" in kwargs:
        models: Final = kwargs["models"]
        kwargs.pop("models")
        futures = {}
        with ThreadPoolExecutor(max_workers=len(models)) as executor:
            for model in models:
                futures[model] = executor.submit(litellm.completion, *args, model=model, **kwargs)

            for model, future in sorted(futures.items(), key=lambda x: models.index(x[0])):
                if future.result() is not None:
                    return future.result()
    elif "deployments" in kwargs:
        deployments: Final = kwargs["deployments"]
        kwargs.pop("deployments")
        kwargs.pop("model_list")
        nested_kwargs: Final = kwargs.pop("kwargs", {})
        futures = {}
        with ThreadPoolExecutor(max_workers=len(deployments)) as executor:
            for deployment in deployments:
                for key in kwargs:
                    if key not in deployment:  # don't override deployment values e.g. model name, api base, etc.
                        deployment[key] = kwargs[key]
                kwargs = {**deployment, **nested_kwargs}
                futures[deployment["model"]] = executor.submit(litellm.completion, **kwargs)

            while futures:
                # wait for the first returned future
                print_verbose("\n\n waiting for next result\n\n")
                done, _ = wait(futures.values(), return_when=FIRST_COMPLETED)
                print_verbose(f"done list\n{done}")
                for future in done:
                    try:
                        result = future.result()
                        return result
                    except Exception:
                        # if model 1 fails, continue with response from model 2, model3
                        print_verbose("\n\ngot an exception, ignoring, removing from futures")
                        print_verbose(futures)
                        new_futures = {}
                        for key, value in futures.items():
                            if future == value:
                                print_verbose(f"removing key{key}")
                                continue
                            else:
                                new_futures[key] = value
                        futures = new_futures
                        print_verbose(f"new futures{futures}")
                        continue

                print_verbose("\n\ndone looping through futures\n\n")
                print_verbose(futures)

    return None  # If no response is received from any model


def batch_completion_models_all_responses(*args: object, **kwargs: object):
    """
    Send a request to multiple language models concurrently and return a list of responses
    from all models that respond.

    Args:
        *args: Variable-length positional arguments passed to the completion function.
        **kwargs: Additional keyword arguments:
            - models (str or list of str): The language models to send requests to.
            - Other keyword arguments to be passed to the completion function.

    Returns:
        list: A list of responses from the language models that responded.

    Note:
        This function utilizes a ThreadPoolExecutor to parallelize requests to multiple models.
        It sends requests concurrently and collects responses from all models that respond.
    """
    import concurrent.futures

    # ANSI escape codes for colored output

    if "model" in kwargs:
        kwargs.pop("model")
    if "models" in kwargs:
        models = kwargs.pop("models")
    else:
        raise Exception("'models' param not in kwargs")

    if isinstance(models, str):
        models = [models]
    elif isinstance(models, (list, tuple)):
        models = list(models)
    else:
        raise TypeError("'models' must be a string or list of strings")

    if len(models) == 0:
        return []

    responses: Final[
        list[ModelResponse | CustomStreamWrapper | Coroutine[object, object, ModelResponse | CustomStreamWrapper]]
    ] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(models)) as executor:
        futures: Final = [executor.submit(litellm.completion, *args, model=model, **kwargs) for model in models]

        for future in futures:
            try:
                result = future.result()
                if result is not None:
                    responses.append(result)
            except Exception as e:
                print_verbose(f"batch_completion_models_all_responses: model request failed: {e}")
                continue

    return responses
