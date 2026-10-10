from collections.abc import Iterable
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from typing import Final, TypeVar

import litellm
from litellm._logging import print_verbose
from litellm.utils import get_optional_params

from ..llms.vllm.completion import handler as vllm_handler

_ResponseT = TypeVar("_ResponseT")


def _first_successful_response(futures: Iterable[Future[_ResponseT]]) -> _ResponseT | None:
    for future in as_completed(futures):
        try:
            result: Final = future.result()
        except Exception as exc:
            print_verbose(f"batch_completion_models: model request failed: {exc}")
            continue
        if result is not None:
            return result
    return None


def batch_completion(
    model: str,
    # Optional OpenAI params: see https://platform.openai.com/docs/api-reference/chat/create
    messages: list = [],
    functions: list | None = None,
    function_call: str | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    n: int | None = None,
    stream: bool | None = None,
    stop=None,
    max_tokens: int | None = None,
    presence_penalty: float | None = None,
    frequency_penalty: float | None = None,
    logit_bias: dict | None = None,
    user: str | None = None,
    deployment_id=None,
    request_timeout: int | None = None,
    timeout: int | None = 600,
    max_workers: int | None = 100,
    # Optional liteLLM function params
    **kwargs,
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
    args: Final = locals()

    batch_messages: Final = messages
    completions: Final = []
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

        def chunks(lst, n):
            """Yield successive n-sized chunks from lst."""
            for i in range(0, len(lst), n):
                yield lst[i : i + n]

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for sub_batch in chunks(batch_messages, 100):
                for message_list in sub_batch:
                    kwargs_modified = args.copy()
                    kwargs_modified.pop("max_workers")
                    kwargs_modified["messages"] = message_list
                    original_kwargs = {}
                    if "kwargs" in kwargs_modified:
                        original_kwargs = kwargs_modified.pop("kwargs")
                    future = executor.submit(litellm.completion, **kwargs_modified, **original_kwargs)
                    completions.append(future)

        # Retrieve the results from the futures
        # results = [future.result() for future in completions]
        # return exceptions if any
        results = []
        for future in completions:
            try:
                results.append(future.result())
            except Exception as exc:
                results.append(exc)

    return results


# send one request to multiple models
# return as soon as one of the llms responds
def batch_completion_models(*args, **kwargs):
    """
    Send a request to multiple language models concurrently and return the response
    as soon as one of the models succeeds.

    Args:
        *args: Variable-length positional arguments passed to the completion function.
        **kwargs: Additional keyword arguments:
            - models (str or list of str): The language models to send requests to.
            - Other keyword arguments to be passed to the completion function.

    Returns:
        ModelResponse, CustomStreamWrapper or None: The first response, or None if no response is received.

    Raises:
        Exception: With models, if no request succeeds, the first failed request's exception in input order.
            With deployments, failures are ignored and None is returned if no request succeeds.

    Note:
        This function utilizes a ThreadPoolExecutor to parallelize requests to multiple models.
        Requests already running continue after the first successful response is returned.
        Streaming returns the first available stream without consuming tokens. Errors raised while
        consuming that stream are passed to the caller; they do not select another model.
    """

    if "model" in kwargs:
        kwargs.pop("model")
    if "models" in kwargs:
        models: Final = kwargs["models"]
        kwargs.pop("models")
        executor: Final = ThreadPoolExecutor(max_workers=len(models))
        try:
            futures: Final = tuple(
                executor.submit(litellm.completion, *args, model=model, **kwargs) for model in models
            )
            response: Final = _first_successful_response(futures)
            if response is None:
                for future in futures:
                    future.result()
            return response
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
    elif "deployments" in kwargs:
        deployments: Final = kwargs["deployments"]
        kwargs.pop("deployments")
        kwargs.pop("model_list")
        nested_kwargs: Final = kwargs.pop("kwargs", {})
        deployment_executor: Final = ThreadPoolExecutor(max_workers=len(deployments))
        try:
            deployment_futures: Final = tuple(
                deployment_executor.submit(litellm.completion, **{**kwargs, **deployment, **nested_kwargs})
                for deployment in deployments
            )
            return _first_successful_response(deployment_futures)
        finally:
            deployment_executor.shutdown(wait=False, cancel_futures=True)

    return None  # If no response is received from any model


def batch_completion_models_all_responses(*args, **kwargs):
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

    responses: Final = []

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
