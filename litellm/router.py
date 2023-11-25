# +-----------------------------------------------+
# |                                               |
# |           Give Feedback / Get Help            |
# | https://github.com/BerriAI/litellm/issues/new |
# |                                               |
# +-----------------------------------------------+
#
#  Thank you ! We ❤️ you! - Krrish & Ishaan 

from datetime import datetime
from typing import Dict, List, Optional, Union, Literal
import random, threading, time, traceback
import litellm, openai
from litellm.caching import RedisCache, InMemoryCache, DualCache
import logging, asyncio
import inspect, concurrent
from openai import AsyncOpenAI

class Router:
    """
    Example usage:
    ```python
    from litellm import Router
    model_list = [
    {
        "model_name": "azure-gpt-3.5-turbo", # model alias 
        "litellm_params": { # params for litellm completion/embedding call
            "model": "azure/<your-deployment-name-1>",
            "api_key": <your-api-key>,
            "api_version": <your-api-version>,
            "api_base": <your-api-base>
        },
    },
    {
        "model_name": "azure-gpt-3.5-turbo", # model alias 
        "litellm_params": { # params for litellm completion/embedding call
            "model": "azure/<your-deployment-name-2>",
            "api_key": <your-api-key>,
            "api_version": <your-api-version>,
            "api_base": <your-api-base>
        },
    },
    {
        "model_name": "openai-gpt-3.5-turbo", # model alias 
        "litellm_params": { # params for litellm completion/embedding call
            "model": "gpt-3.5-turbo",
            "api_key": <your-api-key>,
        },
    ]

    router = Router(model_list=model_list, fallbacks=[{"azure-gpt-3.5-turbo": "openai-gpt-3.5-turbo"}])
    ```
    """
    model_names: List = []
    cache_responses: bool = False
    default_cache_time_seconds: int = 1 * 60 * 60  # 1 hour
    num_retries: int = 0
    tenacity = None

    def __init__(self,
                 model_list: Optional[list] = None,
                 redis_host: Optional[str] = None,
                 redis_port: Optional[int] = None,
                 redis_password: Optional[str] = None,
                 cache_responses: bool = False,
                 num_retries: int = 0,
                 timeout: Optional[float] = None,
                 default_litellm_params = {}, # default params for Router.chat.completion.create 
                 set_verbose: bool = False,
                 fallbacks: List = [],
                 allowed_fails: Optional[int] = None,
                 context_window_fallbacks: List = [], 
                 routing_strategy: Literal["simple-shuffle", "least-busy", "usage-based-routing", "latency-based-routing"] = "simple-shuffle") -> None:

        if model_list:
            self.set_model_list(model_list)
            self.healthy_deployments: List = self.model_list
            self.deployment_latency_map = {}
            for m in model_list: 
                self.deployment_latency_map[m["litellm_params"]["model"]] = 0
        
        self.allowed_fails = allowed_fails or litellm.allowed_fails
        self.failed_calls = InMemoryCache() # cache to track failed call per deployment, if num failed calls within 1 minute > allowed fails, then add it to cooldown
        self.num_retries = num_retries or litellm.num_retries or 0
        self.set_verbose = set_verbose 
        self.timeout = timeout or litellm.request_timeout
        self.routing_strategy = routing_strategy
        self.fallbacks = fallbacks or litellm.fallbacks
        self.context_window_fallbacks = context_window_fallbacks or litellm.context_window_fallbacks

        # make Router.chat.completions.create compatible for openai.chat.completions.create
        self.chat = litellm.Chat(params=default_litellm_params)

        # default litellm args
        self.default_litellm_params = default_litellm_params
        self.default_litellm_params["timeout"] = timeout

        
        ### HEALTH CHECK THREAD ###
        if self.routing_strategy == "least-busy":
            self._start_health_check_thread()

        ### CACHING ###
        redis_cache = None
        if redis_host is not None and redis_port is not None and redis_password is not None:
            cache_config = {
                    'type': 'redis',
                    'host': redis_host,
                    'port': redis_port,
                    'password': redis_password
            }
            redis_cache = RedisCache(host=redis_host, port=redis_port, password=redis_password)
        else: # use an in-memory cache
            cache_config = {
                "type": "local"
            }
        if cache_responses:
            litellm.cache = litellm.Cache(**cache_config) # use Redis for caching completion requests
            self.cache_responses = cache_responses
        self.cache = DualCache(redis_cache=redis_cache, in_memory_cache=InMemoryCache()) # use a dual cache (Redis+In-Memory) for tracking cooldowns, usage, etc.
        ## USAGE TRACKING ## 
        if isinstance(litellm.success_callback, list):
            litellm.success_callback.append(self.deployment_callback)
        else:
            litellm.success_callback = [self.deployment_callback]
        
        if isinstance(litellm.failure_callback, list):
            litellm.failure_callback.append(self.deployment_callback_on_failure)
        else:
            litellm.failure_callback = [self.deployment_callback_on_failure]

    
    ### COMPLETION + EMBEDDING FUNCTIONS

    def completion(self,
                   model: str,
                   messages: List[Dict[str, str]],
                   **kwargs):
        """
        Example usage:
        response = router.completion(model="gpt-3.5-turbo", messages=[{"role": "user", "content": "Hey, how's it going?"}]
        """
        try: 
            kwargs["model"] = model
            kwargs["messages"] = messages
            kwargs["original_function"] = self._completion
            kwargs["num_retries"] = self.num_retries
            kwargs.setdefault("metadata", {}).update({"model_group": model})
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                # Submit the function to the executor with a timeout
                future = executor.submit(self.function_with_fallbacks, **kwargs)
                response = future.result(timeout=self.timeout) # type: ignore

            return response
        except Exception as e: 
            raise e

    def _completion(
            self, 
            model: str, 
            messages: List[Dict[str, str]], 
            **kwargs):
        
        try: 
            # pick the one that is available (lowest TPM/RPM)
            deployment = self.get_available_deployment(model=model, messages=messages)
            data = deployment["litellm_params"]
            for k, v in self.default_litellm_params.items(): 
                if k not in data: # prioritize model-specific params > default router params 
                    data[k] = v
            
            self.print_verbose(f"completion model: {data['model']}")
            return litellm.completion(**{**data, "messages": messages, "caching": self.cache_responses, **kwargs})
        except Exception as e: 
            raise e
    
    async def acompletion(self,
                model: str,
                messages: List[Dict[str, str]],
                **kwargs):
        try: 
            kwargs["model"] = model
            kwargs["messages"] = messages
            kwargs["original_function"] = self._acompletion
            kwargs["num_retries"] = self.num_retries
            kwargs.setdefault("metadata", {}).update({"model_group": model})
            response = await asyncio.wait_for(self.async_function_with_fallbacks(**kwargs), timeout=self.timeout)

            return response
        except Exception as e: 
            raise e

    async def _acompletion(
            self, 
            model: str, 
            messages: List[Dict[str, str]],
            **kwargs):
        try: 
            deployment = self.get_available_deployment(model=model, messages=messages)
            data = deployment["litellm_params"]
            for k, v in self.default_litellm_params.items(): 
                if k not in data: # prioritize model-specific params > default router params 
                    data[k] = v
            self.print_verbose(f"acompletion model: {data['model']}")
            
            response = await litellm.acompletion(**{**data, "messages": messages, "caching": self.cache_responses, **kwargs})
            return response
        except Exception as e: 
            raise e
       
    def text_completion(self,
                        model: str,
                        prompt: str,
                        is_retry: Optional[bool] = False,
                        is_fallback: Optional[bool] = False,
                        is_async: Optional[bool] = False,
                        **kwargs):
        try: 
            kwargs.setdefault("metadata", {}).update({"model_group": model})
            messages=[{"role": "user", "content": prompt}]
            # pick the one that is available (lowest TPM/RPM)
            deployment = self.get_available_deployment(model=model, messages=messages)

            data = deployment["litellm_params"]
            for k, v in self.default_litellm_params.items(): 
                if k not in data: # prioritize model-specific params > default router params 
                    data[k] = v
            # call via litellm.completion()
            return litellm.text_completion(**{**data, "prompt": prompt, "caching": self.cache_responses, **kwargs}) # type: ignore
        except Exception as e: 
            if self.num_retries > 0:
                kwargs["model"] = model
                kwargs["messages"] = messages
                kwargs["original_exception"] = e
                kwargs["original_function"] = self.completion
                return self.function_with_retries(**kwargs)
            else: 
                raise e

    def embedding(self,
                  model: str,
                  input: Union[str, List],
                  is_async: Optional[bool] = False,
                  **kwargs) -> Union[List[float], None]:
        # pick the one that is available (lowest TPM/RPM)
        deployment = self.get_available_deployment(model=model, input=input)

        data = deployment["litellm_params"]
        for k, v in self.default_litellm_params.items(): 
            if k not in data: # prioritize model-specific params > default router params 
                data[k] = v
        # call via litellm.embedding()
        return litellm.embedding(**{**data, "input": input, "caching": self.cache_responses, **kwargs})

    async def aembedding(self,
                         model: str,
                         input: Union[str, List],
                         is_async: Optional[bool] = True,
                         **kwargs) -> Union[List[float], None]:
        # pick the one that is available (lowest TPM/RPM)
        deployment = self.get_available_deployment(model=model, input=input)

        data = deployment["litellm_params"]
        for k, v in self.default_litellm_params.items(): 
            if k not in data: # prioritize model-specific params > default router params 
                data[k] = v
        return await litellm.aembedding(**{**data, "input": input, "caching": self.cache_responses, **kwargs})

    async def async_function_with_fallbacks(self, *args, **kwargs): 
        """
        Try calling the function_with_retries
        If it fails after num_retries, fall back to another model group
        """
        model_group = kwargs.get("model")
        try: 
            response = await self.async_function_with_retries(*args, **kwargs)
            self.print_verbose(f'Async Response: {response}')
            return response
        except Exception as e: 
            self.print_verbose(f"An exception occurs")
            original_exception = e
            try: 
                self.print_verbose(f"Trying to fallback b/w models")
                if isinstance(e, litellm.ContextWindowExceededError) and self.context_window_fallbacks is not None: 
                    fallback_model_group = None
                    for item in self.context_window_fallbacks: # [{"gpt-3.5-turbo": ["gpt-4"]}]
                        if list(item.keys())[0] == model_group:
                            fallback_model_group = item[model_group]
                            break
                    
                    if fallback_model_group is None: 
                        raise original_exception
                    
                    for mg in fallback_model_group: 
                        """
                        Iterate through the model groups and try calling that deployment
                        """
                        try:
                            kwargs["model"] = mg
                            response = await self.async_function_with_retries(*args, **kwargs)
                            return response 
                        except Exception as e: 
                            pass
                else: 
                    if self.fallbacks is None: 
                        raise original_exception
                    self.print_verbose(f"inside model fallbacks: {self.fallbacks}")
                    for item in self.fallbacks:
                        if list(item.keys())[0] == model_group:
                            fallback_model_group = item[model_group]
                            break
                    for mg in fallback_model_group: 
                        """
                        Iterate through the model groups and try calling that deployment
                        """
                        try:
                            kwargs["model"] = mg
                            response = await self.async_function_with_retries(*args, **kwargs)
                            return response 
                        except Exception as e: 
                            pass
            except Exception as e: 
                self.print_verbose(f"An exception occurred - {str(e)}")
                traceback.print_exc()
            raise original_exception
        
    async def async_function_with_retries(self, *args, **kwargs):
        self.print_verbose(f"Inside async function with retries: args - {args}; kwargs - {kwargs}")
        backoff_factor = 1
        original_function = kwargs.pop("original_function")
        num_retries = kwargs.pop("num_retries")
        try: 
            # if the function call is successful, no exception will be raised and we'll break out of the loop
            response = await original_function(*args, **kwargs)
            return response
        except Exception as e: 
            for current_attempt in range(num_retries):
                self.print_verbose(f"retrying request. Current attempt - {current_attempt}; num retries: {num_retries}")
                try:
                    # if the function call is successful, no exception will be raised and we'll break out of the loop
                    response = await original_function(*args, **kwargs)
                    if inspect.iscoroutinefunction(response): # async errors are often returned as coroutines 
                        response = await response
                    return response

                except openai.RateLimitError as e:
                    if num_retries > 0: 
                        # on RateLimitError we'll wait for an exponential time before trying again
                        await asyncio.sleep(backoff_factor)

                        # increase backoff factor for next run
                        backoff_factor *= 2
                    else: 
                        raise e

                except Exception as e:
                    # for any other exception types, immediately retry
                    if num_retries > 0: 
                        pass
                    else: 
                        raise e
            raise e
    
    def function_with_fallbacks(self, *args, **kwargs): 
        """
        Try calling the function_with_retries
        If it fails after num_retries, fall back to another model group
        """
        model_group = kwargs.get("model")
        
        try: 
            response = self.function_with_retries(*args, **kwargs)
            return response
        except Exception as e: 
            original_exception = e
            self.print_verbose(f"An exception occurs {original_exception}")
            try: 
                self.print_verbose(f"Trying to fallback b/w models. Initial model group: {model_group}")
                if isinstance(e, litellm.ContextWindowExceededError) and self.context_window_fallbacks is not None: 
                    self.print_verbose(f"inside context window fallbacks: {self.context_window_fallbacks}")
                    fallback_model_group = None

                    for item in self.context_window_fallbacks: # [{"gpt-3.5-turbo": ["gpt-4"]}]
                        if list(item.keys())[0] == model_group:
                            fallback_model_group = item[model_group]
                            break
                    
                    if fallback_model_group is None: 
                        raise original_exception
                    
                    for mg in fallback_model_group: 
                        """
                        Iterate through the model groups and try calling that deployment
                        """
                        try:
                            kwargs["model"] = mg
                            response = self.function_with_fallbacks(*args, **kwargs)
                            return response 
                        except Exception as e: 
                            pass
                else: 
                    if self.fallbacks is None: 
                        raise original_exception

                    self.print_verbose(f"inside model fallbacks: {self.fallbacks}")
                    fallback_model_group = None
                    for item in self.fallbacks:
                        if list(item.keys())[0] == model_group:
                            fallback_model_group = item[model_group]
                            break

                    if fallback_model_group is None: 
                        raise original_exception
                    
                    for mg in fallback_model_group: 
                        """
                        Iterate through the model groups and try calling that deployment
                        """
                        try:
                            kwargs["model"] = mg
                            response = self.function_with_fallbacks(*args, **kwargs)
                            return response 
                        except Exception as e: 
                            pass
            except Exception as e: 
                raise e
            raise original_exception
            
    def function_with_retries(self, *args, **kwargs): 
        """
        Try calling the model 3 times. Shuffle between available deployments. 
        """
        self.print_verbose(f"Inside function with retries: args - {args}; kwargs - {kwargs}")
        backoff_factor = 1
        original_function = kwargs.pop("original_function")
        num_retries = kwargs.pop("num_retries")
        try: 
            # if the function call is successful, no exception will be raised and we'll break out of the loop
            response = original_function(*args, **kwargs)
            return response
        except Exception as e: 
            original_exception = e
            self.print_verbose(f"num retries in function with retries: {num_retries}")
            for current_attempt in range(num_retries):
                self.print_verbose(f"retrying request. Current attempt - {current_attempt}; retries left: {num_retries}")
                try:
                    # if the function call is successful, no exception will be raised and we'll break out of the loop
                    response = original_function(*args, **kwargs)
                    return response

                except openai.RateLimitError as e:
                    if num_retries > 0: 
                        # on RateLimitError we'll wait for an exponential time before trying again
                        time.sleep(backoff_factor)

                        # increase backoff factor for next run
                        backoff_factor *= 2
                    else: 
                        raise e
                    
                except Exception as e:
                    # for any other exception types, immediately retry
                    if num_retries > 0: 
                        pass
                    else: 
                        raise e
            raise original_exception

    ### HELPER FUNCTIONS
    
    def deployment_callback(
        self,
        kwargs,                 # kwargs to completion
        completion_response,    # response from completion
        start_time, end_time    # start/end time
    ):
        """
        Function LiteLLM submits a callback to after a successful
        completion. Purpose of this is to update TPM/RPM usage per model
        """
        model_name = kwargs.get('model', None)  # i.e. gpt35turbo
        custom_llm_provider = kwargs.get("litellm_params", {}).get('custom_llm_provider', None)  # i.e. azure
        if custom_llm_provider:
            model_name = f"{custom_llm_provider}/{model_name}"
        if kwargs["stream"] is True: 
            if kwargs.get("complete_streaming_response"):
                total_tokens = kwargs.get("complete_streaming_response")['usage']['total_tokens']
                self._set_deployment_usage(model_name, total_tokens)
        else: 
            total_tokens = completion_response['usage']['total_tokens']
            self._set_deployment_usage(model_name, total_tokens)
        
        self.deployment_latency_map[model_name] = (end_time - start_time).total_seconds()

    def deployment_callback_on_failure(
            self,
            kwargs,                 # kwargs to completion
            completion_response,    # response from completion
            start_time, end_time    # start/end time
    ):
        try: 
            model_name = kwargs.get('model', None)  # i.e. gpt35turbo
            custom_llm_provider = kwargs.get("litellm_params", {}).get('custom_llm_provider', None)  # i.e. azure
            if custom_llm_provider:
                model_name = f"{custom_llm_provider}/{model_name}"
            
            self._set_cooldown_deployments(model_name)
        except Exception as e:
            raise e

    def _set_cooldown_deployments(self, 
                                  deployment: str):
        """
        Add a model to the list of models being cooled down for that minute, if it exceeds the allowed fails / minute
        """
        
        current_minute = datetime.now().strftime("%H-%M")
        # get current fails for deployment
        # update the number of failed calls 
        # if it's > allowed fails 
        # cooldown deployment 
        current_fails = self.failed_calls.get_cache(key=deployment) or 0
        updated_fails = current_fails + 1
        self.print_verbose(f"updated_fails: {updated_fails}; self.allowed_fails: {self.allowed_fails}")
        if updated_fails > self.allowed_fails:                
            # get the current cooldown list for that minute
            cooldown_key = f"{current_minute}:cooldown_models" # group cooldown models by minute to reduce number of redis calls
            cached_value = self.cache.get_cache(key=cooldown_key)

            self.print_verbose(f"adding {deployment} to cooldown models")
            # update value
            try:
                if deployment in cached_value: 
                    pass
                else: 
                    cached_value = cached_value + [deployment]
                    # save updated value
                    self.cache.set_cache(value=cached_value, key=cooldown_key, ttl=60) 
            except:
                cached_value = [deployment]
                # save updated value
                self.cache.set_cache(value=cached_value, key=cooldown_key, ttl=60) 
        else:
            self.failed_calls.set_cache(key=deployment, value=updated_fails, ttl=60) 


    def _get_cooldown_deployments(self):
        """
        Get the list of models being cooled down for this minute
        """
        current_minute = datetime.now().strftime("%H-%M")
        # get the current cooldown list for that minute
        cooldown_key = f"{current_minute}:cooldown_models"

        # ----------------------
        # Return cooldown models
        # ----------------------
        cooldown_models = self.cache.get_cache(key=cooldown_key) or []

        self.print_verbose(f"retrieve cooldown models: {cooldown_models}")
        return cooldown_models

    def get_usage_based_available_deployment(self,
                               model: str,
                               messages: Optional[List[Dict[str, str]]] = None,
                               input: Optional[Union[str, List]] = None):
        """
        Returns a deployment with the lowest TPM/RPM usage.
        """
        # get list of potential deployments
        potential_deployments = []
        for item in self.model_list:
            if item["model_name"] == model:
                potential_deployments.append(item)

        # get current call usage
        token_count = 0
        if messages is not None:
            token_count = litellm.token_counter(model=model, messages=messages)
        elif input is not None:
            if isinstance(input, List):
                input_text = "".join(text for text in input)
            else:
                input_text = input
            token_count = litellm.token_counter(model=model, text=input_text)

        # -----------------------
        # Find lowest used model
        # ----------------------
        lowest_tpm = float("inf")
        deployment = None

        # return deployment with lowest tpm usage
        for item in potential_deployments:
            item_tpm, item_rpm = self._get_deployment_usage(deployment_name=item["litellm_params"]["model"])

            if item_tpm == 0:
                return item
            elif ("tpm" in item and item_tpm + token_count > item["tpm"]
                  or "rpm" in item and item_rpm + 1 >= item["rpm"]): # if user passed in tpm / rpm in the model_list
                continue
            elif item_tpm < lowest_tpm:
                lowest_tpm = item_tpm
                deployment = item

        # if none, raise exception
        if deployment is None:
            raise ValueError("No models available.")

        # return model
        return deployment

    def _get_deployment_usage(
        self,
        deployment_name: str
    ):
        # ------------
        # Setup values
        # ------------
        current_minute = datetime.now().strftime("%H-%M")
        tpm_key = f'{deployment_name}:tpm:{current_minute}'
        rpm_key = f'{deployment_name}:rpm:{current_minute}'

        # ------------
        # Return usage
        # ------------
        tpm = self.cache.get_cache(key=tpm_key) or 0
        rpm = self.cache.get_cache(key=rpm_key) or 0

        return int(tpm), int(rpm)

    def increment(self, key: str, increment_value: int):
        # get value
        cached_value = self.cache.get_cache(key=key)
        # update value
        try:
            cached_value = cached_value + increment_value
        except:
            cached_value = increment_value
        # save updated value
        self.cache.set_cache(value=cached_value, key=key, ttl=self.default_cache_time_seconds)

    def _set_deployment_usage(
        self,
        model_name: str,
        total_tokens: int
    ):
        # ------------
        # Setup values
        # ------------
        current_minute = datetime.now().strftime("%H-%M")
        tpm_key = f'{model_name}:tpm:{current_minute}'
        rpm_key = f'{model_name}:rpm:{current_minute}'

        # ------------
        # Update usage
        # ------------
        self.increment(tpm_key, total_tokens)
        self.increment(rpm_key, 1)
    
    def _start_health_check_thread(self):
        """
        Starts a separate thread to perform health checks periodically.
        """
        health_check_thread = threading.Thread(target=self._perform_health_checks, daemon=True)
        health_check_thread.start()
    
    def _perform_health_checks(self):
        """
        Periodically performs health checks on the servers.
        Updates the list of healthy servers accordingly.
        """
        while True:
            self.healthy_deployments = self._health_check()
            # Adjust the time interval based on your needs
            time.sleep(15)
        
    def _health_check(self):
        """
        Performs a health check on the deployments
        Returns the list of healthy deployments
        """
        healthy_deployments = []
        for deployment in self.model_list: 
            litellm_args = deployment["litellm_params"]
            try: 
                start_time = time.time()
                litellm.completion(messages=[{"role": "user", "content": ""}], max_tokens=1, **litellm_args) # hit the server with a blank message to see how long it takes to respond
                end_time = time.time() 
                response_time = end_time - start_time
                logging.debug(f"response_time: {response_time}")
                healthy_deployments.append((deployment, response_time))
                healthy_deployments.sort(key=lambda x: x[1])
            except Exception as e: 
                pass
        return healthy_deployments
    
    def weighted_shuffle_by_latency(self, items):
        # Sort the items by latency
        sorted_items = sorted(items, key=lambda x: x[1])
        # Get only the latencies
        latencies = [i[1] for i in sorted_items]
        # Calculate the sum of all latencies
        total_latency = sum(latencies)
        # Calculate the weight for each latency (lower latency = higher weight)
        weights = [total_latency-latency for latency in latencies]
        # Get a weighted random item
        if sum(weights) == 0: 
            chosen_item = random.choice(sorted_items)[0]
        else: 
            chosen_item = random.choices(sorted_items, weights=weights, k=1)[0][0]
        return chosen_item

    def set_model_list(self, model_list: list):
        self.model_list = model_list
        self.model_names = [m["model_name"] for m in model_list]

    def get_model_names(self):
        return self.model_names

    def print_verbose(self, print_statement): 
        if self.set_verbose or litellm.set_verbose: 
            print(f"LiteLLM.Router: {print_statement}") # noqa

    def get_available_deployment(self,
                               model: str,
                               messages: Optional[List[Dict[str, str]]] = None,
                               input: Optional[Union[str, List]] = None):
        """
        Returns the deployment based on routing strategy
        """
        ## get healthy deployments
        ### get all deployments 
        ### filter out the deployments currently cooling down 
        healthy_deployments = [m for m in self.model_list if m["model_name"] == model]
        deployments_to_remove = [] 
        cooldown_deployments = self._get_cooldown_deployments()
        self.print_verbose(f"cooldown deployments: {cooldown_deployments}")
        ### FIND UNHEALTHY DEPLOYMENTS
        for deployment in healthy_deployments: 
            deployment_name = deployment["litellm_params"]["model"]
            if deployment_name in cooldown_deployments: 
                deployments_to_remove.append(deployment)
        ### FILTER OUT UNHEALTHY DEPLOYMENTS
        for deployment in deployments_to_remove:
            healthy_deployments.remove(deployment)
        self.print_verbose(f"healthy deployments: {healthy_deployments}")
        if len(healthy_deployments) == 0: 
            raise ValueError("No models available")
        if litellm.model_alias_map and model in litellm.model_alias_map:
            model = litellm.model_alias_map[
                model
            ]  # update the model to the actual value if an alias has been passed in
        if self.routing_strategy == "least-busy":
            if len(self.healthy_deployments) > 0:
                for item in self.healthy_deployments:
                    if item[0]["model_name"] == model: # first one in queue will be the one with the most availability
                        return item[0]
            else: 
                raise ValueError("No models available.")
        elif self.routing_strategy == "simple-shuffle": 
            item = random.choice(healthy_deployments)
            return item or item[0]
        elif self.routing_strategy == "latency-based-routing": 
            returned_item = None
            lowest_latency = float('inf')
            ### shuffles with priority for lowest latency
            # items_with_latencies = [('A', 10), ('B', 20), ('C', 30), ('D', 40)]
            items_with_latencies = [] 
            for item in healthy_deployments:
                items_with_latencies.append((item, self.deployment_latency_map[item["litellm_params"]["model"]]))
            returned_item = self.weighted_shuffle_by_latency(items_with_latencies)
            return returned_item
        elif self.routing_strategy == "usage-based-routing": 
            return self.get_usage_based_available_deployment(model=model, messages=messages, input=input)
        
        raise ValueError("No models available.")
    
    def flush_cache(self):
        self.cache.flush_cache()
    
    def reset(self): 
        ## clean up on close
        litellm.success_callback = [] 
        litellm.failure_callback = [] 
        self.flush_cache() 
        