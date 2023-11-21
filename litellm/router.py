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
import random, threading, time
import litellm, openai
import logging, asyncio
import inspect
from openai import AsyncOpenAI

class Router:
    """
    Example usage:
    from litellm import Router
    model_list = [{
        "model_name": "gpt-3.5-turbo", # model alias 
        "litellm_params": { # params for litellm completion/embedding call
            "model": "azure/<your-deployment-name>",
            "api_key": <your-api-key>,
            "api_version": <your-api-version>,
            "api_base": <your-api-base>
        },
    }]

    router = Router(model_list=model_list)
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
                 timeout: float = 600,
                 default_litellm_params = {}, # default params for Router.chat.completion.create 
                 routing_strategy: Literal["simple-shuffle", "least-busy", "usage-based-routing"] = "simple-shuffle") -> None:

        if model_list:
            self.set_model_list(model_list)
            self.healthy_deployments: List = self.model_list
        
        self.num_retries = num_retries
        
        self.chat = litellm.Chat(params=default_litellm_params)

        self.default_litellm_params = default_litellm_params
        self.default_litellm_params["timeout"] = timeout

        self.routing_strategy = routing_strategy
        ### HEALTH CHECK THREAD ###
        if self.routing_strategy == "least-busy":
            self._start_health_check_thread()

        ### CACHING ###
        if redis_host is not None and redis_port is not None and redis_password is not None:
            cache_config = {
                    'type': 'redis',
                    'host': redis_host,
                    'port': redis_port,
                    'password': redis_password
            }
        else: # use an in-memory cache
            cache_config = {
                "type": "local"
            }
        if cache_responses:
            litellm.cache = litellm.Cache(**cache_config) # use Redis for caching completion requests
            self.cache_responses = cache_responses
        self.cache = litellm.Cache(cache_config) # use Redis for tracking load balancing
        ## USAGE TRACKING ## 
        litellm.success_callback = [self.deployment_callback]

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

    def set_model_list(self, model_list: list):
        self.model_list = model_list
        self.model_names = [m["model_name"] for m in model_list]

    def get_model_names(self):
        return self.model_names

    def get_available_deployment(self,
                               model: str,
                               messages: Optional[List[Dict[str, str]]] = None,
                               input: Optional[Union[str, List]] = None):
        """
        Returns the deployment based on routing strategy
        """
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
            potential_deployments = []
            for item in self.model_list:
                if item["model_name"] == model:
                    potential_deployments.append(item)
            item = random.choice(potential_deployments)
            return item or item[0]
        elif self.routing_strategy == "usage-based-routing": 
            return self.get_usage_based_available_deployment(model=model, messages=messages, input=input)
        
        raise ValueError("No models available.")
    
    def retry_if_rate_limit_error(self, exception):
        return isinstance(exception, openai.RateLimitError)

    def retry_if_api_error(self, exception):
        return isinstance(exception, openai.APIError)
    
    async def async_function_with_retries(self, *args, **kwargs):
        # we'll backoff exponentially with each retry
        backoff_factor = 1
        original_exception = kwargs.pop("original_exception")
        original_function = kwargs.pop("original_function")
        for current_attempt in range(self.num_retries):
            try:
                # if the function call is successful, no exception will be raised and we'll break out of the loop
                response = await original_function(*args, **kwargs)
                if inspect.iscoroutinefunction(response): # async errors are often returned as coroutines 
                    response = await response
                return response

            except openai.RateLimitError as e:
                # on RateLimitError we'll wait for an exponential time before trying again
                await asyncio.sleep(backoff_factor)

                # increase backoff factor for next run
                backoff_factor *= 2

            except openai.APIError as e:
                # on APIError we immediately retry without any wait, change this if necessary
                pass

            except Exception as e:
                # for any other exception types, don't retry
                raise e
    
    def function_with_retries(self, *args, **kwargs): 
        try:
            import tenacity
        except Exception as e:
            raise Exception(f"tenacity import failed please run `pip install tenacity`. Error{e}")
        
        retry_info = {"attempts": 0, "final_result": None}

        def after_callback(retry_state):
            retry_info["attempts"] = retry_state.attempt_number
            retry_info["final_result"] = retry_state.outcome.result()

        if 'model' not in kwargs or 'messages' not in kwargs:
            raise ValueError("'model' and 'messages' must be included as keyword arguments")
        
        try: 
            original_exception = kwargs.pop("original_exception")
            original_function = kwargs.pop("original_function")
            if isinstance(original_exception, openai.RateLimitError):
                retryer = tenacity.Retrying(wait=tenacity.wait_exponential(multiplier=1, max=10), 
                                            stop=tenacity.stop_after_attempt(self.num_retries), 
                                            reraise=True,
                                            after=after_callback)
            elif isinstance(original_exception, openai.APIError):
                retryer = tenacity.Retrying(stop=tenacity.stop_after_attempt(self.num_retries), 
                                            reraise=True,
                                            after=after_callback)
                
            return retryer(self.acompletion, *args, **kwargs)
        except Exception as e: 
            raise Exception(f"Error in function_with_retries: {e}\n\nRetry Info: {retry_info}")

    ### COMPLETION + EMBEDDING FUNCTIONS

    def completion(self,
                   model: str,
                   messages: List[Dict[str, str]],
                   is_retry: Optional[bool] = False,
                   is_fallback: Optional[bool] = False,
                   **kwargs):
        """
        Example usage:
        response = router.completion(model="gpt-3.5-turbo", messages=[{"role": "user", "content": "Hey, how's it going?"}]
        """

        # pick the one that is available (lowest TPM/RPM)
        deployment = self.get_available_deployment(model=model, messages=messages)
        data = deployment["litellm_params"]
        for k, v in self.default_litellm_params.items(): 
            if k not in data: # prioritize model-specific params > default router params 
                data[k] = v
        return litellm.completion(**{**data, "messages": messages, "caching": self.cache_responses, **kwargs})


    async def acompletion(self,
                    model: str,
                    messages: List[Dict[str, str]],
                    is_retry: Optional[bool] = False,
                    is_fallback: Optional[bool] = False,
                    **kwargs):
        try: 
            deployment = self.get_available_deployment(model=model, messages=messages)
            data = deployment["litellm_params"]
            for k, v in self.default_litellm_params.items(): 
                if k not in data: # prioritize model-specific params > default router params 
                    data[k] = v
            response = await litellm.acompletion(**{**data, "messages": messages, "caching": self.cache_responses, **kwargs})
            # client = AsyncOpenAI()
            # print(f"MAKING OPENAI CALL")
            # response = await client.chat.completions.create(model=model, messages=messages)
            return response
        except Exception as e: 
            if self.num_retries > 0:
                kwargs["model"] = model
                kwargs["messages"] = messages
                kwargs["original_exception"] = e
                kwargs["original_function"] = self.acompletion
                return await self.async_function_with_retries(**kwargs)
            else: 
                raise e

    def text_completion(self,
                        model: str,
                        prompt: str,
                        is_retry: Optional[bool] = False,
                        is_fallback: Optional[bool] = False,
                        is_async: Optional[bool] = False,
                        **kwargs):

        messages=[{"role": "user", "content": prompt}]
        # pick the one that is available (lowest TPM/RPM)
        deployment = self.get_available_deployment(model=model, messages=messages)

        data = deployment["litellm_params"]
        for k, v in self.default_litellm_params.items(): 
            if k not in data: # prioritize model-specific params > default router params 
                data[k] = v
        # call via litellm.completion()
        return litellm.text_completion(**{**data, "prompt": prompt, "caching": self.cache_responses, **kwargs}) # type: ignore

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
        tpm = self.cache.get_cache(cache_key=tpm_key) or 0
        rpm = self.cache.get_cache(cache_key=rpm_key) or 0

        return int(tpm), int(rpm)

    def increment(self, key: str, increment_value: int):
        # get value
        cached_value = self.cache.get_cache(cache_key=key)
        # update value
        try:
            cached_value = cached_value + increment_value
        except:
            cached_value = increment_value
        # save updated value
        self.cache.add_cache(result=cached_value, cache_key=key, ttl=self.default_cache_time_seconds)

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