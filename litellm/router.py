from typing import Union, List, Dict, Optional
from datetime import datetime
import litellm


class Router: 
    """
    Example usage:
    from litellm import Router
    model_list = [{
        "model_name": "gpt-3.5-turbo", # openai model name 
        "litellm_params": { # params for litellm completion/embedding call 
            "model": "azure/<your-deployment-name>", 
            "api_key": <your-api-key>,
            "api_version": <your-api-version>,
            "api_base": <your-api-base>
        },
        "tpm": <your-model-tpm>, e.g. 240000
        "rpm": <your-model-rpm>, e.g. 1800
    }]

    router = Router(model_list=model_list)
    """
    model_names: List = []
    cache_responses: bool = False
    def __init__(self, 
                 model_list: Optional[list]=None,
                 redis_host: Optional[str] = None,
                 redis_port: Optional[int] = None,
                 redis_password: Optional[str] = None, 
                 cache_responses: bool = False) -> None:
        if model_list:
            self.model_list = model_list
            self.model_names = [m["model_name"] for m in model_list]
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
        self.cache = litellm.Cache(cache_config) # use Redis for tracking load balancing
        if cache_responses:
            litellm.cache = litellm.Cache(**cache_config) # use Redis for caching completion requests 
            self.cache_responses = cache_responses
        litellm.success_callback = [self.deployment_callback]
    
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
        data["messages"] = messages
        data["caching"] = self.cache_responses
        # call via litellm.completion() 
        return litellm.completion(**{**data, **kwargs})

    async def acompletion(self, 
                    model: str, 
                    messages: List[Dict[str, str]], 
                    is_retry: Optional[bool] = False,
                    is_fallback: Optional[bool] = False,
                    **kwargs):
        # pick the one that is available (lowest TPM/RPM)
        deployment = self.get_available_deployment(model=model, messages=messages)
        data = deployment["litellm_params"]
        data["messages"] = messages
        data["caching"] = self.cache_responses
        return await litellm.acompletion(**{**data, **kwargs})
    
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
        data["prompt"] = prompt
        data["caching"] = self.cache_responses
        # call via litellm.completion() 
        return litellm.text_completion(**{**data, **kwargs})        

    def embedding(self, 
                  model: str,
                  input: Union[str, List],
                  is_async: Optional[bool] = False,
                  **kwargs) -> Union[List[float], None]:
        # pick the one that is available (lowest TPM/RPM)
        deployment = self.get_available_deployment(model=model, input=input)

        data = deployment["litellm_params"]
        data["input"] = input
        data["caching"] = self.cache_responses
        # call via litellm.embedding() 
        return litellm.embedding(**{**data, **kwargs})

    async def aembedding(self,
                         model: str,
                         input: Union[str, List],
                         is_async: Optional[bool] = True,
                         **kwargs) -> Union[List[float], None]:
        # pick the one that is available (lowest TPM/RPM)
        deployment = self.get_available_deployment(model=model, input=input)

        data = deployment["litellm_params"]
        data["input"] = input
        data["caching"] = self.cache_responses
        return await litellm.aembedding(**{**data, **kwargs})

    def set_model_list(self, model_list: list):
        self.model_list = model_list

    def get_model_names(self): 
        return self.model_names

    def deployment_callback(
        self,
        kwargs,                 # kwargs to completion
        completion_response,    # response from completion
        start_time, end_time    # start/end time
    ):
        """
        Function LiteLLM submits a callback to after a successful
        completion. Purpose of this is ti update TPM/RPM usage per model
        """
        model_name = kwargs.get('model', None)  # i.e. azure/gpt35turbo
        total_tokens = completion_response['usage']['total_tokens']
        self._set_deployment_usage(model_name, total_tokens)

    def get_available_deployment(self, 
                               model: str, 
                               messages: Optional[List[Dict[str, str]]]=None,
                               input: Optional[Union[str, List]]=None): 
        """
        Returns a deployment with the lowest TPM/RPM usage.
        """
        # get list of potential deployments 
        potential_deployments = [] 
        for item in self.model_list: 
            if item["model_name"] == model: 
                potential_deployments.append(item)
        
        # set first model as current model
        deployment = potential_deployments[0] 


        # get model tpm, rpm limits
        tpm = deployment["tpm"]
        rpm = deployment["rpm"]

        # get deployment current usage
        current_tpm, current_rpm = self._get_deployment_usage(deployment_name=deployment["litellm_params"]["model"])

        # get encoding 
        if messages:
            token_count = litellm.token_counter(model=deployment["model_name"], messages=messages)
        elif input:
            if isinstance(input, List):
                input_text = "".join(text for text in input)
            else:
                input_text = input
            token_count = litellm.token_counter(model=deployment["model_name"], text=input_text)
        
        # if at model limit, return lowest used
        if current_tpm + token_count > tpm or current_rpm + 1 >= rpm: 
            # -----------------------
            # Find lowest used model
            # ----------------------
            lowest_tpm = float('inf')
            deployment = None

            # Go through all the models to get tpm, rpm
            for item in potential_deployments:
                item_tpm, item_rpm = self._get_deployment_usage(deployment_name=item["litellm_params"]["model"])

                if item_tpm == 0:
                    return item
                elif item_tpm + token_count > item["tpm"] or item_rpm + 1 >= item["rpm"]: 
                    continue
                elif item_tpm < lowest_tpm:
                    lowest_tpm = item_tpm
                    deployment = item
        
            # if none, raise exception 
            if deployment is None: 
                raise ValueError(f"No models available.")

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
        tpm = self.cache.get_cache(tpm_key)
        rpm = self.cache.get_cache(rpm_key)

        if tpm is None: 
            tpm = 0
        if rpm is None: 
            rpm = 0

        return int(tpm), int(rpm)
    
    def increment(self, key: str, increment_value: int): 
        # get value 
        cached_value = self.cache.get_cache(key)
        # update value 
        try:
            cached_value = cached_value + increment_value
        except: 
            cached_value = increment_value
        # save updated value
        self.cache.add_cache(result=cached_value, cache_key=key)
    
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
