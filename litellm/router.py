from datetime import datetime
from typing import Dict, List, Optional, Union
import random, threading, time
import litellm
import logging

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
    }]

    router = Router(model_list=model_list)
    """
    model_names: List = []
    cache_responses: bool = False
    default_cache_time_seconds: int = 1 * 60 * 60  # 1 hour

    def __init__(self,
                 model_list: Optional[list] = None,
                 redis_host: Optional[str] = None,
                 redis_port: Optional[int] = None,
                 redis_password: Optional[str] = None,
                 cache_responses: bool = False) -> None:
        if model_list:
            self.set_model_list(model_list)
            self.healthy_deployments: List = []
        ### HEALTH CHECK THREAD ### - commenting out as further testing required
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
        Returns the deployment with the shortest queue 
        """
        ### COMMENTING OUT AS IT NEEDS FURTHER TESTING
        logging.debug(f"self.healthy_deployments: {self.healthy_deployments}")
        if len(self.healthy_deployments) > 0:
            for item in self.healthy_deployments:
                if item[0]["model_name"] == model: # first one in queue will be the one with the most availability
                    return item
        else: 
            potential_deployments = []
            for item in self.model_list:
                if item["model_name"] == model:
                    potential_deployments.append(item)
            item = random.choice(potential_deployments)
            return item
        
        raise ValueError("No models available.")

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
        # call via litellm.completion()
        # return litellm.completion(**{**data, "messages": messages, "caching": self.cache_responses, **kwargs})
        # litellm.set_verbose = True
        return litellm.completion(**{**data, "messages": messages, "caching": self.cache_responses, **kwargs})



    async def acompletion(self,
                    model: str,
                    messages: List[Dict[str, str]],
                    is_retry: Optional[bool] = False,
                    is_fallback: Optional[bool] = False,
                    **kwargs):
        # pick the one that is available (lowest TPM/RPM)
        deployment = self.get_available_deployment(model=model, messages=messages)
        data = deployment["litellm_params"]
        return await litellm.acompletion(**{**data, "messages": messages, "caching": self.cache_responses, **kwargs})

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
        return await litellm.aembedding(**{**data, "input": input, "caching": self.cache_responses, **kwargs})