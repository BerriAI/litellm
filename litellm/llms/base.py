## This is a template base class to be used for adding new LLM providers via API calls
import litellm 
import httpx, certifi, ssl
from typing import Optional

class BaseLLM:
    _client_session: Optional[httpx.Client] = None
    def create_client_session(self):
        if litellm.client_session: 
            _client_session = litellm.client_session
        else: 
            _client_session = httpx.Client(timeout=600)
        
        return _client_session
    
    def __exit__(self):
        if hasattr(self, '_client_session'):
            self._client_session.close()

    def validate_environment(self):  # set up the environment required to run the model
        pass

    def completion(
        self,
        *args, 
        **kwargs
    ):  # logic for parsing in - calling - parsing out model completion calls
        pass

    def embedding(
        self,
        *args, 
        **kwargs
    ):  # logic for parsing in - calling - parsing out model embedding calls
        pass
