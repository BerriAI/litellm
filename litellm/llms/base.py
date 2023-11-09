## This is a template base class to be used for adding new LLM providers via API calls
import litellm 
import requests, certifi, ssl

class BaseLLM:
    def create_client_session(self):
        if litellm.client_session: 
            session = litellm.client_session
        else: 
            session = requests.Session()
        
        return session
        
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
