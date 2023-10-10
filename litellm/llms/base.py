## This is a template base class to be used for adding new LLM providers via API calls
import litellm 
import requests, certifi, ssl

class BaseLLM:
    def create_client_session(self):
        if litellm.verify_ssl is False:
            session = requests.Session()
            session.verify = False
        else:
            ca_bundle_path = certifi.where() if litellm.ca_bundle_path is None else litellm.ca_bundle_path
            session = requests.Session()
            session.verify = ca_bundle_path
        
        return session
        
    def validate_environment(self):  # set up the environment required to run the model
        pass

    def completion(
        self,
    ):  # logic for parsing in - calling - parsing out model completion calls
        pass

    def embedding(
        self,
    ):  # logic for parsing in - calling - parsing out model embedding calls
        pass
