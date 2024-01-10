## This is a template base class to be used for adding new LLM providers via API calls
import litellm
import httpx
from typing import Optional


class BaseLLM:
    _client_session: Optional[httpx.Client] = None

    def create_client_session(self):
        if litellm.client_session:
            _client_session = litellm.client_session
        else:
            _client_session = httpx.Client()

        return _client_session

    def create_aclient_session(self):
        if litellm.aclient_session:
            _aclient_session = litellm.aclient_session
        else:
            _aclient_session = httpx.AsyncClient()

        return _aclient_session

    def __exit__(self):
        if hasattr(self, "_client_session"):
            self._client_session.close()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if hasattr(self, "_aclient_session"):
            await self._aclient_session.aclose()

    def validate_environment(self):  # set up the environment required to run the model
        pass

    def completion(
        self, *args, **kwargs
    ):  # logic for parsing in - calling - parsing out model completion calls
        pass

    def embedding(
        self, *args, **kwargs
    ):  # logic for parsing in - calling - parsing out model embedding calls
        pass
