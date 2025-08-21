import requests
from typing import List, Dict, Any, Optional, Union
from .exceptions import UnauthorizedError


class ChatClient:
    def __init__(self, base_url: str, api_key: Optional[str] = None):
        """
        Initialize the ChatClient.

        Args:
            base_url (str): The base URL of the LiteLLM proxy server (e.g., "http://localhost:8000")
            api_key (Optional[str]): API key for authentication. If provided, it will be sent as a Bearer token.
        """
        self._base_url = base_url.rstrip("/")  # Remove trailing slash if present
        self._api_key = api_key

    def _get_headers(self) -> Dict[str, str]:
        """
        Get the headers for API requests, including authorization if api_key is set.

        Returns:
            Dict[str, str]: Headers to use for API requests
        """
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def completions(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        n: Optional[int] = None,
        max_tokens: Optional[int] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        user: Optional[str] = None,
        return_request: bool = False,
    ) -> Union[Dict[str, Any], requests.Request]:
        """
        Create a chat completion.

        Args:
            model (str): The model to use for completion
            messages (List[Dict[str, str]]): The messages to generate a completion for
            temperature (Optional[float]): Sampling temperature between 0 and 2
            top_p (Optional[float]): Nucleus sampling parameter between 0 and 1
            n (Optional[int]): Number of completions to generate
            max_tokens (Optional[int]): Maximum number of tokens to generate
            presence_penalty (Optional[float]): Presence penalty between -2.0 and 2.0
            frequency_penalty (Optional[float]): Frequency penalty between -2.0 and 2.0
            user (Optional[str]): Unique identifier for the end user
            return_request (bool): If True, returns the prepared request object instead of executing it

        Returns:
            Union[Dict[str, Any], requests.Request]: Either the completion response from the server or
            a prepared request object if return_request is True

        Raises:
            UnauthorizedError: If the request fails with a 401 status code
            requests.exceptions.RequestException: If the request fails with any other error
        """
        url = f"{self._base_url}/chat/completions"

        # Build request data with required fields
        data: Dict[str, Any] = {"model": model, "messages": messages}

        # Add optional parameters if provided
        if temperature is not None:
            data["temperature"] = temperature
        if top_p is not None:
            data["top_p"] = top_p
        if n is not None:
            data["n"] = n
        if max_tokens is not None:
            data["max_tokens"] = max_tokens
        if presence_penalty is not None:
            data["presence_penalty"] = presence_penalty
        if frequency_penalty is not None:
            data["frequency_penalty"] = frequency_penalty
        if user is not None:
            data["user"] = user

        request = requests.Request("POST", url, headers=self._get_headers(), json=data)

        if return_request:
            return request

        # Prepare and send the request
        session = requests.Session()
        try:
            response = session.send(request.prepare())
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                raise UnauthorizedError(e)
            raise
