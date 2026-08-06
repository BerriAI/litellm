import json
from collections.abc import Iterator
from typing import Any, Final

import requests

from .exceptions import UnauthorizedError


class ChatClient:
    def __init__(self, base_url: str, api_key: str | None = None):
        """
        Initialize the ChatClient.

        Args:
            base_url (str): The base URL of the LiteLLM proxy server (e.g., "http://localhost:8000")
            api_key (Optional[str]): API key for authentication. If provided, it will be sent as a Bearer token.
        """
        self._base_url = base_url.rstrip("/")  # Remove trailing slash if present
        self._api_key = api_key

    def _get_headers(self) -> dict[str, str]:
        """
        Get the headers for API requests, including authorization if api_key is set.

        Returns:
            Dict[str, str]: Headers to use for API requests
        """
        headers: Final = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def completions(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        top_p: float | None = None,
        n: int | None = None,
        max_tokens: int | None = None,
        presence_penalty: float | None = None,
        frequency_penalty: float | None = None,
        user: str | None = None,
        return_request: bool = False,
    ) -> dict[str, Any] | requests.Request:
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
        url: Final = f"{self._base_url}/chat/completions"

        # Build request data with required fields
        data: Final[dict[str, Any]] = {"model": model, "messages": messages}

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

        request: Final = requests.Request("POST", url, headers=self._get_headers(), json=data)

        if return_request:
            return request

        # Prepare and send the request
        session: Final = requests.Session()
        try:
            response: Final = session.send(request.prepare())
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                raise UnauthorizedError(e)
            raise

    def completions_stream(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        top_p: float | None = None,
        n: int | None = None,
        max_tokens: int | None = None,
        presence_penalty: float | None = None,
        frequency_penalty: float | None = None,
        user: str | None = None,
    ) -> Iterator[dict[str, Any]]:
        """
        Create a streaming chat completion.

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

        Yields:
            Dict[str, Any]: Streaming response chunks from the server

        Raises:
            UnauthorizedError: If the request fails with a 401 status code
            requests.exceptions.RequestException: If the request fails with any other error
        """
        url: Final = f"{self._base_url}/chat/completions"

        # Build request data with required fields
        data: Final[dict[str, Any]] = {"model": model, "messages": messages, "stream": True}

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

        # Make streaming request
        session: Final = requests.Session()
        try:
            response: Final = session.post(url, headers=self._get_headers(), json=data, stream=True)
            response.raise_for_status()

            # Parse SSE stream
            for line in response.iter_lines():
                if line:
                    line = line.decode("utf-8")
                    if line.startswith("data: "):
                        data_str = line[6:]  # Remove 'data: ' prefix
                        if data_str.strip() == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data_str)
                            yield chunk
                        except json.JSONDecodeError:
                            continue

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                raise UnauthorizedError(e)
            raise
