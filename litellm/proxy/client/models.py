import requests
from typing import List, Dict, Any, Optional, Union
from .exceptions import UnauthorizedError, NotFoundError


class ModelsManagementClient:
    def __init__(self, base_url: str, api_key: Optional[str] = None):
        """
        Initialize the ModelsManagementClient.

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
        headers = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def list(self, return_request: bool = False) -> Union[List[Dict[str, Any]], requests.Request]:
        """
        Get the list of models supported by the server.

        Args:
            return_request (bool): If True, returns the prepared request object instead of executing it.
                                 Useful for inspection or modification before sending.

        Returns:
            Union[List[Dict[str, Any]], requests.Request]: Either a list of model information dictionaries
            or a prepared request object if return_request is True.

        Raises:
            UnauthorizedError: If the request fails with a 401 status code
            requests.exceptions.RequestException: If the request fails with any other error
        """
        url = f"{self._base_url}/models"
        request = requests.Request("GET", url, headers=self._get_headers())

        if return_request:
            return request

        # Prepare and send the request
        session = requests.Session()
        try:
            response = session.send(request.prepare())
            response.raise_for_status()
            return response.json()["data"]
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                raise UnauthorizedError(e)
            raise

    def new(
        self,
        model_name: str,
        model_params: Dict[str, Any],
        model_info: Optional[Dict[str, Any]] = None,
        return_request: bool = False,
    ) -> Union[Dict[str, Any], requests.Request]:
        """
        Add a new model to the proxy.

        Args:
            model_name (str): Name of the model to add
            model_params (Dict[str, Any]): Parameters for the model (e.g., model type, api_base, api_key)
            model_info (Optional[Dict[str, Any]]): Additional information about the model
            return_request (bool): If True, returns the prepared request object instead of executing it

        Returns:
            Union[Dict[str, Any], requests.Request]: Either the response from the server or
            a prepared request object if return_request is True

        Raises:
            UnauthorizedError: If the request fails with a 401 status code
            requests.exceptions.RequestException: If the request fails with any other error
        """
        url = f"{self._base_url}/model/new"

        data = {
            "model_name": model_name,
            "litellm_params": model_params,
        }
        if model_info:
            data["model_info"] = model_info

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

    def delete(self, model_id: str, return_request: bool = False) -> Union[Dict[str, Any], requests.Request]:
        """
        Delete a model from the proxy.

        Args:
            model_id (str): ID of the model to delete (e.g., "2f23364f-4579-4d79-a43a-2d48dd551c2e")
            return_request (bool): If True, returns the prepared request object instead of executing it

        Returns:
            Union[Dict[str, Any], requests.Request]: Either the response from the server or
            a prepared request object if return_request is True

        Raises:
            UnauthorizedError: If the request fails with a 401 status code
            NotFoundError: If the request fails with a 404 status code or indicates the model was not found
            requests.exceptions.RequestException: If the request fails with any other error
        """
        url = f"{self._base_url}/model/delete"
        data = {"id": model_id}

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
            if e.response.status_code == 404 or "not found" in e.response.text.lower():
                raise NotFoundError(e)
            raise

    def get(
        self, model_id: Optional[str] = None, model_name: Optional[str] = None, return_request: bool = False
    ) -> Union[Dict[str, Any], requests.Request]:
        """
        Get information about a specific model by its ID or name.

        Args:
            model_id (Optional[str]): ID of the model to retrieve
            model_name (Optional[str]): Name of the model to retrieve
            return_request (bool): If True, returns the prepared request object instead of executing it

        Returns:
            Union[Dict[str, Any], requests.Request]: Either the model information from the server or
            a prepared request object if return_request is True

        Raises:
            ValueError: If neither model_id nor model_name is provided, or if both are provided
            UnauthorizedError: If the request fails with a 401 status code
            NotFoundError: If the model is not found
            requests.exceptions.RequestException: If the request fails with any other error
        """
        if (model_id is None and model_name is None) or (model_id is not None and model_name is not None):
            raise ValueError("Exactly one of model_id or model_name must be provided")

        # If return_request is True, delegate to info
        if return_request:
            result = self.info(return_request=True)
            assert isinstance(result, requests.Request)
            return result

        # Get all models and filter
        models = self.info()
        assert isinstance(models, List)

        # Find the matching model
        for model in models:
            if (model_id and model.get("model_info", {}).get("id") == model_id) or (
                model_name and model.get("model_name") == model_name
            ):
                return model

        # If we get here, no model was found
        if model_id:
            msg = f"Model with id={model_id} not found"
        elif model_name:
            msg = f"Model with model_name={model_name} not found"
        else:
            msg = "Unknown error trying to find model"
        raise NotFoundError(
            requests.exceptions.HTTPError(
                msg,
                response=requests.Response(),  # Empty response since we didn't make a direct request
            )
        )

    def info(self, return_request: bool = False) -> Union[List[Dict[str, Any]], requests.Request]:
        """
        Get detailed information about all models from the server.

        Args:
            return_request (bool): If True, returns the prepared request object instead of executing it

        Returns:
            Union[List[Dict[str, Any]], requests.Request]: Either a list of model information dictionaries
            or a prepared request object if return_request is True

        Raises:
            UnauthorizedError: If the request fails with a 401 status code
            requests.exceptions.RequestException: If the request fails with any other error
        """
        url = f"{self._base_url}/v1/model/info"
        request = requests.Request("GET", url, headers=self._get_headers())

        if return_request:
            return request

        # Prepare and send the request
        session = requests.Session()
        try:
            response = session.send(request.prepare())
            response.raise_for_status()
            return response.json()["data"]
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                raise UnauthorizedError(e)
            raise

    def update(
        self,
        model_id: str,
        model_params: Dict[str, Any],
        model_info: Optional[Dict[str, Any]] = None,
        return_request: bool = False,
    ) -> Union[Dict[str, Any], requests.Request]:
        """
        Update an existing model's configuration.

        Args:
            model_id (str): ID of the model to update
            model_params (Dict[str, Any]): New parameters for the model (e.g., model type, api_base, api_key)
            model_info (Optional[Dict[str, Any]]): Additional information about the model
            return_request (bool): If True, returns the prepared request object instead of executing it

        Returns:
            Union[Dict[str, Any], requests.Request]: Either the response from the server or
            a prepared request object if return_request is True

        Raises:
            UnauthorizedError: If the request fails with a 401 status code
            NotFoundError: If the model is not found
            requests.exceptions.RequestException: If the request fails with any other error
        """
        url = f"{self._base_url}/model/update"

        data = {
            "id": model_id,
            "litellm_params": model_params,
        }
        if model_info:
            data["model_info"] = model_info

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
            if e.response.status_code == 404 or "not found" in e.response.text.lower():
                raise NotFoundError(e)
            raise
