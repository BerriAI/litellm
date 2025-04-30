import requests
from typing import List, Dict, Any, Optional, Union
from .exceptions import UnauthorizedError

class ModelsManagementClient:
    def __init__(self, base_url: str, api_key: Optional[str] = None):
        """
        Initialize the ModelsManagementClient.
        
        Args:
            base_url (str): The base URL of the LiteLLM proxy server (e.g., "http://localhost:8000")
            api_key (Optional[str]): API key for authentication. If provided, it will be sent as a Bearer token.
        """
        self.base_url = base_url.rstrip('/')  # Remove trailing slash if present
        self.api_key = api_key
        
    def _get_headers(self) -> Dict[str, str]:
        """
        Get the headers for API requests, including authorization if api_key is set.
        
        Returns:
            Dict[str, str]: Headers to use for API requests
        """
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers
        
    def list_models(self, return_request: bool = False) -> Union[List[Dict[str, Any]], requests.Request]:
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
        url = f"{self.base_url}/models"
        request = requests.Request('GET', url, headers=self._get_headers())
        
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

    def add_model(
        self,
        model_name: str,
        model_params: Dict[str, Any],
        model_info: Optional[Dict[str, Any]] = None,
        return_request: bool = False
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
        url = f"{self.base_url}/model/new"
        
        data = {
            "model_name": model_name,
            "litellm_params": model_params,
        }
        if model_info:
            data["model_info"] = model_info
            
        request = requests.Request('POST', url, headers=self._get_headers(), json=data)
        
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

    def delete_model(
        self,
        model_id: str,
        return_request: bool = False
    ) -> Union[Dict[str, Any], requests.Request]:
        """
        Delete a model from the proxy.
        
        Args:
            model_id (str): ID of the model to delete
            return_request (bool): If True, returns the prepared request object instead of executing it
        
        Returns:
            Union[Dict[str, Any], requests.Request]: Either the response from the server or
            a prepared request object if return_request is True
            
        Raises:
            UnauthorizedError: If the request fails with a 401 status code
            requests.exceptions.RequestException: If the request fails with any other error
        """
        url = f"{self.base_url}/model/delete"
        data = {"id": model_id}
            
        request = requests.Request('POST', url, headers=self._get_headers(), json=data)
        
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