"""
Azure Token Handler

Handles authentication token issuance and caching for Azure Cognitive Services
"""

import time
from typing import Dict, Optional, Tuple

from litellm._logging import verbose_logger
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    HTTPHandler,
    get_async_httpx_client,
)
from litellm.types.utils import LlmProviders


class AzureBaseIssueTokenHandler:
    """
    Handler for Azure Cognitive Services token issuance with in-memory caching.
    
    Azure tokens are valid for 10 minutes. This handler caches tokens and automatically
    refreshes them when expired.
    """
    
    def __init__(self) -> None:
        super().__init__()
        self._token_cache: Dict[Tuple[str, str], Tuple[str, float]] = {}
        self.http_handler: Optional[HTTPHandler] = None
        self.async_handler: Optional[AsyncHTTPHandler] = None
        # Azure tokens are valid for 10 minutes (600 seconds)
        # We refresh slightly before expiry to avoid edge cases
        self.token_validity_seconds = 580  # 9 minutes 40 seconds
    
    def _get_token_endpoint(self, api_base: str) -> str:
        """
        Construct the token endpoint URL from the API base.
        
        Args:
            api_base: Azure API base URL (e.g., 'https://eastus.api.cognitive.microsoft.com')
            
        Returns:
            Complete token endpoint URL
        """
        # Remove trailing slash if present
        api_base = api_base.rstrip("/")
        return f"{api_base}/sts/v1.0/issueToken"
    
    def _is_token_expired(self, expiry_time: float) -> bool:
        """
        Check if a cached token has expired.
        
        Args:
            expiry_time: Unix timestamp when the token expires
            
        Returns:
            True if token is expired, False otherwise
        """
        current_time = time.time()
        return current_time >= expiry_time
    
    def _fetch_token(self, subscription_key: str, api_base: str) -> str:
        """
        Fetch a new token from Azure.
        
        Args:
            subscription_key: Azure subscription key (Ocp-Apim-Subscription-Key)
            api_base: Azure API base URL (e.g., 'https://eastus.api.cognitive.microsoft.com')
            
        Returns:
            Access token string
            
        Raises:
            Exception: If token fetch fails
        """
        endpoint = self._get_token_endpoint(api_base=api_base)
        
        headers = {
            "Ocp-Apim-Subscription-Key": subscription_key,
            "Content-type": "application/x-www-form-urlencoded",
        }
        
        verbose_logger.debug(
            f"Fetching new Azure token from: {api_base}"
        )
        
        if self.http_handler is None:
            self.http_handler = HTTPHandler()
        
        try:
            response = self.http_handler.post(
                url=endpoint,
                headers=headers,
                data="",
            )
            
            if response.status_code != 200:
                raise Exception(
                    f"Failed to fetch Azure token. Status: {response.status_code}, "
                    f"Response: {response.text}"
                )
            
            token = response.text
            verbose_logger.debug(
                f"Successfully fetched Azure token from: {api_base}"
            )
            return token
            
        except Exception as e:
            verbose_logger.error(
                f"Error fetching Azure token from {api_base}: {str(e)}"
            )
            raise
    
    def get_token(self, subscription_key: str, api_base: str) -> str:
        """
        Get an Azure access token, using cached token if available and valid.
        
        This method:
        1. Checks if a valid cached token exists
        2. Returns cached token if valid
        3. Fetches and caches a new token if expired or missing
        
        Args:
            subscription_key: Azure subscription key
            api_base: Azure API base URL (e.g., 'https://eastus.api.cognitive.microsoft.com')
            
        Returns:
            Valid access token
            
        Raises:
            Exception: If token fetch fails
        """
        cache_key = (subscription_key, api_base)
        
        verbose_logger.debug(
            f"Getting Azure token for API base: {api_base}"
        )
        
        # Check if we have a cached token
        if cache_key in self._token_cache:
            cached_token, expiry_time = self._token_cache[cache_key]
            
            if not self._is_token_expired(expiry_time=expiry_time):
                verbose_logger.debug(
                    f"Using cached Azure token for API base: {api_base}"
                )
                return cached_token
            else:
                verbose_logger.debug(
                    f"Cached Azure token expired for API base: {api_base}, fetching new token"
                )
        else:
            verbose_logger.debug(
                f"No cached token found for API base: {api_base}, fetching new token"
            )
        
        # Fetch new token
        token = self._fetch_token(
            subscription_key=subscription_key,
            api_base=api_base,
        )
        
        # Cache the token with expiry time
        expiry_time = time.time() + self.token_validity_seconds
        self._token_cache[cache_key] = (token, expiry_time)
        
        verbose_logger.debug(
            f"Cached new Azure token for API base: {api_base}, expires in {self.token_validity_seconds}s"
        )
        
        return token
    
    async def _fetch_token_async(self, subscription_key: str, api_base: str) -> str:
        """
        Fetch a new token from Azure asynchronously.
        
        Args:
            subscription_key: Azure subscription key (Ocp-Apim-Subscription-Key)
            api_base: Azure API base URL (e.g., 'https://eastus.api.cognitive.microsoft.com')
            
        Returns:
            Access token string
            
        Raises:
            Exception: If token fetch fails
        """
        endpoint = self._get_token_endpoint(api_base=api_base)
        
        headers = {
            "Ocp-Apim-Subscription-Key": subscription_key,
            "Content-type": "application/x-www-form-urlencoded",
        }
        
        verbose_logger.debug(
            f"Fetching new Azure token from: {api_base} (async)"
        )
        
        if self.async_handler is None:
            self.async_handler = get_async_httpx_client(llm_provider=LlmProviders.AZURE)
        
        try:
            response = await self.async_handler.post(
                url=endpoint,
                headers=headers,
                data="",
            )
            
            if response.status_code != 200:
                raise Exception(
                    f"Failed to fetch Azure token. Status: {response.status_code}, "
                    f"Response: {response.text}"
                )
            
            token = response.text
            verbose_logger.debug(
                f"Successfully fetched Azure token from: {api_base} (async)"
            )
            return token
            
        except Exception as e:
            verbose_logger.error(
                f"Error fetching Azure token from {api_base} (async): {str(e)}"
            )
            raise
    
    async def get_token_async(self, subscription_key: str, api_base: str) -> str:
        """
        Get an Azure access token asynchronously, using cached token if available and valid.
        
        This method:
        1. Checks if a valid cached token exists
        2. Returns cached token if valid
        3. Fetches and caches a new token if expired or missing
        
        Args:
            subscription_key: Azure subscription key
            api_base: Azure API base URL (e.g., 'https://eastus.api.cognitive.microsoft.com')
            
        Returns:
            Valid access token
            
        Raises:
            Exception: If token fetch fails
        """
        cache_key = (subscription_key, api_base)
        
        verbose_logger.debug(
            f"Getting Azure token for API base: {api_base} (async)"
        )
        
        # Check if we have a cached token
        if cache_key in self._token_cache:
            cached_token, expiry_time = self._token_cache[cache_key]
            
            if not self._is_token_expired(expiry_time=expiry_time):
                verbose_logger.debug(
                    f"Using cached Azure token for API base: {api_base} (async)"
                )
                return cached_token
            else:
                verbose_logger.debug(
                    f"Cached Azure token expired for API base: {api_base}, fetching new token (async)"
                )
        else:
            verbose_logger.debug(
                f"No cached token found for API base: {api_base}, fetching new token (async)"
            )
        
        # Fetch new token
        token = await self._fetch_token_async(
            subscription_key=subscription_key,
            api_base=api_base,
        )
        
        # Cache the token with expiry time
        expiry_time = time.time() + self.token_validity_seconds
        self._token_cache[cache_key] = (token, expiry_time)
        
        verbose_logger.debug(
            f"Cached new Azure token for API base: {api_base}, expires in {self.token_validity_seconds}s (async)"
        )
        
        return token
