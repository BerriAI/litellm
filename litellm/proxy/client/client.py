from typing import Optional
from .models import ModelsManagementClient

class Client:
    """Main client for interacting with the LiteLLM proxy API."""
    
    def __init__(self, base_url: str, api_key: Optional[str] = None):
        """
        Initialize the LiteLLM proxy client.
        
        Args:
            base_url (str): The base URL of the LiteLLM proxy server (e.g., "http://localhost:8000")
            api_key (Optional[str]): API key for authentication. If provided, it will be sent as a Bearer token.
        """
        self.base_url = base_url.rstrip('/')  # Remove trailing slash if present
        self.api_key = api_key
        
        # Initialize resource clients
        self.models = ModelsManagementClient(base_url=self.base_url, api_key=self.api_key) 