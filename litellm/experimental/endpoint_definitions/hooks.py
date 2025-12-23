"""
Generic endpoint hooks for pre/post call processing.

Hooks allow customizing behavior like authentication, logging, etc.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict


class GenericEndpointHooks(ABC):
    """
    Base class for endpoint hooks.
    
    Implement this to add custom pre/post call behavior for an endpoint.
    """
    
    @abstractmethod
    async def async_pre_call_hook(
        self,
        operation_name: str,
        headers: Dict[str, str],
        kwargs: Dict[str, Any],
    ) -> Dict[str, str]:
        """
        Called before making the HTTP request (async version).
        
        Args:
            operation_name: Name of the operation being called
            headers: Current headers dict (can be modified)
            kwargs: The kwargs being passed to the operation
            
        Returns:
            Modified headers dict
        """
        pass
    
    @abstractmethod
    def sync_pre_call_hook(
        self,
        operation_name: str,
        headers: Dict[str, str],
        kwargs: Dict[str, Any],
    ) -> Dict[str, str]:
        """
        Called before making the HTTP request (sync version).
        
        Args:
            operation_name: Name of the operation being called
            headers: Current headers dict (can be modified)
            kwargs: The kwargs being passed to the operation
            
        Returns:
            Modified headers dict
        """
        pass
