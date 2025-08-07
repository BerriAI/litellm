import os
import time
import uuid
import asyncio
import httpx
from typing import Optional, Union, Any, Dict, Tuple
from datetime import datetime
from fastapi import HTTPException

from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy._types import UserAPIKeyAuth
from litellm.caching import DualCache

from moneta.call_data_store import CallDataStore
from moneta.config import LagoConfig
from moneta.error_handler import ErrorHandler
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    List,
    Literal,
    Optional,
    Tuple,
    Union
)
class EntitlementCallback(CustomLogger):
    def __init__(self):
        """Initialize the Lago logger with configuration and storage"""
        super().__init__()
        
        # Load configuration
        try:
            self.config = LagoConfig()
        except ValueError as e:
            ErrorHandler.handle_config_error(e)
        
        # Initialize storage
        self.call_store = CallDataStore()
        
        print(f"LagoLogger initialized with base: {self.config.api_base}")
    
    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict,
        call_type: Literal[
            "completion",
            "text_completion",
            "embeddings", 
            "image_generation",
            "moderation",
            "audio_transcription",
            "pass_through_endpoint",
            "rerank",
        ],
    ) -> Optional[Union[Exception, str, dict]]:
        """
        Pre-call entitlement check.
        
        This method is called before making an LLM API call to check if the customer
        has sufficient credits or an active subscription.
        
        Args:
            user_api_key_dict: User authentication information
            cache: LiteLLM cache instance
            data: Request data including call_id and parameters
            call_type: Type of API call (completion, embeddings, etc.)
            
        Returns:
            None to allow request, Exception to block request
        """
        print("async_pre_call_hook")
        try:
            # Extract identifiers
            call_id = data.get("litellm_call_id")
            user_id = data["metadata"]["headers"]["x-openwebui-user-id"]
            external_customer_id = user_id
            print(f"call_id: {call_id}, external_customer_id: {external_customer_id}")
            print(f"call type: {call_type}")
            if not call_id or not external_customer_id:
                ErrorHandler.handle_missing_customer_id(call_id)
                raise HTTPException(
                    status_code=401,
                    detail={
                        "error": {
                            "message": "Authentication required. Please provide valid user credentials.",
                            "type": "authentication_error",
                            "param": None,
                            "code": "401"
                        }
                    }
                )
            
            # Check entitlement first to get external_subscription_id
            authorized, external_subscription_id = await self._check_authorization(external_customer_id)
            print(f"authorized: {authorized}, external_subscription_id: {external_subscription_id}")

            if not authorized:
                raise HTTPException(
                    status_code=403,
                    detail={
                        "error": {
                            "message": "Insufficient credits or inactive subscription",
                            "type": "authentication_error",
                            "param": None,
                            "code": "403"
                        }
                    }
                )
            
            # Store external_subscription_id for later retrieval (fallback to customer_id if not available)
            subscription_id_to_store = external_subscription_id
            try:
                self.call_store.store(call_id, subscription_id_to_store)
            except Exception as e:
                ErrorHandler.handle_storage_error(e, "storing call data")
                return None  # Continue even if storage fails
            
            return None  # Allow request
            
        except HTTPException:
            # Re-raise HTTPException as-is
            raise
        except Exception as e:
            print(f"Exception in async_pre_call_hook: {e}")
            raise HTTPException(
                status_code=500,
                detail={"error": "Service temporarily unavailable. Please try again later."}
            )
    

    async def async_log_success_event(
        self,
        kwargs: dict,
        response_obj: Any,
        start_time: datetime,
        end_time: datetime
    ) -> None:
        """
        Post-call usage reporting.

        This method is called after a successful LLM API call to report
        actual usage to Lago for billing purposes.

        Args:
            kwargs: Request parameters including call_id and response_cost
            response_obj: Response object from the LLM API
            start_time: When the API call started
            end_time: When the API call completed
        """
        try:
            # Get call data
            call_id = kwargs.get("litellm_call_id")
            if not call_id:
                return

            external_subscription_id = self.call_store.get_and_remove(call_id)
            if not external_subscription_id:
                print(f"Warning: No stored external_subscription_id for call {call_id}")
                return

            # Get usage data
            cost = kwargs.get("response_cost", 0.0)
            print(f"cost: {cost}")
            if cost <= 0:
                return  # No cost to report

            # Send usage event (fire and forget)
            asyncio.create_task(self._send_usage_event(external_subscription_id, cost, call_id))

        except Exception as e:
            ErrorHandler.handle_usage_error(e)

    async def async_log_failure_event(
        self,
        kwargs: dict,
        response_obj: Any,
        start_time: datetime,
        end_time: datetime
    ) -> None:
        """
        Cleanup on failure.

        This method is called when an LLM API call fails to clean up
        any stored call data.

        Args:
            kwargs: Request parameters including call_id
            response_obj: Error response object
            start_time: When the API call started
            end_time: When the API call failed
        """
        try:
            call_id = kwargs.get("litellm_call_id")
            if call_id:
                self.call_store.get_and_remove(call_id)  # Cleanup
        except Exception as e:
            ErrorHandler.handle_storage_error(e, "failure cleanup")

    async def _check_authorization(self, customer_id: str) -> Tuple[bool, Optional[str]]:
        """
        Check authorization with Lago.

        Makes a request to Lago's entitlement authorization endpoint
        to verify if the customer has sufficient credits or an active subscription.

        Args:
            customer_id: Customer identifier for authorization check

        Returns:
            Tuple of (is_authorized, external_subscription_id)
        """
        try:
            payload = {
                "external_customer_id": customer_id,
                "emit_event": False,
                "publisher_id": self.config.publisher_id,
                "action_name": "read",
                "context": [],
                "resource": {
                    "id": 1, 
                    "name": "", 
                    "type": "article",
                    "author": "any.email@is.fine",
                    "tags": []                             
                },
                "timestamp": int(time.time())
            }

            async with httpx.AsyncClient(timeout=self.config.timeout) as client:
                response = await client.post(
                    self.config.get_entitlement_url(),
                    json=payload,
                    headers=self.config.get_auth_headers()
                )
                print(f"Authorization response: {response.json()}")
                if response.status_code == 200:
                    result = response.json()
                    is_authorized = result.get("status") == "Allow"

                    # Extract external_subscription_id from the response
                    if is_authorized:
                        external_subscription_id = None
                        if "extra" in result and "authorized_subscription" in result["extra"]:
                            external_subscription_id = result["extra"]["authorized_subscription"].get("external_id")
                    else:
                        external_subscription_id = None
                    return is_authorized, external_subscription_id
                else:
                    print(f"Authorization failed: {response.status_code}")
                    return self.config.fallback_allow, None

        except Exception as e:
            print(f"Authorization error: {e}")
            fallback_allow = ErrorHandler.handle_authorization_error(e, self.config.fallback_allow)
            return fallback_allow, None

    async def _send_usage_event(self, external_subscription_id: str, cost: float, call_id: str) -> None:
        """
        Send usage event to Lago.

        Reports actual usage to Lago's events endpoint for billing purposes.
        This method runs asynchronously and doesn't block the main request flow.

        Args:
            external_subscription_id: External subscription identifier from Lago authorization
            cost: Actual cost of the API call
            call_id: Unique identifier for the API call
        """
        try:
            payload = {
                "event": {
                    "transaction_id": str(uuid.uuid4()),
                    "external_subscription_id": external_subscription_id,
                    "code": "credits_in_cent",
                    "timestamp": int(time.time()),
                    "properties": {
                        "credits_in_cent": int(cost * 100),  # Convert to cents
                        "call_id": call_id
                    }
                }
            }
            print(f"Sending usage event for subscription {external_subscription_id}: ${cost:.4f}")
            async with httpx.AsyncClient(timeout=self.config.timeout) as client:
                response = await client.post(
                    self.config.get_events_url(),
                    json=payload,
                    headers=self.config.get_auth_headers()
                )

                if response.status_code in [200, 201]:
                    print(f"Usage event sent for subscription {external_subscription_id}: ${cost:.4f}")
                else:
                    print(f"Usage event failed: {response.status_code}")

        except Exception as e:
            ErrorHandler.handle_usage_error(e)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics for monitoring.

        Returns:
            Dictionary with logger and storage statistics
        """
        return {
            "config_valid": self.config.is_valid(),
            "storage_stats": self.call_store.get_stats(),
            "lago_api_base": self.config.api_base,
            "publisher_id": self.config.publisher_id,
            "fallback_allow": self.config.fallback_allow
        }

# Instantiate the callback (instance name will be used in config)
entitlement_checker = EntitlementCallback()
