"""
Key Rotation Manager - Automated key rotation based on rotation schedules

Handles finding keys that need rotation based on their individual schedules.
"""

from datetime import datetime, timezone
from typing import List

from litellm._logging import verbose_proxy_logger
from litellm.constants import LITELLM_INTERNAL_JOBS_SERVICE_ACCOUNT_NAME
from litellm.proxy._types import (
    GenerateKeyResponse,
    LiteLLM_VerificationToken,
    RegenerateKeyRequest,
)
from litellm.proxy.hooks.key_management_event_hooks import KeyManagementEventHooks
from litellm.proxy.management_endpoints.key_management_endpoints import (
    _calculate_key_rotation_time,
    regenerate_key_fn,
)
from litellm.proxy.utils import PrismaClient


class KeyRotationManager:
    """
    Manages automated key rotation based on individual key rotation schedules.
    """
    
    def __init__(self, prisma_client: PrismaClient):
        self.prisma_client = prisma_client
        
    async def process_rotations(self):
        """
        Main entry point - find and rotate keys that are due for rotation
        """
        try:
            verbose_proxy_logger.info("Starting scheduled key rotation check...")
            
            # Find keys that are due for rotation
            keys_to_rotate = await self._find_keys_needing_rotation()
            
            if not keys_to_rotate:
                verbose_proxy_logger.debug("No keys are due for rotation at this time")
                return
                
            verbose_proxy_logger.info(f"Found {len(keys_to_rotate)} keys due for rotation")
            
            # Rotate each key
            for key in keys_to_rotate:
                try:
                    await self._rotate_key(key)
                    key_identifier = key.key_name or (key.token[:8] + "..." if key.token else "unknown")
                    verbose_proxy_logger.info(f"Successfully rotated key: {key_identifier}")
                except Exception as e:
                    key_identifier = key.key_name or (key.token[:8] + "..." if key.token else "unknown")
                    verbose_proxy_logger.error(f"Failed to rotate key {key_identifier}: {e}")
                    
        except Exception as e:
            verbose_proxy_logger.error(f"Key rotation process failed: {e}")
    
    async def _find_keys_needing_rotation(self) -> List[LiteLLM_VerificationToken]:
        """
        Find keys that are due for rotation based on their key_rotation_at timestamp.
        
        Logic:
        - Key has auto_rotate = true
        - key_rotation_at is null (needs initial setup) OR key_rotation_at <= now
        """
        now = datetime.now(timezone.utc)
        
        keys_with_rotation = await self.prisma_client.db.litellm_verificationtoken.find_many(
            where={
                "auto_rotate": True,  # Only keys marked for auto rotation
                "OR": [
                    {"key_rotation_at": None},  # Keys that need initial rotation time setup
                    {"key_rotation_at": {"lte": now}}  # Keys where rotation time has passed
                ]
            }
        )
        
        return keys_with_rotation
    
    def _should_rotate_key(self, key: LiteLLM_VerificationToken, now: datetime) -> bool:
        """
        Determine if a key should be rotated based on key_rotation_at timestamp.
        """
        if not key.rotation_interval:
            return False
        
        # If key_rotation_at is not set, rotate immediately (and set it)
        if key.key_rotation_at is None:
            return True
        
        # Check if the rotation time has passed
        return now >= key.key_rotation_at
    
    async def _rotate_key(self, key: LiteLLM_VerificationToken):
        """
        Rotate a single key using existing regenerate_key_fn and call the rotation hook
        """
        # Create regenerate request 
        regenerate_request = RegenerateKeyRequest(
            key=key.token or ""
        )
        
        # Create a system user for key rotation
        from litellm.proxy._types import UserAPIKeyAuth
        system_user = UserAPIKeyAuth.get_litellm_internal_jobs_user_api_key_auth()
        
        # Use existing regenerate key function
        response = await regenerate_key_fn(
            data=regenerate_request,
            user_api_key_dict=system_user,
            litellm_changed_by=LITELLM_INTERNAL_JOBS_SERVICE_ACCOUNT_NAME
        )
       
        # Update the NEW key with rotation info (regenerate_key_fn creates a new token)
        if isinstance(response, GenerateKeyResponse) and response.token_id and key.rotation_interval:
            # Calculate next rotation time using helper function
            now = datetime.now(timezone.utc)
            next_rotation_time = _calculate_key_rotation_time(key.rotation_interval)
            await self.prisma_client.db.litellm_verificationtoken.update(
                where={"token": response.token_id},
                data={
                    "rotation_count": (key.rotation_count or 0) + 1,
                    "last_rotation_at": now,
                    "key_rotation_at": next_rotation_time
                }
            )
        
        # Call the existing rotation hook for notifications, audit logs, etc.
        if isinstance(response, GenerateKeyResponse):
            await KeyManagementEventHooks.async_key_rotated_hook(
                data=regenerate_request,
                existing_key_row=key,
                response=response,
                user_api_key_dict=system_user,
                litellm_changed_by=LITELLM_INTERNAL_JOBS_SERVICE_ACCOUNT_NAME
            )
    