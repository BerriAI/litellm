"""
This is a file for the Infisical Secret Manager Integration

Handles Operations for:
- Read Secret
- Write Secret 
- Delete Secret

Requires:
* `pip install infisical-python`
"""

import os
from typing import Optional, Union, Any, Dict
import httpx
from litellm.secret_managers.base_secret_manager import BaseSecretManager
from litellm import verbose_logger
import litellm
from litellm.types.secret_managers.main import KeyManagementSystem

class InfisicalSecretManager(BaseSecretManager):
    def __init__(
        self,
        infisical_site_url: Optional[str] = None,
        infisical_client_id: Optional[str] = None,
        infisical_client_secret: Optional[str] = None,
        infisical_project_id: Optional[str] = None,
        infisical_environment_slug: Optional[str] = None,
        infisical_secret_path: Optional[str] = None,
        **kwargs
    ):
        super().__init__()
        self.infisical_site_url = infisical_site_url or os.getenv("INFISICAL_SITE_URL") or "https://app.infisical.com"
        self.infisical_client_id = infisical_client_id or os.getenv("INFISICAL_CLIENT_ID")
        self.infisical_client_secret = infisical_client_secret or os.getenv("INFISICAL_CLIENT_SECRET")
        self.infisical_project_id = infisical_project_id or os.getenv("INFISICAL_PROJECT_ID")
        self.infisical_environment_slug = infisical_environment_slug or os.getenv("INFISICAL_ENVIRONMENT_SLUG") or "dev"
        self.infisical_secret_path = infisical_secret_path or os.getenv("INFISICAL_SECRET_PATH") or "/"
        
        try:
            from infisical import InfisicalClient
            self.client = InfisicalClient(
                site_url=self.infisical_site_url,
                client_id=self.infisical_client_id,
                client_secret=self.infisical_client_secret,
            )
        except ImportError:
            raise ImportError("Missing infisical-python to call Infisical. Run 'pip install infisical-python'.")
        except Exception as e:
            raise e

    @classmethod
    def load_infisical_secret_manager(
        cls,
        use_infisical_secret_manager: Optional[bool],
        key_management_settings: Optional[Any] = None,
    ):
        """
        Initialize InfisicalSecretManager with settings from key_management_settings
        """
        if use_infisical_secret_manager is None or use_infisical_secret_manager is False:
            return
        try:
            # Extract Infisical settings from key_management_settings if provided
            infisical_kwargs = {}
            if key_management_settings is not None:
                infisical_kwargs = {
                    "infisical_site_url": getattr(key_management_settings, "infisical_site_url", None),
                    "infisical_client_id": getattr(key_management_settings, "infisical_client_id", None),
                    "infisical_client_secret": getattr(key_management_settings, "infisical_client_secret", None),
                    "infisical_project_id": getattr(key_management_settings, "infisical_project_id", None),
                    "infisical_environment_slug": getattr(key_management_settings, "infisical_environment_slug", None),
                    "infisical_secret_path": getattr(key_management_settings, "infisical_secret_path", None),
                }
                # Remove None values
                infisical_kwargs = {k: v for k, v in infisical_kwargs.items() if v is not None}
            
            litellm.secret_manager_client = cls(**infisical_kwargs)
            litellm._key_management_system = KeyManagementSystem.INFISICAL

        except Exception as e:
            raise e

    async def async_read_secret(
        self,
        secret_name: str,
        optional_params: Optional[dict] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
    ) -> Optional[str]:
        return self.sync_read_secret(secret_name, optional_params, timeout)

    def sync_read_secret(
        self,
        secret_name: str,
        optional_params: Optional[dict] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
    ) -> Optional[str]:
        try:
            # Infisical client operations are typically sync in the python SDK? 
            # Assuming sync for now.
            secret = self.client.getSecret(
                options={
                    "secretName": secret_name,
                    "projectId": self.infisical_project_id,
                    "environment": self.infisical_environment_slug,
                    "path": self.infisical_secret_path,
                    "type": "shared"
                }
            )
            return secret.secretValue
        except Exception as e:
            # Check if it's a "not found" error
            if "not found" in str(e).lower():
                 return None
            
            verbose_logger.exception(
                f"Error reading secret='{secret_name}' from Infisical: {str(e)}"
            )
            return None

    async def async_write_secret(
        self,
        secret_name: str,
        secret_value: str,
        description: Optional[str] = None,
        optional_params: Optional[dict] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        tags: Optional[Union[dict, list]] = None
    ) -> Dict[str, Any]:
        try:
            new_secret = self.client.createSecret(
                options={
                    "secretName": secret_name,
                    "secretValue": secret_value,
                    "projectId": self.infisical_project_id,
                    "environment": self.infisical_environment_slug,
                    "path": self.infisical_secret_path,
                    "type": "shared",
                    "secretComment": description
                }
            )
            return {"name": new_secret.secretName, "version": new_secret.version}
        except Exception as e:
             # Try update if create fails (maybe it exists)
             try:
                updated_secret = self.client.updateSecret(
                    options={
                        "secretName": secret_name,
                        "secretValue": secret_value,
                        "projectId": self.infisical_project_id,
                        "environment": self.infisical_environment_slug,
                        "path": self.infisical_secret_path,
                        "type": "shared"
                    }
                )
                return {"name": updated_secret.secretName, "version": updated_secret.version}
             except Exception as update_error:
                verbose_logger.exception(
                    f"Error writing secret='{secret_name}' to Infisical: {str(e)}"
                )
                raise e

    async def async_delete_secret(
        self,
        secret_name: str,
        recovery_window_in_days: Optional[int] = 7,
        optional_params: Optional[dict] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
    ) -> dict:
        try:
            self.client.deleteSecret(
                options={
                    "secretName": secret_name,
                    "projectId": self.infisical_project_id,
                    "environment": self.infisical_environment_slug,
                    "path": self.infisical_secret_path,
                    "type": "shared"
                }
            )
            return {"deleted": True, "secret_name": secret_name}
        except Exception as e:
            verbose_logger.exception(
                f"Error deleting secret='{secret_name}' from Infisical: {str(e)}"
            )
            raise e
