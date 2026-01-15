"""
Endpoints for managing callbacks
"""
import json
import os

from fastapi import APIRouter, Depends

from litellm.litellm_core_utils.logging_callback_manager import CallbacksByType
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

router = APIRouter()


@router.get(
    "/callbacks/list",
    tags=["Logging Callbacks"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=CallbacksByType,
)
async def list_callbacks():
    """
    View List of Active Logging Callbacks
    """
    from litellm import logging_callback_manager

    # Get callbacks organized by type using the callback manager utility
    callbacks_by_type = logging_callback_manager.get_callbacks_by_type()
    
    return callbacks_by_type


@router.get(
    "/callbacks/configs",
    tags=["Logging Callbacks"],
    dependencies=[Depends(user_api_key_auth)],
)
async def get_callback_configs():
    """
    Get Available Callback Configurations
    
    Returns the configuration details for all available logging callbacks,
    including supported parameters, field types, and descriptions.
    """
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "integrations",
        "callback_configs.json"
    )
    
    with open(config_path, "r") as f:
        configs = json.load(f)
    
    return configs