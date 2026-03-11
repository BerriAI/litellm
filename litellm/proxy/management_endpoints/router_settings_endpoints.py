"""
ROUTER SETTINGS MANAGEMENT

Endpoints for accessing router configuration and metadata

GET /router/settings - Get router configuration including available routing strategies
GET /router/fields - Get router settings field definitions without values (for UI rendering)
"""

import inspect
from typing import Any, Dict, List, get_args

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.router import Router
from litellm.types.management_endpoints import (
    ROUTER_SETTINGS_FIELDS,
    ROUTING_STRATEGY_DESCRIPTIONS,
    RouterSettingsField,
)

router = APIRouter()


class RouterSettingsResponse(BaseModel):
    fields: List[RouterSettingsField] = Field(
        description="List of all configurable router settings with metadata"
    )
    current_values: Dict[str, Any] = Field(
        description="Current values of router settings"
    )
    routing_strategy_descriptions: Dict[str, str] = Field(
        description="Descriptions for each routing strategy option"
    )


class RouterFieldsResponse(BaseModel):
    fields: List[RouterSettingsField] = Field(
        description="List of all configurable router settings with metadata (without field values)"
    )
    routing_strategy_descriptions: Dict[str, str] = Field(
        description="Descriptions for each routing strategy option"
    )


def _get_routing_strategies_from_router_class() -> List[str]:
    """
    Dynamically extract routing strategies from the Router class __init__ method.
    """
    # Get the __init__ signature
    sig = inspect.signature(Router.__init__)
    
    # Get the routing_strategy parameter
    routing_strategy_param = sig.parameters.get("routing_strategy")
    
    if routing_strategy_param and routing_strategy_param.annotation:
        # Extract Literal values using get_args
        literal_values = get_args(routing_strategy_param.annotation)
        if literal_values:
            return list(literal_values)
    
    raise ValueError("Unable to extract routing strategies from Router class")


@router.get(
    "/router/settings",
    tags=["Router Settings"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=RouterSettingsResponse,
)
async def get_router_settings(
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Get router configuration and available settings.
    
    Returns:
    - fields: List of all configurable router settings with their metadata (type, description, default, options)
              The routing_strategy field includes available options extracted from the Router class
    - current_values: Current values of router settings from config
    """
    from litellm.proxy.proxy_server import llm_router, proxy_config
    
    try:
        # Get available routing strategies dynamically from Router class
        available_routing_strategies = _get_routing_strategies_from_router_class()
        
        # Get router settings fields from types file
        router_fields = [field.model_copy(deep=True) for field in ROUTER_SETTINGS_FIELDS]
        
        # Populate routing_strategy field with available options and descriptions
        for field in router_fields:
            if field.field_name == "routing_strategy":
                field.options = available_routing_strategies
                break
        
        # Try to get router settings from config
        config = await proxy_config.get_config()
        router_settings_from_config = config.get("router_settings", {})
        
        # Get current values from llm_router if initialized
        current_values = {}
        if llm_router is not None:
            # Check all field names from the fields list
            for field in router_fields:
                if hasattr(llm_router, field.field_name):
                    value = getattr(llm_router, field.field_name)
                    current_values[field.field_name] = value
        
        # Merge with config values (config takes precedence)
        current_values.update(router_settings_from_config)
        
        # Update field values with current values
        for field in router_fields:
            if field.field_name in current_values:
                field.field_value = current_values[field.field_name]
        
        return RouterSettingsResponse(
            fields=router_fields,
            current_values=current_values,
            routing_strategy_descriptions=ROUTING_STRATEGY_DESCRIPTIONS,
        )
    except Exception as e:
        verbose_proxy_logger.error(
            f"Error fetching router settings: {str(e)}"
        )
        raise


@router.get(
    "/router/fields",
    tags=["Router Settings"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=RouterFieldsResponse,
)
async def get_router_fields(
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Get router settings field definitions without values.
    
    Returns only the field metadata (type, description, default, options) without
    populating field_value. This is useful for UI components that need to know
    what fields to render, but will get the actual values from a different endpoint.
    
    Returns:
    - fields: List of all configurable router settings with their metadata (type, description, default, options)
              The routing_strategy field includes available options extracted from the Router class
              Note: field_value will be None for all fields
    - routing_strategy_descriptions: Descriptions for each routing strategy option
    """
    try:
        # Get available routing strategies dynamically from Router class
        available_routing_strategies = _get_routing_strategies_from_router_class()
        
        # Get router settings fields from types file
        router_fields = [field.model_copy(deep=True) for field in ROUTER_SETTINGS_FIELDS]
        
        # Populate routing_strategy field with available options
        for field in router_fields:
            if field.field_name == "routing_strategy":
                field.options = available_routing_strategies
                break
        
        # Ensure field_value is None for all fields (don't populate values)
        for field in router_fields:
            field.field_value = None
        
        return RouterFieldsResponse(
            fields=router_fields,
            routing_strategy_descriptions=ROUTING_STRATEGY_DESCRIPTIONS,
        )
    except Exception as e:
        verbose_proxy_logger.error(
            f"Error fetching router fields: {str(e)}"
        )
        raise

