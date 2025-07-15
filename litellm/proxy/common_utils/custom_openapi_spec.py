import logging
from typing import Any, Dict, List, Optional

verbose_proxy_logger = logging.getLogger("litellm.proxy.proxy_server")


class CustomOpenAPISpec:
    """
    Handler for customizing OpenAPI specifications with Pydantic models
    for documentation purposes without runtime validation.
    """
    
    CHAT_COMPLETION_PATHS = [
        "/v1/chat/completions",
        "/chat/completions", 
        "/engines/{model}/chat/completions",
        "/openai/deployments/{model}/chat/completions"
    ]
    
    @staticmethod
    def get_pydantic_schema(model_class) -> Optional[Dict[str, Any]]:
        """
        Get JSON schema from a Pydantic model, handling both v1 and v2 APIs.
        
        Args:
            model_class: Pydantic model class
            
        Returns:
            JSON schema dict or None if failed
        """
        try:
            # Try Pydantic v2 method first
            return model_class.model_json_schema()  # type: ignore
        except AttributeError:
            try:
                # Fallback to Pydantic v1 method
                return model_class.schema()  # type: ignore
            except AttributeError:
                # If both methods fail, return None
                return None
    
    @staticmethod
    def add_schema_to_components(openapi_schema: Dict[str, Any], schema_name: str, schema_def: Dict[str, Any]) -> None:
        """
        Add a schema definition to the OpenAPI components/schemas section.
        
        Args:
            openapi_schema: The OpenAPI schema dict to modify
            schema_name: Name for the schema component
            schema_def: The schema definition
        """
        # Ensure components/schemas structure exists
        if "components" not in openapi_schema:
            openapi_schema["components"] = {}
        if "schemas" not in openapi_schema["components"]:
            openapi_schema["components"]["schemas"] = {}
        
        # Add the schema
        openapi_schema["components"]["schemas"][schema_name] = schema_def
    
    @staticmethod
    def add_request_body_to_paths(openapi_schema: Dict[str, Any], paths: List[str], schema_ref: str) -> None:
        """
        Add request body schema reference to specified paths.
        
        Args:
            openapi_schema: The OpenAPI schema dict to modify
            paths: List of paths to update
            schema_ref: Reference to the schema component (e.g., "#/components/schemas/ModelName")
        """
        for path in paths:
            if path in openapi_schema.get("paths", {}) and "post" in openapi_schema["paths"][path]:
                openapi_schema["paths"][path]["post"]["requestBody"] = {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "$ref": schema_ref
                            }
                        }
                    }
                }
    
    @staticmethod
    def add_chat_completion_request_schema(openapi_schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add ProxyChatCompletionRequest schema to chat completion endpoints for documentation.
        This shows the request body in Swagger without runtime validation.
        
        Args:
            openapi_schema: The OpenAPI schema dict to modify
            
        Returns:
            Modified OpenAPI schema
        """
        try:
            from litellm.proxy._types import ProxyChatCompletionRequest

            # Get the schema for ProxyChatCompletionRequest
            request_schema = CustomOpenAPISpec.get_pydantic_schema(ProxyChatCompletionRequest)
            
            # Only proceed if we successfully got the schema
            if request_schema is not None:
                # Add schema to components
                CustomOpenAPISpec.add_schema_to_components(openapi_schema, "ProxyChatCompletionRequest", request_schema)
                
                # Add request body to chat completion endpoints
                CustomOpenAPISpec.add_request_body_to_paths(
                    openapi_schema, 
                    CustomOpenAPISpec.CHAT_COMPLETION_PATHS, 
                    "#/components/schemas/ProxyChatCompletionRequest"
                )
                
                verbose_proxy_logger.debug("Successfully added ProxyChatCompletionRequest schema to OpenAPI spec")
            else:
                verbose_proxy_logger.debug("Could not get schema for ProxyChatCompletionRequest")
                
        except Exception as e:
            # If schema addition fails, continue without it
            verbose_proxy_logger.debug(f"Failed to add chat completion request schema: {str(e)}")
        
        return openapi_schema
    
    @staticmethod
    def customize_openapi_schema(openapi_schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply all custom OpenAPI schema modifications.
        
        Args:
            openapi_schema: The base OpenAPI schema
            
        Returns:
            Customized OpenAPI schema
        """
        # Add chat completion request schema
        openapi_schema = CustomOpenAPISpec.add_chat_completion_request_schema(openapi_schema)
        
        # Future customizations can be added here
        # e.g., CustomOpenAPISpec.add_embedding_request_schema(openapi_schema)
        
        return openapi_schema 