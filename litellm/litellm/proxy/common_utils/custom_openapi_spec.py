from typing import Any, Dict, List, Optional, Type

from litellm._logging import verbose_proxy_logger


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
    
    EMBEDDING_PATHS = [
        "/v1/embeddings",
        "/embeddings",
        "/engines/{model}/embeddings", 
        "/openai/deployments/{model}/embeddings"
    ]
    
    RESPONSES_API_PATHS = [
        "/v1/responses",
        "/responses"
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
        Add request body with expanded form fields for better Swagger UI display.
        This keeps the request body but expands it to show individual fields in the UI.
        
        Args:
            openapi_schema: The OpenAPI schema dict to modify
            paths: List of paths to update
            schema_ref: Reference to the schema component (e.g., "#/components/schemas/ModelName")
        """
        for path in paths:
            if path in openapi_schema.get("paths", {}) and "post" in openapi_schema["paths"][path]:
                # Get the actual schema to extract ALL field definitions
                schema_name = schema_ref.split("/")[-1]  # Extract "ProxyChatCompletionRequest" from the ref
                actual_schema = openapi_schema.get("components", {}).get("schemas", {}).get(schema_name, {})
                schema_properties = actual_schema.get("properties", {})
                required_fields = actual_schema.get("required", [])
                
                # Create an expanded inline schema instead of just a $ref
                # This makes Swagger UI show all individual fields in the request body editor
                expanded_schema = {
                    "type": "object",
                    "required": required_fields,
                    "properties": {}
                }
                
                # Add all properties with their full definitions
                for field_name, field_def in schema_properties.items():
                    expanded_field = CustomOpenAPISpec._expand_field_definition(field_def)
                    
                    # Add a simple example for the messages field
                    if field_name == "messages":
                        expanded_field["example"] = [
                            {"role": "user", "content": "Hello, how are you?"}
                        ]
                    
                    expanded_schema["properties"][field_name] = expanded_field
                
                # Include $defs from the original schema to support complex types like AllMessageValues
                # This ensures that message types and other complex union types work properly
                if "$defs" in actual_schema:
                    expanded_schema["$defs"] = actual_schema["$defs"]
                
                # Set the request body with the expanded schema
                openapi_schema["paths"][path]["post"]["requestBody"] = {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": expanded_schema
                        }
                    }
                }
                
                # Keep any existing parameters (like path parameters) but remove conflicting query params
                if "parameters" in openapi_schema["paths"][path]["post"]:
                    existing_params = openapi_schema["paths"][path]["post"]["parameters"]
                    # Only keep path parameters, remove query params that conflict with request body
                    filtered_params = [
                        param for param in existing_params 
                        if param.get("in") == "path"
                    ]
                    openapi_schema["paths"][path]["post"]["parameters"] = filtered_params
    
    @staticmethod
    def _extract_field_schema(field_def: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract a simple schema from a Pydantic field definition for parameter display.
        
        Args:
            field_def: Pydantic field definition
            
        Returns:
            Simplified schema for OpenAPI parameter
        """
        # Handle simple types
        if "type" in field_def:
            return {"type": field_def["type"]}
        
        # Handle anyOf (Optional fields in Pydantic v2)
        if "anyOf" in field_def:
            any_of = field_def["anyOf"]
            # Find the non-null type
            for option in any_of:
                if option.get("type") != "null":
                    return option
            # Fallback to string if all else fails
            return {"type": "string"}
        
        # Default fallback
        return {"type": "string"}
    
    @staticmethod
    def _expand_field_definition(field_def: Dict[str, Any]) -> Dict[str, Any]:
        """
        Expand a Pydantic field definition for inline use in OpenAPI schema.
        This creates a full field definition that Swagger UI can render as individual form fields.
        
        Args:
            field_def: Pydantic field definition
            
        Returns:
            Expanded field definition for OpenAPI schema
        """
        # Return the field definition as-is since Pydantic already provides proper schemas
        return field_def.copy()
    
    @staticmethod
    def add_request_schema(
        openapi_schema: Dict[str, Any], 
        model_class: Type, 
        schema_name: str, 
        paths: List[str],
        operation_name: str
    ) -> Dict[str, Any]:
        """
        Generic method to add a request schema to OpenAPI specification.
        
        Args:
            openapi_schema: The OpenAPI schema dict to modify
            model_class: The Pydantic model class to get schema from
            schema_name: Name for the schema component
            paths: List of paths to add the request body to
            operation_name: Name of the operation for logging (e.g., "chat completion", "embedding")
            
        Returns:
            Modified OpenAPI schema
        """
        try:
            # Get the schema for the model class
            request_schema = CustomOpenAPISpec.get_pydantic_schema(model_class)
            
            # Only proceed if we successfully got the schema
            if request_schema is not None:
                # Add schema to components
                CustomOpenAPISpec.add_schema_to_components(openapi_schema, schema_name, request_schema)
                
                # Add request body to specified endpoints
                CustomOpenAPISpec.add_request_body_to_paths(
                    openapi_schema, 
                    paths, 
                    f"#/components/schemas/{schema_name}"
                )
                
                verbose_proxy_logger.debug(f"Successfully added {schema_name} schema to OpenAPI spec")
            else:
                verbose_proxy_logger.debug(f"Could not get schema for {schema_name}")
                
        except Exception as e:
            # If schema addition fails, continue without it
            verbose_proxy_logger.debug(f"Failed to add {operation_name} request schema: {str(e)}")
        
        return openapi_schema
    
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
            
            return CustomOpenAPISpec.add_request_schema(
                openapi_schema=openapi_schema,
                model_class=ProxyChatCompletionRequest,
                schema_name="ProxyChatCompletionRequest",
                paths=CustomOpenAPISpec.CHAT_COMPLETION_PATHS,
                operation_name="chat completion"
            )
        except ImportError as e:
            verbose_proxy_logger.debug(f"Failed to import ProxyChatCompletionRequest: {str(e)}")
            return openapi_schema
    
    @staticmethod
    def add_embedding_request_schema(openapi_schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add EmbeddingRequest schema to embedding endpoints for documentation.
        This shows the request body in Swagger without runtime validation.
        
        Args:
            openapi_schema: The OpenAPI schema dict to modify
            
        Returns:
            Modified OpenAPI schema
        """
        try:
            from litellm.types.embedding import EmbeddingRequest
            
            return CustomOpenAPISpec.add_request_schema(
                openapi_schema=openapi_schema,
                model_class=EmbeddingRequest,
                schema_name="EmbeddingRequest",
                paths=CustomOpenAPISpec.EMBEDDING_PATHS,
                operation_name="embedding"
            )
        except ImportError as e:
            verbose_proxy_logger.debug(f"Failed to import EmbeddingRequest: {str(e)}")
            return openapi_schema
    
    @staticmethod
    def add_responses_api_request_schema(openapi_schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add ResponsesAPIRequestParams schema to responses API endpoints for documentation.
        This shows the request body in Swagger without runtime validation.
        
        Args:
            openapi_schema: The OpenAPI schema dict to modify
            
        Returns:
            Modified OpenAPI schema
        """
        try:
            from litellm.types.llms.openai import ResponsesAPIRequestParams
            
            return CustomOpenAPISpec.add_request_schema(
                openapi_schema=openapi_schema,
                model_class=ResponsesAPIRequestParams,
                schema_name="ResponsesAPIRequestParams",
                paths=CustomOpenAPISpec.RESPONSES_API_PATHS,
                operation_name="responses API"
            )
        except ImportError as e:
            verbose_proxy_logger.debug(f"Failed to import ResponsesAPIRequestParams: {str(e)}")
            return openapi_schema
    
    @staticmethod
    def add_llm_api_request_schema_body(openapi_schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add LLM API request schema bodies to OpenAPI specification for documentation.
        
        Args:
            openapi_schema: The base OpenAPI schema
            
        Returns:
            OpenAPI schema with added request body schemas
        """
        # Add chat completion request schema
        openapi_schema = CustomOpenAPISpec.add_chat_completion_request_schema(openapi_schema)
        
        # Add embedding request schema
        openapi_schema = CustomOpenAPISpec.add_embedding_request_schema(openapi_schema)
        
        # Add responses API request schema
        openapi_schema = CustomOpenAPISpec.add_responses_api_request_schema(openapi_schema)
        
        return openapi_schema 