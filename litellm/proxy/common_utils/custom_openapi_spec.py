from collections.abc import Mapping, Sequence
from typing import Final, TypeAlias, Union

from litellm._logging import verbose_proxy_logger

JsonValue: TypeAlias = Union["JsonObject", "JsonArray", str, int, float, bool, None]
JsonObject: TypeAlias = dict[str, JsonValue]
JsonArray: TypeAlias = list[JsonValue]


class CustomOpenAPISpec:
    """
    Handler for customizing OpenAPI specifications with Pydantic models
    for documentation purposes without runtime validation.
    """

    CHAT_COMPLETION_PATHS = [
        "/v1/chat/completions",
        "/chat/completions",
        "/engines/{model}/chat/completions",
        "/openai/deployments/{model}/chat/completions",
    ]

    EMBEDDING_PATHS = [
        "/v1/embeddings",
        "/embeddings",
        "/engines/{model}/embeddings",
        "/openai/deployments/{model}/embeddings",
    ]

    RESPONSES_API_PATHS = ["/v1/responses", "/responses"]

    @staticmethod
    def _as_object(node: JsonValue) -> JsonObject:
        return node if isinstance(node, dict) else {}

    @staticmethod
    def _as_array(node: JsonValue) -> JsonArray:
        return node if isinstance(node, list) else []

    @staticmethod
    def _components_schemas(openapi_schema: JsonObject) -> JsonObject:
        components: Final = CustomOpenAPISpec._as_object(openapi_schema.setdefault("components", {}))
        return CustomOpenAPISpec._as_object(components.setdefault("schemas", {}))

    @staticmethod
    def get_pydantic_schema(model_class) -> JsonObject | None:
        """
        Get JSON schema from a Pydantic model, handling both v1 and v2 APIs.

        Args:
            model_class: Pydantic model class

        Returns:
            JSON schema dict or None if failed
        """
        try:
            # Try Pydantic v2 method first
            return model_class.model_json_schema()
        except AttributeError:
            try:
                # Fallback to Pydantic v1 method
                return model_class.schema()
            except AttributeError:
                # If both methods fail, return None
                return None
        except Exception as e:
            # FastAPI 0.120+ may fail schema generation for certain types (e.g., openai.Timeout)
            # Log the error and return None to skip schema generation for this model
            verbose_proxy_logger.debug("Failed to generate schema for %s: %s", model_class, e)
            return None

    @staticmethod
    def add_schema_to_components(openapi_schema: JsonObject, schema_name: str, schema_def: JsonObject) -> None:
        """
        Add a schema definition to the OpenAPI components/schemas section.

        Args:
            openapi_schema: The OpenAPI schema dict to modify
            schema_name: Name for the schema component
            schema_def: The schema definition
        """
        # Ensure components/schemas structure exists
        _ = CustomOpenAPISpec._components_schemas(openapi_schema)

        # Add the schema
        CustomOpenAPISpec._move_defs_to_components(openapi_schema, {schema_name: schema_def})

    @staticmethod
    def _expanded_request_field(field_name: str, field_def: JsonValue) -> JsonValue:
        expanded: Final = CustomOpenAPISpec._rewrite_defs_refs(
            CustomOpenAPISpec._expand_field_definition(CustomOpenAPISpec._as_object(field_def))
        )
        if field_name != "messages":
            return expanded
        return {
            **CustomOpenAPISpec._as_object(expanded),
            "example": [{"role": "user", "content": "Hello, how are you?"}],
        }

    @staticmethod
    def add_request_body_to_paths(openapi_schema: JsonObject, paths: Sequence[str], schema_ref: str) -> None:
        """
        Add request body with expanded form fields for better Swagger UI display.
        This keeps the request body but expands it to show individual fields in the UI.

        Args:
            openapi_schema: The OpenAPI schema dict to modify
            paths: List of paths to update
            schema_ref: Reference to the schema component (e.g., "#/components/schemas/ModelName")
        """
        for path in paths:
            path_item = CustomOpenAPISpec._as_object(
                CustomOpenAPISpec._as_object(openapi_schema.get("paths")).get(path)
            )
            if "post" not in path_item:
                continue

            post_operation = CustomOpenAPISpec._as_object(path_item["post"])

            # Get the actual schema to extract ALL field definitions
            schema_name = schema_ref.split("/")[-1]  # Extract "ProxyChatCompletionRequest" from the ref
            components = CustomOpenAPISpec._as_object(openapi_schema.get("components"))
            actual_schema = CustomOpenAPISpec._as_object(
                CustomOpenAPISpec._as_object(components.get("schemas")).get(schema_name)
            )
            schema_properties = CustomOpenAPISpec._as_object(actual_schema.get("properties"))
            required_fields = actual_schema.get("required", [])

            # Extract $defs and add them to components/schemas
            # This fixes Pydantic v2 $defs not being resolvable in Swagger/OpenAPI
            if "$defs" in actual_schema:
                CustomOpenAPISpec._move_defs_to_components(
                    openapi_schema, CustomOpenAPISpec._as_object(actual_schema["$defs"])
                )

            # Create an expanded inline schema instead of just a $ref
            # This makes Swagger UI show all individual fields in the request body editor
            expanded_schema: JsonObject = {
                "type": "object",
                "required": required_fields,
                "properties": {
                    field_name: CustomOpenAPISpec._expanded_request_field(field_name, field_def)
                    for field_name, field_def in schema_properties.items()
                },
            }

            # Set the request body with the expanded schema
            post_operation["requestBody"] = {
                "required": True,
                "content": {"application/json": {"schema": expanded_schema}},
            }

            # Keep any existing parameters (like path parameters) but remove conflicting query params
            if "parameters" in post_operation:
                # Only keep path parameters, remove query params that conflict with request body
                post_operation["parameters"] = [
                    param
                    for param in CustomOpenAPISpec._as_array(post_operation["parameters"])
                    if CustomOpenAPISpec._as_object(param).get("in") == "path"
                ]

    @staticmethod
    def _move_defs_to_components(openapi_schema: JsonObject, defs: Mapping[str, JsonValue]) -> None:
        """
        Move $defs from Pydantic v2 schema to OpenAPI components/schemas.
        This makes the definitions resolvable in Swagger/OpenAPI viewers.

        Args:
            openapi_schema: The OpenAPI schema dict to modify
            defs: The $defs dictionary from Pydantic schema
        """
        if not defs:
            return

        # Ensure components/schemas exists
        schemas: Final = CustomOpenAPISpec._components_schemas(openapi_schema)

        # Add each definition to components/schemas
        for def_name, def_schema in defs.items():
            # Recursively rewrite any nested $defs references within this definition
            schemas[def_name] = CustomOpenAPISpec._rewrite_defs_refs(def_schema)

            # If this definition also has $defs, process them recursively
            def_object = CustomOpenAPISpec._as_object(def_schema)
            if "$defs" in def_object:
                CustomOpenAPISpec._move_defs_to_components(
                    openapi_schema, CustomOpenAPISpec._as_object(def_object["$defs"])
                )

    @staticmethod
    def _rewritten_defs_entry(key: str, value: JsonValue) -> JsonValue:
        if key == "$ref" and isinstance(value, str) and value.startswith("#/$defs/"):
            # Rewrite the reference to use components/schemas
            def_name: Final = value.replace("#/$defs/", "")
            return f"#/components/schemas/{def_name}"
        # Recursively process nested structures
        return CustomOpenAPISpec._rewrite_defs_refs(value)

    @staticmethod
    def _rewrite_defs_refs(schema: JsonValue) -> JsonValue:
        """
        Recursively rewrite $ref values from #/$defs/... to #/components/schemas/...
        This converts Pydantic v2 references to OpenAPI-compatible references.

        Args:
            schema: Schema object to process (can be dict, list, or primitive)

        Returns:
            Schema with rewritten references
        """
        if isinstance(schema, dict):
            return {
                key: CustomOpenAPISpec._rewritten_defs_entry(key, value)
                for key, value in schema.items()
                if key != "$defs"
            }
        if isinstance(schema, list):
            return [CustomOpenAPISpec._rewrite_defs_refs(item) for item in schema]
        return schema

    @staticmethod
    def _extract_field_schema(field_def: JsonObject) -> JsonValue:
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
            any_of: Final = CustomOpenAPISpec._as_array(field_def["anyOf"])
            # Find the non-null type
            for option in any_of:
                if CustomOpenAPISpec._as_object(option).get("type") != "null":
                    return option
            # Fallback to string if all else fails
            return {"type": "string"}

        # Default fallback
        return {"type": "string"}

    @staticmethod
    def _expand_field_definition(field_def: JsonObject) -> JsonObject:
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
        openapi_schema: JsonObject,
        model_class: type,
        schema_name: str,
        paths: Sequence[str],
        operation_name: str,
    ) -> JsonObject:
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
            request_schema: Final = CustomOpenAPISpec.get_pydantic_schema(model_class)

            # Only proceed if we successfully got the schema
            if request_schema is not None:
                # Add schema to components
                CustomOpenAPISpec.add_schema_to_components(openapi_schema, schema_name, request_schema)

                # Add request body to specified endpoints
                CustomOpenAPISpec.add_request_body_to_paths(
                    openapi_schema, paths, f"#/components/schemas/{schema_name}"
                )

                verbose_proxy_logger.debug("Successfully added %s schema to OpenAPI spec", schema_name)
            else:
                verbose_proxy_logger.debug("Could not get schema for %s", schema_name)

        except Exception as e:
            # If schema addition fails, continue without it
            verbose_proxy_logger.debug("Failed to add %s request schema: %s", operation_name, e)

        return openapi_schema

    @staticmethod
    def add_chat_completion_request_schema(
        openapi_schema: JsonObject,
    ) -> JsonObject:
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
                operation_name="chat completion",
            )
        except ImportError as e:
            verbose_proxy_logger.debug("Failed to import ProxyChatCompletionRequest: %s", e)
            return openapi_schema

    @staticmethod
    def add_embedding_request_schema(openapi_schema: JsonObject) -> JsonObject:
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
                operation_name="embedding",
            )
        except ImportError as e:
            verbose_proxy_logger.debug("Failed to import EmbeddingRequest: %s", e)
            return openapi_schema

    @staticmethod
    def add_responses_api_request_schema(
        openapi_schema: JsonObject,
    ) -> JsonObject:
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
                operation_name="responses API",
            )
        except ImportError as e:
            verbose_proxy_logger.debug("Failed to import ResponsesAPIRequestParams: %s", e)
            return openapi_schema

    @staticmethod
    def add_llm_api_request_schema_body(
        openapi_schema: JsonObject,
    ) -> JsonObject:
        """
        Add LLM API request schema bodies to OpenAPI specification for documentation.

        Args:
            openapi_schema: The base OpenAPI schema

        Returns:
            OpenAPI schema with added request body schemas
        """
        # Add chat completion request schema
        with_chat_completions: Final = CustomOpenAPISpec.add_chat_completion_request_schema(openapi_schema)

        # Add embedding request schema
        with_embeddings: Final = CustomOpenAPISpec.add_embedding_request_schema(with_chat_completions)

        # Add responses API request schema
        return CustomOpenAPISpec.add_responses_api_request_schema(with_embeddings)
