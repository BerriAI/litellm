"""
Unit tests for OpenAPI 3.1.0 to 3.0.3 downgrade functionality.

Tests the transformation of OpenAPI schemas from version 3.1.0 to 3.0.3
for compatibility with tools like Apigee that require 3.0.3.
"""

import pytest

from litellm.proxy.common_utils.openapi_downgrade import (
    _process_schema_object,
    convert_pydantic_v2_to_openapi_3_0_3,
    downgrade_openapi_schema_to_3_0_3,
    get_openapi_3_0_3_compatible_version,
)


class TestTypeArrayConversion:
    """Test conversion of type arrays to nullable fields."""

    def test_string_or_null_to_nullable(self):
        """Test ["string", "null"] converts to {type: "string", nullable: true}"""
        schema = {"type": ["string", "null"]}
        result = _process_schema_object(schema)
        assert result == {"type": "string", "nullable": True}

    def test_number_or_null_to_nullable(self):
        """Test ["number", "null"] converts to {type: "number", nullable: true}"""
        schema = {"type": ["number", "null"]}
        result = _process_schema_object(schema)
        assert result == {"type": "number", "nullable": True}

    def test_multiple_non_null_types_to_oneof(self):
        """Test ["string", "number"] converts to oneOf"""
        schema = {"type": ["string", "number"]}
        result = _process_schema_object(schema)
        assert "oneOf" in result
        assert {"type": "string"} in result["oneOf"]
        assert {"type": "number"} in result["oneOf"]
        assert "nullable" not in result

    def test_multiple_types_with_null_to_oneof_nullable(self):
        """Test ["string", "number", "null"] converts to oneOf + nullable"""
        schema = {"type": ["string", "number", "null"]}
        result = _process_schema_object(schema)
        assert "oneOf" in result
        assert {"type": "string"} in result["oneOf"]
        assert {"type": "number"} in result["oneOf"]
        assert result["nullable"] is True

    def test_single_type_unchanged(self):
        """Test single type string remains unchanged"""
        schema = {"type": "string"}
        result = _process_schema_object(schema)
        assert result == {"type": "string"}

    def test_only_null_type(self):
        """Test ["null"] converts to {type: "null"}"""
        schema = {"type": ["null"]}
        result = _process_schema_object(schema)
        assert result == {"type": "null"}


class TestExamplesConversion:
    """Test conversion of examples array to single example."""

    def test_examples_array_to_single_example(self):
        """Test examples array converts to single example (first item)"""
        schema = {"type": "string", "examples": ["value1", "value2", "value3"]}
        result = _process_schema_object(schema)
        assert result == {"type": "string", "example": "value1"}

    def test_empty_examples_array(self):
        """Test empty examples array is removed"""
        schema = {"type": "string", "examples": []}
        result = _process_schema_object(schema)
        assert result == {"type": "string"}
        assert "examples" not in result
        assert "example" not in result

    def test_examples_with_objects(self):
        """Test examples with complex objects"""
        schema = {
            "type": "object",
            "examples": [
                {"name": "John", "age": 30},
                {"name": "Jane", "age": 25}
            ]
        }
        result = _process_schema_object(schema)
        assert result["example"] == {"name": "John", "age": 30}


class TestUnsupportedKeywordRemoval:
    """Test removal of 3.1.0-specific keywords."""

    def test_const_removed(self):
        """Test const keyword is removed"""
        schema = {"type": "string", "const": "fixed_value"}
        result = _process_schema_object(schema)
        assert "const" not in result
        assert result["type"] == "string"

    def test_dynamic_ref_removed(self):
        """Test $dynamicRef is removed"""
        schema = {"$dynamicRef": "#meta"}
        result = _process_schema_object(schema)
        assert "$dynamicRef" not in result

    def test_dynamic_anchor_removed(self):
        """Test $dynamicAnchor is removed"""
        schema = {"$dynamicAnchor": "meta"}
        result = _process_schema_object(schema)
        assert "$dynamicAnchor" not in result

    def test_prefix_items_removed(self):
        """Test prefixItems is removed"""
        schema = {"prefixItems": [{"type": "number"}, {"type": "string"}]}
        result = _process_schema_object(schema)
        assert "prefixItems" not in result

    def test_schema_keyword_removed(self):
        """Test $schema is removed"""
        schema = {"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "string"}
        result = _process_schema_object(schema)
        assert "$schema" not in result
        assert result["type"] == "string"


class TestExclusiveMinMaxConversion:
    """Test conversion of exclusiveMinimum/exclusiveMaximum."""

    def test_exclusive_minimum_number_to_boolean(self):
        """Test exclusiveMinimum as number converts to minimum + boolean"""
        schema = {"type": "number", "exclusiveMinimum": 0}
        result = _process_schema_object(schema)
        assert result["minimum"] == 0
        assert result["exclusiveMinimum"] is True
        assert result["type"] == "number"

    def test_exclusive_maximum_number_to_boolean(self):
        """Test exclusiveMaximum as number converts to maximum + boolean"""
        schema = {"type": "number", "exclusiveMaximum": 100}
        result = _process_schema_object(schema)
        assert result["maximum"] == 100
        assert result["exclusiveMaximum"] is True

    def test_exclusive_minimum_boolean_unchanged(self):
        """Test exclusiveMinimum as boolean stays unchanged"""
        schema = {"type": "number", "minimum": 0, "exclusiveMinimum": True}
        result = _process_schema_object(schema)
        assert result["minimum"] == 0
        assert result["exclusiveMinimum"] is True

    def test_exclusive_maximum_float(self):
        """Test exclusiveMaximum with float value"""
        schema = {"type": "number", "exclusiveMaximum": 99.99}
        result = _process_schema_object(schema)
        assert result["maximum"] == 99.99
        assert result["exclusiveMaximum"] is True


class TestNestedSchemaProcessing:
    """Test recursive processing of nested schemas."""

    def test_properties_processed_recursively(self):
        """Test object properties are processed recursively"""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": ["string", "null"]},
                "age": {"type": "integer", "examples": [25, 30, 35]}
            }
        }
        result = _process_schema_object(schema)
        assert result["properties"]["name"] == {"type": "string", "nullable": True}
        assert result["properties"]["age"]["type"] == "integer"
        assert result["properties"]["age"]["example"] == 25

    def test_items_processed_recursively(self):
        """Test array items are processed recursively"""
        schema = {
            "type": "array",
            "items": {"type": ["string", "null"], "examples": ["test"]}
        }
        result = _process_schema_object(schema)
        assert result["items"] == {"type": "string", "nullable": True, "example": "test"}

    def test_allof_processed_recursively(self):
        """Test allOf schemas are processed recursively"""
        schema = {
            "allOf": [
                {"type": ["string", "null"]},
                {"minLength": 1}
            ]
        }
        result = _process_schema_object(schema)
        assert result["allOf"][0] == {"type": "string", "nullable": True}
        assert result["allOf"][1] == {"minLength": 1}

    def test_anyof_processed_recursively(self):
        """Test anyOf schemas are processed recursively"""
        schema = {
            "anyOf": [
                {"type": "string"},
                {"type": ["number", "null"]}
            ]
        }
        result = _process_schema_object(schema)
        assert result["anyOf"][0] == {"type": "string"}
        assert result["anyOf"][1] == {"type": "number", "nullable": True}

    def test_oneof_processed_recursively(self):
        """Test oneOf schemas are processed recursively"""
        schema = {
            "oneOf": [
                {"type": ["string", "null"]},
                {"type": "integer"}
            ]
        }
        result = _process_schema_object(schema)
        assert result["oneOf"][0] == {"type": "string", "nullable": True}
        assert result["oneOf"][1] == {"type": "integer"}

    def test_not_processed_recursively(self):
        """Test not schema is processed recursively"""
        schema = {
            "not": {"type": ["string", "null"]}
        }
        result = _process_schema_object(schema)
        assert result["not"] == {"type": "string", "nullable": True}

    def test_additional_properties_object_processed(self):
        """Test additionalProperties as object is processed"""
        schema = {
            "type": "object",
            "additionalProperties": {"type": ["string", "null"]}
        }
        result = _process_schema_object(schema)
        assert result["additionalProperties"] == {"type": "string", "nullable": True}

    def test_additional_properties_boolean_unchanged(self):
        """Test additionalProperties as boolean is unchanged"""
        schema = {
            "type": "object",
            "additionalProperties": False
        }
        result = _process_schema_object(schema)
        assert result["additionalProperties"] is False


class TestComplexSchemaConversion:
    """Test conversion of complex, realistic schemas."""

    def test_chat_completion_message_schema(self):
        """Test realistic chat completion message schema"""
        schema = {
            "type": "object",
            "properties": {
                "role": {
                    "type": "string",
                    "examples": ["user", "assistant", "system"]
                },
                "content": {
                    "type": ["string", "null"],
                    "description": "The message content"
                },
                "name": {
                    "type": ["string", "null"]
                },
                "function_call": {
                    "type": ["object", "null"],
                    "properties": {
                        "name": {"type": "string"},
                        "arguments": {"type": "string"}
                    }
                }
            },
            "required": ["role"]
        }
        result = _process_schema_object(schema)
        
        assert result["properties"]["role"]["type"] == "string"
        assert result["properties"]["role"]["example"] == "user"
        assert result["properties"]["content"]["type"] == "string"
        assert result["properties"]["content"]["nullable"] is True
        assert result["properties"]["name"]["type"] == "string"
        assert result["properties"]["name"]["nullable"] is True
        assert "oneOf" in result["properties"]["function_call"]
        assert result["properties"]["function_call"]["nullable"] is True
        assert result["required"] == ["role"]

    def test_pydantic_model_schema(self):
        """Test conversion of Pydantic v2 generated schema"""
        schema = {
            "$defs": {
                "Message": {
                    "type": "object",
                    "properties": {
                        "role": {"type": "string"},
                        "content": {"type": ["string", "null"]}
                    }
                }
            },
            "type": "object",
            "properties": {
                "model": {"type": "string"},
                "messages": {
                    "type": "array",
                    "items": {"$ref": "#/$defs/Message"}
                },
                "temperature": {
                    "type": ["number", "null"],
                    "exclusiveMinimum": 0,
                    "exclusiveMaximum": 2
                }
            }
        }
        result = _process_schema_object(schema)
        
        # $defs should be removed at this level (handled separately in full schema)
        assert "$defs" not in result
        assert result["properties"]["messages"]["items"]["$ref"] == "#/$defs/Message"
        assert result["properties"]["temperature"]["type"] == "number"
        assert result["properties"]["temperature"]["nullable"] is True
        assert result["properties"]["temperature"]["minimum"] == 0
        assert result["properties"]["temperature"]["exclusiveMinimum"] is True
        assert result["properties"]["temperature"]["maximum"] == 2
        assert result["properties"]["temperature"]["exclusiveMaximum"] is True


class TestFullOpenAPISchemaDowngrade:
    """Test downgrade of complete OpenAPI schemas."""

    def test_basic_openapi_3_1_to_3_0_3(self):
        """Test basic OpenAPI 3.1.0 schema conversion"""
        schema = {
            "openapi": "3.1.0",
            "info": {
                "title": "Test API",
                "version": "1.0.0"
            },
            "paths": {
                "/test": {
                    "get": {
                        "summary": "Test endpoint",
                        "responses": {
                            "200": {
                                "description": "Success"
                            }
                        }
                    }
                }
            }
        }
        result = downgrade_openapi_schema_to_3_0_3(schema)
        assert result["openapi"] == "3.0.3"
        assert result["info"]["title"] == "Test API"
        assert "/test" in result["paths"]

    def test_webhooks_removed(self):
        """Test webhooks section is removed (3.1.0 feature)"""
        schema = {
            "openapi": "3.1.0",
            "info": {"title": "Test", "version": "1.0"},
            "paths": {},
            "webhooks": {
                "newPet": {
                    "post": {
                        "summary": "New pet webhook"
                    }
                }
            }
        }
        result = downgrade_openapi_schema_to_3_0_3(schema)
        assert "webhooks" not in result
        assert result["openapi"] == "3.0.3"

    def test_license_identifier_removed(self):
        """Test license.identifier is removed (3.1.0 feature)"""
        schema = {
            "openapi": "3.1.0",
            "info": {
                "title": "Test",
                "version": "1.0",
                "license": {
                    "name": "Apache 2.0",
                    "identifier": "Apache-2.0",
                    "url": "https://www.apache.org/licenses/LICENSE-2.0.html"
                }
            },
            "paths": {}
        }
        result = downgrade_openapi_schema_to_3_0_3(schema)
        assert "identifier" not in result["info"]["license"]
        assert result["info"]["license"]["name"] == "Apache 2.0"
        assert result["info"]["license"]["url"] == "https://www.apache.org/licenses/LICENSE-2.0.html"

    def test_schema_in_components(self):
        """Test schemas in components are processed"""
        schema = {
            "openapi": "3.1.0",
            "info": {"title": "Test", "version": "1.0"},
            "paths": {},
            "components": {
                "schemas": {
                    "Pet": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "tag": {"type": ["string", "null"]}
                        }
                    }
                }
            }
        }
        result = downgrade_openapi_schema_to_3_0_3(schema)
        assert result["components"]["schemas"]["Pet"]["properties"]["name"]["type"] == "string"
        assert result["components"]["schemas"]["Pet"]["properties"]["tag"]["type"] == "string"
        assert result["components"]["schemas"]["Pet"]["properties"]["tag"]["nullable"] is True

    def test_request_body_content_schema(self):
        """Test request body content schemas are processed"""
        schema = {
            "openapi": "3.1.0",
            "info": {"title": "Test", "version": "1.0"},
            "paths": {
                "/pets": {
                    "post": {
                        "requestBody": {
                            "required": True,
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "name": {"type": ["string", "null"]}
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        result = downgrade_openapi_schema_to_3_0_3(schema)
        schema_result = result["paths"]["/pets"]["post"]["requestBody"]["content"]["application/json"]["schema"]
        assert schema_result["properties"]["name"]["type"] == "string"
        assert schema_result["properties"]["name"]["nullable"] is True

    def test_response_content_schema(self):
        """Test response content schemas are processed"""
        schema = {
            "openapi": "3.1.0",
            "info": {"title": "Test", "version": "1.0"},
            "paths": {
                "/pets": {
                    "get": {
                        "responses": {
                            "200": {
                                "description": "Success",
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "type": "array",
                                            "items": {
                                                "type": ["object", "null"]
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        result = downgrade_openapi_schema_to_3_0_3(schema)
        items_schema = result["paths"]["/pets"]["get"]["responses"]["200"]["content"]["application/json"]["schema"]["items"]
        assert "oneOf" in items_schema
        assert items_schema["nullable"] is True

    def test_parameter_schema_processing(self):
        """Test parameter schemas are processed"""
        schema = {
            "openapi": "3.1.0",
            "info": {"title": "Test", "version": "1.0"},
            "paths": {
                "/pets": {
                    "get": {
                        "parameters": [
                            {
                                "name": "limit",
                                "in": "query",
                                "schema": {
                                    "type": ["integer", "null"]
                                }
                            }
                        ]
                    }
                }
            }
        }
        result = downgrade_openapi_schema_to_3_0_3(schema)
        param_schema = result["paths"]["/pets"]["get"]["parameters"][0]["schema"]
        assert param_schema["type"] == "integer"
        assert param_schema["nullable"] is True

    def test_parameter_content_converted_to_schema(self):
        """Test parameter with content is converted to schema (3.1.0 feature)"""
        schema = {
            "openapi": "3.1.0",
            "info": {"title": "Test", "version": "1.0"},
            "paths": {
                "/pets": {
                    "get": {
                        "parameters": [
                            {
                                "name": "filter",
                                "in": "query",
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "type": "object"
                                        }
                                    }
                                }
                            }
                        ]
                    }
                }
            }
        }
        result = downgrade_openapi_schema_to_3_0_3(schema)
        param = result["paths"]["/pets"]["get"]["parameters"][0]
        assert "content" not in param
        assert "schema" in param
        assert param["schema"]["type"] == "object"

    def test_examples_in_media_type(self):
        """Test examples in media type objects are converted"""
        schema = {
            "openapi": "3.1.0",
            "info": {"title": "Test", "version": "1.0"},
            "paths": {
                "/pets": {
                    "post": {
                        "requestBody": {
                            "content": {
                                "application/json": {
                                    "schema": {"type": "object"},
                                    "examples": {
                                        "cat": {
                                            "value": {"name": "Fluffy"}
                                        },
                                        "dog": {
                                            "value": {"name": "Rex"}
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        result = downgrade_openapi_schema_to_3_0_3(schema)
        media_type = result["paths"]["/pets"]["post"]["requestBody"]["content"]["application/json"]
        assert "examples" not in media_type
        assert "example" in media_type
        assert media_type["example"] == {"name": "Fluffy"}


class TestPydanticV2SchemaConversion:
    """Test conversion of Pydantic v2 schemas."""

    def test_convert_pydantic_v2_basic(self):
        """Test basic Pydantic v2 schema conversion"""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": ["integer", "null"]}
            },
            "$defs": {
                "Address": {
                    "type": "object",
                    "properties": {
                        "street": {"type": "string"}
                    }
                }
            }
        }
        result = convert_pydantic_v2_to_openapi_3_0_3(schema)
        assert "$defs" not in result
        assert result["properties"]["age"]["type"] == "integer"
        assert result["properties"]["age"]["nullable"] is True


class TestGetCompatibleVersion:
    """Test the main entry point function."""

    def test_convert_3_1_to_3_0_3(self):
        """Test converting 3.1.0 to 3.0.3"""
        schema = {
            "openapi": "3.1.0",
            "info": {"title": "Test", "version": "1.0"},
            "paths": {}
        }
        result = get_openapi_3_0_3_compatible_version(schema)
        assert result["openapi"] == "3.0.3"

    def test_keep_3_0_x_as_3_0_3(self):
        """Test 3.0.x versions are kept but version is set to 3.0.3"""
        schema = {
            "openapi": "3.0.2",
            "info": {"title": "Test", "version": "1.0"},
            "paths": {}
        }
        result = get_openapi_3_0_3_compatible_version(schema)
        assert result["openapi"] == "3.0.3"

    def test_handle_missing_version(self):
        """Test handling of schema without version"""
        schema = {
            "info": {"title": "Test", "version": "1.0"},
            "paths": {}
        }
        result = get_openapi_3_0_3_compatible_version(schema)
        assert result["openapi"] == "3.0.3"


class TestEdgeCases:
    """Test edge cases and unusual scenarios."""

    def test_empty_schema(self):
        """Test empty schema object"""
        schema = {}
        result = _process_schema_object(schema)
        assert result == {}

    def test_non_dict_returns_unchanged(self):
        """Test non-dict values return unchanged"""
        assert _process_schema_object("string") == "string"
        assert _process_schema_object(123) == 123
        assert _process_schema_object(None) is None

    def test_list_processed_recursively(self):
        """Test lists are processed recursively"""
        schema_list = [
            {"type": ["string", "null"]},
            {"type": "integer"}
        ]
        result = _process_schema_object(schema_list)
        assert result[0] == {"type": "string", "nullable": True}
        assert result[1] == {"type": "integer"}

    def test_deeply_nested_schema(self):
        """Test deeply nested schema structure"""
        schema = {
            "type": "object",
            "properties": {
                "level1": {
                    "type": "object",
                    "properties": {
                        "level2": {
                            "type": "object",
                            "properties": {
                                "level3": {
                                    "type": ["string", "null"]
                                }
                            }
                        }
                    }
                }
            }
        }
        result = _process_schema_object(schema)
        level3 = result["properties"]["level1"]["properties"]["level2"]["properties"]["level3"]
        assert level3 == {"type": "string", "nullable": True}

    def test_mixed_type_and_nullable(self):
        """Test schema with both type and other properties"""
        schema = {
            "type": ["string", "null"],
            "minLength": 1,
            "maxLength": 100,
            "pattern": "^[a-z]+$",
            "examples": ["test", "example"]
        }
        result = _process_schema_object(schema)
        assert result["type"] == "string"
        assert result["nullable"] is True
        assert result["minLength"] == 1
        assert result["maxLength"] == 100
        assert result["pattern"] == "^[a-z]+$"
        assert result["example"] == "test"
