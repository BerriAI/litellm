"""
Test Gemini format filtering for tool schemas.

This test verifies that the GoogleAIStudioGeminiConfig correctly filters out
unsupported format values from tool schemas, addressing GitHub issue #11427.
"""

import pytest
from litellm.llms.gemini.chat.transformation import GoogleAIStudioGeminiConfig


class TestGeminiFormatFiltering:
    def test_gemini_unsupported_format_removal(self):
        """Test that unsupported formats like 'email' are removed from tool schemas."""
        config = GoogleAIStudioGeminiConfig()
        
        test_tools = [
            {
                "type": "function",
                "function": {
                    "name": "git_commit",
                    "description": "Commits changes to the Git repository",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "author": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string", "description": "Author name for the commit"},
                                    "email": {
                                        "type": "string", 
                                        "format": "email", 
                                        "description": "Author email for the commit"
                                    },
                                },
                                "required": ["name", "email"],
                                "additionalProperties": False,
                            },
                        },
                        "required": ["message"],
                        "additionalProperties": False,
                    },
                },
            }
        ]
        
        # Apply the transformation
        result = config._map_function(test_tools)
        
        # Verify the structure is correct
        assert len(result) == 1
        tool = result[0]
        assert "function_declarations" in tool
        assert len(tool["function_declarations"]) == 1
        
        func_declaration = tool["function_declarations"][0]
        assert "parameters" in func_declaration
        parameters = func_declaration["parameters"]
        
        # Check that the email format was removed but other properties remain
        author_props = parameters["properties"]["author"]["properties"]
        email_prop = author_props["email"]
        
        # The 'format' field should be removed for unsupported formats
        assert "format" not in email_prop
        # But other properties should remain
        assert email_prop["type"] == "string"
        assert email_prop["description"] == "Author email for the commit"

    def test_gemini_supported_format_preservation(self):
        """Test that supported formats like 'date-time' are preserved."""
        config = GoogleAIStudioGeminiConfig()
        
        test_tools = [
            {
                "type": "function",
                "function": {
                    "name": "schedule_event",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "start_time": {
                                "type": "string",
                                "format": "date-time",
                                "description": "Event start time"
                            },
                            "category": {
                                "type": "string",
                                "enum": ["work", "personal", "other"],
                                "description": "Event category"
                            }
                        }
                    }
                }
            }
        ]
        
        result = config._map_function(test_tools)
        parameters = result[0]["function_declarations"][0]["parameters"]
        
        # date-time format should be preserved
        start_time_prop = parameters["properties"]["start_time"]
        assert "format" in start_time_prop
        assert start_time_prop["format"] == "date-time"
        
        # enum should be preserved (it's not a format issue)
        category_prop = parameters["properties"]["category"]
        assert "enum" in category_prop
        assert category_prop["enum"] == ["work", "personal", "other"]

    def test_gemini_nested_format_filtering(self):
        """Test that format filtering works recursively in nested objects and arrays."""
        config = GoogleAIStudioGeminiConfig()
        
        test_tools = [
            {
                "type": "function",
                "function": {
                    "name": "complex_function",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "nested_object": {
                                "type": "object",
                                "properties": {
                                    "email": {"type": "string", "format": "email"},
                                    "datetime": {"type": "string", "format": "date-time"}
                                }
                            },
                            "array_field": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "uri": {"type": "string", "format": "uri"},
                                        "timestamp": {"type": "string", "format": "date-time"}
                                    }
                                }
                            }
                        }
                    }
                }
            }
        ]
        
        result = config._map_function(test_tools)
        parameters = result[0]["function_declarations"][0]["parameters"]
        
        # Check nested object
        nested_obj = parameters["properties"]["nested_object"]["properties"]
        assert "format" not in nested_obj["email"]  # email format removed
        assert nested_obj["datetime"]["format"] == "date-time"  # date-time preserved
        
        # Check array items
        array_items = parameters["properties"]["array_field"]["items"]["properties"]
        assert "format" not in array_items["uri"]  # uri format removed
        assert array_items["timestamp"]["format"] == "date-time"  # date-time preserved

    def test_gemini_format_filtering_preserves_other_fields(self):
        """Test that filtering only removes the format field and preserves all other schema properties."""
        config = GoogleAIStudioGeminiConfig()
        
        test_tools = [
            {
                "type": "function", 
                "function": {
                    "name": "test_function",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "email_field": {
                                "type": "string",
                                "format": "email", 
                                "description": "Email description",
                                "minLength": 5,
                                "maxLength": 100,
                                "pattern": ".*@.*",
                                "title": "Email Field"
                            }
                        }
                    }
                }
            }
        ]
        
        result = config._map_function(test_tools)
        parameters = result[0]["function_declarations"][0]["parameters"]
        email_field = parameters["properties"]["email_field"]
        
        # Format should be removed
        assert "format" not in email_field
        
        # All other fields should be preserved
        assert email_field["type"] == "string"
        assert email_field["description"] == "Email description"
        assert email_field["minLength"] == 5
        assert email_field["maxLength"] == 100
        assert email_field["pattern"] == ".*@.*"
        assert email_field["title"] == "Email Field" 