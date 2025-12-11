#!/usr/bin/env python3
"""
Test to verify the Google GenAI transformation logic for generateContent parameters
"""
import os
import sys

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

import pytest

from litellm.llms.gemini.google_genai.transformation import GoogleGenAIConfig


def test_map_generate_content_optional_params_response_json_schema_camelcase():
    """Test that responseJsonSchema (camelCase) is passed through correctly"""
    config = GoogleGenAIConfig()
    
    generate_content_config_dict = {
        "responseJsonSchema": {
            "type": "object",
            "properties": {
                "recipe_name": {"type": "string"}
            }
        },
        "temperature": 1.0
    }
    
    result = config.map_generate_content_optional_params(
        generate_content_config_dict=generate_content_config_dict,
        model="gemini/skyhawk"
    )
    
    # responseJsonSchema should be in the result (camelCase format for Google GenAI API)
    assert "responseJsonSchema" in result
    assert result["responseJsonSchema"] == generate_content_config_dict["responseJsonSchema"]
    assert "temperature" in result
    assert result["temperature"] == 1.0


def test_map_generate_content_optional_params_response_schema_snakecase():
    """Test that response_schema (snake_case) is converted to responseJsonSchema (camelCase)"""
    config = GoogleGenAIConfig()
    
    generate_content_config_dict = {
        "response_json_schema": {
            "type": "object",
            "properties": {
                "recipe_name": {"type": "string"}
            }
        },
        "temperature": 1.0
    }
    
    result = config.map_generate_content_optional_params(
        generate_content_config_dict=generate_content_config_dict,
        model="gemini/skyhawk"
    )
    
    # response_schema should be converted to responseJsonSchema (camelCase)
    assert "responseJsonSchema" in result
    assert result["responseJsonSchema"] == generate_content_config_dict["response_json_schema"]
    assert "temperature" in result


def test_map_generate_content_optional_params_thinking_config_camelcase():
    """Test that thinkingConfig (camelCase) is passed through correctly"""
    config = GoogleGenAIConfig()
    
    generate_content_config_dict = {
        "thinkingConfig": {
            "thinkingLevel": "minimal",
            "includeThoughts": True
        },
        "temperature": 1.0
    }
    
    result = config.map_generate_content_optional_params(
        generate_content_config_dict=generate_content_config_dict,
        model="gemini/skyhawk"
    )
    
    # thinkingConfig should be in the result (camelCase format for Google GenAI API)
    assert "thinkingConfig" in result
    assert result["thinkingConfig"]["thinkingLevel"] == "minimal"
    assert result["thinkingConfig"]["includeThoughts"] is True
    assert "temperature" in result


def test_map_generate_content_optional_params_thinking_config_snakecase():
    """Test that thinking_config (snake_case) is converted to thinkingConfig (camelCase)"""
    config = GoogleGenAIConfig()
    
    generate_content_config_dict = {
        "thinking_config": {
            "thinkingLevel": "medium",
            "includeThoughts": True
        },
        "temperature": 1.0
    }
    
    result = config.map_generate_content_optional_params(
        generate_content_config_dict=generate_content_config_dict,
        model="gemini/skyhawk"
    )
    
    # thinking_config should be converted to thinkingConfig (camelCase)
    assert "thinkingConfig" in result
    assert result["thinkingConfig"]["thinkingLevel"] == "medium"
    assert result["thinkingConfig"]["includeThoughts"] is True
    assert "thinking_config" not in result  # Should not be in snake_case format
    assert "temperature" in result


def test_map_generate_content_optional_params_mixed_formats():
    """Test that both camelCase and snake_case parameters work together"""
    config = GoogleGenAIConfig()
    
    generate_content_config_dict = {
        "responseJsonSchema": {
            "type": "object",
            "properties": {
                "recipe_name": {"type": "string"}
            }
        },
        "thinking_config": {
            "thinkingLevel": "low",
            "includeThoughts": True
        },
        "temperature": 1.0,
        "max_output_tokens": 100
    }
    
    result = config.map_generate_content_optional_params(
        generate_content_config_dict=generate_content_config_dict,
        model="gemini/skyhawk"
    )
    
    # All parameters should be converted to camelCase
    assert "responseJsonSchema" in result
    assert "thinkingConfig" in result
    assert result["thinkingConfig"]["thinkingLevel"] == "low"
    assert "temperature" in result
    assert "maxOutputTokens" in result  # This one stays as-is if it's in supported list


def test_map_generate_content_optional_params_response_mime_type():
    """Test that responseMimeType is handled correctly"""
    config = GoogleGenAIConfig()
    
    generate_content_config_dict = {
        "responseMimeType": "application/json",
        "responseJsonSchema": {
            "type": "object",
            "properties": {
                "recipe_name": {"type": "string"}
            }
        }
    }
    
    result = config.map_generate_content_optional_params(
        generate_content_config_dict=generate_content_config_dict,
        model="gemini/skyhawk"
    )
    
    # responseMimeType should be passed through (it's already camelCase)
    assert "responseMimeType" in result or "response_mime_type" in result
    assert "responseJsonSchema" in result

