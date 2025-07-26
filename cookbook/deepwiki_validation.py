"""
Custom Validation Module for DeepWiki MCP Guardrails

This module provides custom validation functions for DeepWiki MCP tools.
These functions are called during pre-call and post-call hooks to validate tool arguments and results.
"""

import re
import json
from typing import Any, Dict, List, Optional, Union


def validate_deepwiki_request(tool_name: str, arguments: Dict[str, Any], validation_rules: Dict[str, Any]) -> Dict[str, Any]:
    """
    Custom validation function for DeepWiki MCP tools.
    
    Args:
        tool_name: Name of the tool being called
        arguments: Tool arguments to validate
        validation_rules: Validation rules from the guardrail configuration
    
    Returns:
        Dict[str, Any]: Validated/modified arguments
        
    Raises:
        ValueError: If validation fails
    """
    
    # Check for forbidden keywords in any string arguments
    forbidden_keywords = validation_rules.get("forbidden_keywords", [])
    for key, value in arguments.items():
        if isinstance(value, str):
            value_lower = value.lower()
            for keyword in forbidden_keywords:
                if keyword.lower() in value_lower:
                    raise ValueError(f"Forbidden keyword '{keyword}' found in argument '{key}'")
    
    # Validate query length for search tools
    if "query" in arguments:
        max_query_length = validation_rules.get("max_query_length", 1000)
        if len(arguments["query"]) > max_query_length:
            raise ValueError(f"Query exceeds maximum length of {max_query_length} characters")
    
    # Validate search type if specified
    if "search_type" in arguments:
        allowed_search_types = validation_rules.get("allowed_search_types", [])
        if allowed_search_types and arguments["search_type"] not in allowed_search_types:
            raise ValueError(f"Search type '{arguments['search_type']}' is not allowed. Allowed types: {allowed_search_types}")
    
    # Validate results count limit
    if "max_results" in arguments:
        max_results_count = validation_rules.get("max_results_count", 50)
        if arguments["max_results"] > max_results_count:
            arguments["max_results"] = max_results_count  # Cap the results count
    
    return arguments


def validate_search_request(tool_name: str, arguments: Dict[str, Any], validation_rules: Dict[str, Any]) -> Dict[str, Any]:
    """
    Custom validation function for DeepWiki search tools.
    
    Args:
        tool_name: Name of the tool being called
        arguments: Tool arguments to validate
        validation_rules: Validation rules from the guardrail configuration
    
    Returns:
        Dict[str, Any]: Validated/modified arguments
        
    Raises:
        ValueError: If validation fails
    """
    
    if tool_name == "search_wikipedia":
        # Check for required query field
        if "query" not in arguments:
            raise ValueError("Search query is required")
        
        query = arguments["query"]
        
        # Validate minimum query length
        min_query_length = validation_rules.get("min_query_length", 3)
        if len(query) < min_query_length:
            raise ValueError(f"Search query must be at least {min_query_length} characters long")
        
        # Validate maximum query length
        max_query_length = validation_rules.get("max_query_length", 500)
        if len(query) > max_query_length:
            raise ValueError(f"Search query exceeds maximum length of {max_query_length} characters")
        
        # Check for forbidden search terms
        forbidden_search_terms = validation_rules.get("forbidden_search_terms", [])
        query_lower = query.lower()
        for term in forbidden_search_terms:
            if term.lower() in query_lower:
                raise ValueError(f"Forbidden search term '{term}' found in query")
        
        # Validate language parameter if present
        if "language" in arguments:
            allowed_languages = ["en", "es", "fr", "de", "it", "pt", "ru", "ja", "zh", "ko"]
            if arguments["language"] not in allowed_languages:
                raise ValueError(f"Language '{arguments['language']}' is not supported. Supported languages: {allowed_languages}")
    
    return arguments


def validate_general_deepwiki(tool_name: str, arguments: Dict[str, Any], validation_rules: Dict[str, Any]) -> Dict[str, Any]:
    """
    General validation function for DeepWiki MCP tools.
    
    Args:
        tool_name: Name of the tool being called
        arguments: Tool arguments to validate
        validation_rules: Validation rules from the guardrail configuration
    
    Returns:
        Dict[str, Any]: Validated/modified arguments
        
    Raises:
        ValueError: If validation fails
    """
    
    # Check tool name patterns
    allowed_patterns = validation_rules.get("allowed_tool_patterns", [])
    forbidden_patterns = validation_rules.get("forbidden_tool_patterns", [])
    
    # Check if tool matches forbidden patterns
    for pattern in forbidden_patterns:
        if re.match(pattern.replace("*", ".*"), tool_name):
            raise ValueError(f"Tool '{tool_name}' matches forbidden pattern '{pattern}'")
    
    # Check if tool matches allowed patterns (if specified)
    if allowed_patterns:
        tool_allowed = False
        for pattern in allowed_patterns:
            if re.match(pattern.replace("*", ".*"), tool_name):
                tool_allowed = True
                break
        
        if not tool_allowed:
            raise ValueError(f"Tool '{tool_name}' does not match any allowed patterns")
    
    # Validate argument size
    max_size = validation_rules.get("max_argument_size", 2048)
    args_json = json.dumps(arguments)
    if len(args_json.encode('utf-8')) > max_size:
        raise ValueError(f"Arguments exceed maximum size of {max_size} bytes")
    
    # Check for forbidden patterns in arguments
    forbidden_patterns = ["<script>", "javascript:", "data:", "vbscript:"]
    args_str = json.dumps(arguments).lower()
    for pattern in forbidden_patterns:
        if pattern in args_str:
            raise ValueError(f"Forbidden pattern '{pattern}' found in arguments")
    
    return arguments


def validate_deepwiki_result(tool_name: str, result: Any, validation_rules: Dict[str, Any]) -> Any:
    """
    Custom validation function for DeepWiki MCP tool results.
    
    Args:
        tool_name: Name of the tool being called
        result: Tool result to validate
        validation_rules: Validation rules from the guardrail configuration
    
    Returns:
        Any: Validated/modified result
        
    Raises:
        ValueError: If validation fails
    """
    
    if tool_name == "search_wikipedia":
        # Check for required result fields
        if isinstance(result, dict):
            required_fields = ["results", "total_count"]
            for field in required_fields:
                if field not in result:
                    raise ValueError(f"Required result field '{field}' is missing")
            
            # Validate results count
            if "results" in result and isinstance(result["results"], list):
                max_results = validation_rules.get("max_results_count", 50)
                if len(result["results"]) > max_results:
                    # Truncate results to max count
                    result["results"] = result["results"][:max_results]
                    result["total_count"] = min(result.get("total_count", 0), max_results)
            
            # Check for sensitive data in result
            result_str = json.dumps(result).lower()
            sensitive_patterns = ["password", "secret", "token", "api_key", "private_key"]
            for pattern in sensitive_patterns:
                if pattern in result_str:
                    # Log warning but don't modify result
                    print(f"Warning: Sensitive pattern '{pattern}' found in DeepWiki search result")
    
    return result


def validate_search_result(tool_name: str, result: Any, validation_rules: Dict[str, Any]) -> Any:
    """
    Custom validation function for DeepWiki search tool results.
    
    Args:
        tool_name: Name of the tool being called
        result: Tool result to validate
        validation_rules: Validation rules from the guardrail configuration
    
    Returns:
        Any: Validated/modified result
        
    Raises:
        ValueError: If validation fails
    """
    
    if tool_name == "search_wikipedia":
        # Validate search result structure
        if isinstance(result, dict):
            if "results" not in result:
                raise ValueError("Search result missing 'results' field")
            
            # Check result size
            max_result_size = validation_rules.get("max_result_size", 10000)
            result_json = json.dumps(result)
            if len(result_json.encode('utf-8')) > max_result_size:
                raise ValueError(f"Result exceeds maximum size of {max_result_size} bytes")
            
            # Validate individual result items
            if "results" in result and isinstance(result["results"], list):
                for i, item in enumerate(result["results"]):
                    if isinstance(item, dict):
                        # Ensure each result has required fields
                        if "title" not in item:
                            print(f"Warning: Search result {i} missing 'title' field")
                        if "url" not in item:
                            print(f"Warning: Search result {i} missing 'url' field")
    
    return result


def validate_general_result(tool_name: str, result: Any, validation_rules: Dict[str, Any]) -> Any:
    """
    General validation function for DeepWiki MCP tool results.
    
    Args:
        tool_name: Name of the tool being called
        result: Tool result to validate
        validation_rules: Validation rules from the guardrail configuration
    
    Returns:
        Any: Validated/modified result
        
    Raises:
        ValueError: If validation fails
    """
    
    # Check result size
    max_result_size = validation_rules.get("max_result_size", 50000)
    if result is not None:
        result_json = json.dumps(result)
        if len(result_json.encode('utf-8')) > max_result_size:
            raise ValueError(f"Result exceeds maximum size of {max_result_size} bytes")
    
    return result 