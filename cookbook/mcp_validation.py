"""
Custom Validation Module for MCP Guardrails

This module provides custom validation functions that can be used with MCP guardrails.
These functions are called during pre-call and post-call hooks to validate tool arguments and results.
"""

import re
import json
from typing import Any, Dict, List, Optional, Union


def validate_github_request(tool_name: str, arguments: Dict[str, Any], validation_rules: Dict[str, Any]) -> Dict[str, Any]:
    """
    Custom validation function for GitHub MCP tools.
    
    Args:
        tool_name: Name of the tool being called
        arguments: Tool arguments to validate
        validation_rules: Validation rules from the guardrail configuration
    
    Returns:
        dict: Validated/modified arguments
        
    Raises:
        ValueError: If validation fails
    """
    
    # Check for required fields
    if tool_name == "create_issue":
        required_fields = ["title", "body"]
        for field in required_fields:
            if field not in arguments:
                raise ValueError(f"Required field '{field}' is missing for {tool_name}")
        
        # Validate title length
        max_title_length = validation_rules.get("max_title_length", 256)
        if len(arguments.get("title", "")) > max_title_length:
            raise ValueError(f"Title exceeds maximum length of {max_title_length} characters")
        
        # Validate body length
        max_body_length = validation_rules.get("max_body_length", 65536)
        if len(arguments.get("body", "")) > max_body_length:
            raise ValueError(f"Body exceeds maximum length of {max_body_length} characters")
        
        # Check for forbidden keywords
        forbidden_keywords = validation_rules.get("forbidden_keywords", [])
        title_lower = arguments.get("title", "").lower()
        body_lower = arguments.get("body", "").lower()
        
        for keyword in forbidden_keywords:
            if keyword.lower() in title_lower or keyword.lower() in body_lower:
                raise ValueError(f"Forbidden keyword '{keyword}' found in issue content")
        
        # Validate repository access
        allowed_repositories = validation_rules.get("allowed_repositories", [])
        if allowed_repositories:
            owner = arguments.get("owner")
            repo = arguments.get("repo")
            if owner and repo:
                repo_path = owner + "/" + repo
                if repo_path not in allowed_repositories:
                    raise ValueError(f"Repository '{repo_path}' is not in the allowed list")
    
    elif tool_name == "list_repositories":
        # Validate pagination limits
        max_per_page = validation_rules.get("max_per_page", 100)
        if "per_page" in arguments and arguments["per_page"] > max_per_page:
            arguments["per_page"] = max_per_page
    
    return arguments


def validate_zapier_webhook(tool_name: str, arguments: Dict[str, Any], validation_rules: Dict[str, Any]) -> Dict[str, Any]:
    """
    Custom validation function for Zapier MCP tools.
    
    Args:
        tool_name: Name of the tool being called
        arguments: Tool arguments to validate
        validation_rules: Validation rules from the guardrail configuration
    
    Returns:
        dict: Validated/modified arguments
        
    Raises:
        ValueError: If validation fails
    """
    
    if tool_name == "create_webhook":
        # Validate webhook URL
        webhook_url = arguments.get("url")
        if not webhook_url:
            raise ValueError("Webhook URL is required")
        
        # Check if URL is in allowed list
        allowed_urls = validation_rules.get("allowed_webhook_urls", [])
        if allowed_urls:
            if webhook_url not in allowed_urls:
                raise ValueError(f"Webhook URL '{webhook_url}' is not in the allowed list")
        
        # Validate URL format
        if not webhook_url.startswith(("http://", "https://")):
            raise ValueError("Webhook URL must start with http:// or https://")
        
        # Check for forbidden patterns
        forbidden_patterns = ["localhost", "127.0.0.1", "0.0.0.0"]
        for pattern in forbidden_patterns:
            if pattern in webhook_url:
                raise ValueError(f"Forbidden pattern '{pattern}' found in webhook URL")
    
    return arguments


def validate_general_mcp(tool_name: str, arguments: Dict[str, Any], validation_rules: Dict[str, Any]) -> Dict[str, Any]:
    """
    General validation function for MCP tools.
    
    Args:
        tool_name: Name of the tool being called
        arguments: Tool arguments to validate
        validation_rules: Validation rules from the guardrail configuration
    
    Returns:
        dict: Validated/modified arguments
        
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
    max_size = validation_rules.get("max_argument_size", 10240)
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


def validate_github_result(tool_name: str, result: Any, validation_rules: Dict[str, Any]) -> Any:
    """
    Custom validation function for GitHub MCP tool results.
    
    Args:
        tool_name: Name of the tool being called
        result: Tool result to validate
        validation_rules: Validation rules from the guardrail configuration
    
    Returns:
        Any: Validated/modified result
        
    Raises:
        ValueError: If validation fails
    """
    
    if tool_name == "create_issue":
        # Check for required result fields
        required_fields = ["id", "number", "html_url"]
        if isinstance(result, dict):
            for field in required_fields:
                if field not in result:
                    raise ValueError(f"Required result field '{field}' is missing")
        
        # Check for sensitive data in result
        if isinstance(result, dict):
            result_str = json.dumps(result).lower()
            sensitive_patterns = ["password", "secret", "token", "api_key"]
            for pattern in sensitive_patterns:
                if pattern in result_str:
                    # Remove or mask sensitive data
                    if "body" in result:
                        result["body"] = re.sub(
                            rf'{pattern}[^"]*["\']([^"\']*)["\']', 
                            f'{pattern}="***"', 
                            result["body"]
                        )
    
    return result


def validate_zapier_result(tool_name: str, result: Any, validation_rules: Dict[str, Any]) -> Any:
    """
    Custom validation function for Zapier MCP tool results.
    
    Args:
        tool_name: Name of the tool being called
        result: Tool result to validate
        validation_rules: Validation rules from the guardrail configuration
    
    Returns:
        Any: Validated/modified result
        
    Raises:
        ValueError: If validation fails
    """
    
    if tool_name == "create_webhook":
        # Validate webhook creation result
        if isinstance(result, dict):
            if "id" not in result:
                raise ValueError("Webhook creation result missing 'id' field")
            
            # Check result size
            max_result_size = validation_rules.get("max_result_size", 10000)
            result_json = json.dumps(result)
            if len(result_json.encode('utf-8')) > max_result_size:
                raise ValueError(f"Result exceeds maximum size of {max_result_size} bytes")
    
    return result


def validate_general_result(tool_name: str, result: Any, validation_rules: Dict[str, Any]) -> Any:
    """
    General validation function for MCP tool results.
    
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