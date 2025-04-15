"""
ASI JSON Extraction Module

This module provides a clean, generalized approach to JSON extraction from ASI responses.
It avoids overly specific pattern matching and focuses on core extraction capabilities.
"""

import json
import re
from typing import Optional

def extract_json(content: str) -> Optional[str]:
    """
    Extract JSON from content using a simplified, generalized approach.
    This avoids overly specific pattern matching for particular content types.
    
    Args:
        content: The text content to extract JSON from
        
    Returns:
        A JSON string if extraction is successful, None otherwise
    """
    if not content:
        return None
    
    # 1. First attempt: Check for markdown code blocks
    json_block_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
    json_matches = re.findall(json_block_pattern, content)
    
    if json_matches:
        # Try each match until we find valid JSON
        for json_str in json_matches:
            try:
                parsed_json = json.loads(json_str)
                return json.dumps(parsed_json)
            except json.JSONDecodeError:
                continue
    
    # 2. Second attempt: Try to find JSON objects or arrays directly
    # Look for patterns that might be JSON objects or arrays
    json_patterns = [
        r'(\{[^{]*\})',  # JSON objects
        r'(\[[^[]*\])'   # JSON arrays
    ]
    
    for pattern in json_patterns:
        matches = re.findall(pattern, content)
        for match in matches:
            try:
                parsed_json = json.loads(match)
                return json.dumps(parsed_json)
            except json.JSONDecodeError:
                continue
    
    # 3. Third attempt: Try to parse numbered or bulleted lists
    # This handles formats like "1. Item - Description" or "* Key: Value"
    list_pattern = r'^\s*(?:\d+\.|-|\*|•)\s+(.+?)(?:\s*-\s*|:\s*|\s+)(.+?)$'
    list_matches = re.findall(list_pattern, content, re.MULTILINE)
    
    if list_matches and len(list_matches) > 0:
        # Convert the list to a JSON object with a generic structure
        items = []
        for match in list_matches:
            key = match[0].strip().replace('**', '').replace('*', '') # Remove markdown formatting
            value = match[1].strip().replace('**', '').replace('*', '')
            items.append({"key": key, "value": value})
        
        # Return a generic items array
        return json.dumps({"items": items})
    
    # 4. Fourth attempt: For unstructured text, return as a simple text object
    return json.dumps({"text": content})
