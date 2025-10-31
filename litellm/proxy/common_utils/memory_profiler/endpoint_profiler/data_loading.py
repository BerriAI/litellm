"""
Data loading utilities for endpoint memory profiling.

This module provides functions for loading and extracting data from profile files.

Features:
- Load profile data from JSON files
- Extract request numbers from IDs
- Sort and prepare data for analysis
"""

import json
from typing import Any, Dict, List


def load_profile_data(file_path: str) -> List[Dict[str, Any]]:
    """
    Load profile data from JSON file.
    
    Args:
        file_path: Path to profile JSON file
        
    Returns:
        List of profile dictionaries
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file doesn't contain a list
        
    Example:
        >>> profiles = load_profile_data("endpoint_profiles/chat_completions.json")
        >>> len(profiles)
        100
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {file_path}, got {type(data)}")
    
    return data


def extract_request_number(request_id: str) -> int:
    """
    Extract numeric request number from request_id.
    
    Args:
        request_id: Request ID like "req-123"
        
    Returns:
        Numeric part (123), or 0 if parsing fails
        
    Example:
        >>> extract_request_number("req-123")
        123
        >>> extract_request_number("invalid")
        0
    """
    try:
        return int(request_id.split('-')[1])
    except (IndexError, ValueError):
        return 0


def sort_profiles_by_request_id(profiles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Sort profiles by request ID in ascending order.
    
    Args:
        profiles: List of profile dictionaries
        
    Returns:
        Sorted list of profiles
        
    Example:
        >>> profiles = [{"request_id": "req-2"}, {"request_id": "req-1"}]
        >>> sorted_profiles = sort_profiles_by_request_id(profiles)
        >>> sorted_profiles[0]["request_id"]
        'req-1'
    """
    return sorted(
        profiles,
        key=lambda p: extract_request_number(p.get('request_id', 'req-0'))
    )

