"""
Utility functions for endpoint memory profiling.

Provides helper functions for:
- Filename sanitization  
- Sampling decisions
- Response status categorization
- Statistics extraction

Imports common utilities from parent memory profiler module for consistency.
"""

import re
from typing import Any, Dict, List



def sanitize_endpoint_name(endpoint: str) -> str:
    """
    Convert endpoint name/path to a safe filename.
    
    Args:
        endpoint: The endpoint path (e.g., "/chat/completions", "/v1/embeddings")
        
    Returns:
        Safe filename string
        
    Example:
        >>> sanitize_endpoint_name("/chat/completions")
        'chat_completions'
        >>> sanitize_endpoint_name("/v1/models/{model_id}")
        'v1_models_model_id'
    """
    # Remove leading/trailing slashes
    safe_name = endpoint.strip('/')
    
    # Replace slashes with underscores
    safe_name = safe_name.replace('/', '_')
    
    # Remove or replace special characters
    safe_name = re.sub(r'[^\w\s-]', '', safe_name)
    safe_name = re.sub(r'[-\s]+', '_', safe_name)
    
    return safe_name.strip('_').lower() or 'unknown_endpoint'


def format_memory(mb: float) -> str:
    """
    Format memory in human-readable form.
    
    Args:
        mb: Memory in megabytes
        
    Returns:
        Formatted string (e.g., "123.4 MB", "1.2 GB")
    """
    if mb >= 1024:
        return f"{mb / 1024:.2f} GB"
    elif mb >= 1.0:
        return f"{mb:.1f} MB"
    else:
        return f"{mb * 1024:.1f} KB"


def format_latency(seconds: float) -> str:
    """
    Format latency in human-readable form.
    
    Args:
        seconds: Latency in seconds
        
    Returns:
        Formatted string (e.g., "1.234s", "123.4ms", "12.3µs")
    """
    if seconds >= 1.0:
        return f"{seconds:.3f}s"
    elif seconds >= 0.001:
        return f"{seconds * 1000:.1f}ms"
    else:
        return f"{seconds * 1_000_000:.1f}µs"


def get_response_status_category(status_code: int) -> str:
    """
    Categorize HTTP response status code.
    
    Args:
        status_code: HTTP status code (e.g., 200, 404, 500)
        
    Returns:
        Category string ("success", "client_error", "server_error", "other")
    """
    if 200 <= status_code < 300:
        return "success"
    elif 400 <= status_code < 500:
        return "client_error"
    elif 500 <= status_code < 600:
        return "server_error"
    else:
        return "other"


def is_error_response(status_code: int) -> bool:
    """
    Check if HTTP status code indicates an error.
    
    Args:
        status_code: HTTP status code
        
    Returns:
        True if status code is 4xx or 5xx
    """
    return status_code >= 400


def extract_basic_stats(profiles: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Extract basic statistics from a list of profile entries.
    
    Args:
        profiles: List of profile dictionaries
        
    Returns:
        Dictionary with aggregated statistics
    """
    if not profiles:
        return {
            'total_requests': 0,
            'avg_latency': 0.0,
            'max_latency': 0.0,
            'min_latency': 0.0,
            'avg_memory': 0.0,
            'error_rate': 0.0,
        }
    
    latencies = [p['latency'] for p in profiles if 'latency' in p]
    memories = []
    for p in profiles:
        if 'memory' in p and 'current_mb' in p['memory']:
            memories.append(p['memory']['current_mb'])
    
    errors = sum(1 for p in profiles if p.get('had_error', False))
    
    return {
        'total_requests': len(profiles),
        'avg_latency': sum(latencies) / len(latencies) if latencies else 0.0,
        'max_latency': max(latencies) if latencies else 0.0,
        'min_latency': min(latencies) if latencies else 0.0,
        'avg_memory': sum(memories) / len(memories) if memories else 0.0,
        'error_rate': (errors / len(profiles)) * 100 if profiles else 0.0,
    }


def should_sample_request(request_counter: int, sampling_rate: float) -> bool:
    """
    Determine if current request should be profiled based on sampling rate.
    
    Uses deterministic sampling based on counter for consistent rate.
    
    Args:
        request_counter: Global request counter
        sampling_rate: Sampling rate (0.0 to 1.0)
        
    Returns:
        True if request should be sampled
        
    Example:
        >>> should_sample_request(1, 1.0)
        True
        >>> should_sample_request(5, 0.1)  # Sample every 10th request
        False
    """
    if sampling_rate >= 1.0:
        return True
    elif sampling_rate <= 0.0:
        return False
    
    # Sample based on rate (e.g., 0.1 means sample every 10th request)
    interval = int(1.0 / sampling_rate)
    return (request_counter % interval) == 0
