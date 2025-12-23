"""
Loader for JSON endpoint definitions.
"""

import json
import os
from pathlib import Path
from typing import Optional

from litellm.experimental.endpoint_definitions.schema import EndpointDefinition


def load_endpoint_definition(name: str) -> EndpointDefinition:
    """
    Load an endpoint definition from a JSON file.
    
    Args:
        name: Name of the endpoint (e.g., "poc_interactions")
        
    Returns:
        Parsed EndpointDefinition
    """
    definitions_dir = Path(__file__).parent
    json_path = definitions_dir / f"{name}.json"
    
    if not json_path.exists():
        raise FileNotFoundError(f"Endpoint definition not found: {json_path}")
    
    with open(json_path, "r") as f:
        data = json.load(f)
    
    return EndpointDefinition(**data)


def get_api_key(env_vars: list) -> Optional[str]:
    """Get API key from environment variables."""
    for var in env_vars:
        key = os.environ.get(var)
        if key:
            return key
    return None

