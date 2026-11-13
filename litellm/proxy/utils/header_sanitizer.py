\"\"\"
Header sanitization and guard utilities for LiteLLM proxy authentication forwarding.
\"\"\"
from typing import Dict, Any, Optional

def sanitize_proxy_headers(headers: Optional[Dict[str, Any]]) -> Dict[str, str]:
    if not headers:
        return {}
    
    sanitized: Dict[str, str] = {}
    for k, v in headers.items():
        if k is None:
            continue
        key_str = str(k).strip()
        if not key_str:
            continue
        val_str = "" if v is None else str(v).strip()
        sanitized[key_str] = val_str
        
    return sanitized