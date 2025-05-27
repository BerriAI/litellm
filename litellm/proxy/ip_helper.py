from fastapi import Request
from typing import Dict

def get_client_ip_address(request: Request, general_settings: Dict[str, any]) -> str:
    """
    Determines the client IP address.
    Relies on Uvicorn (or a similar ASGI server) being correctly configured
    with 'forwarded_allow_ips' to set request.client.host accurately
    when behind trusted proxies.
    """
    return request.client.host if request.client else ""
