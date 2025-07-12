from typing import Optional

from pydantic import BaseModel


class UiDiscoveryEndpoints(BaseModel):
    server_root_path: str
    proxy_base_url: Optional[str]
    docs_title: Optional[str]
    docs_description: Optional[str]
    litellm_version: Optional[str]
