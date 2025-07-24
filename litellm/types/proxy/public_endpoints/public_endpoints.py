from typing import Dict, Optional

from pydantic import BaseModel


class PublicModelHubInfo(BaseModel):
    docs_title: str
    custom_docs_description: Optional[str]
    litellm_version: str
    useful_links: Optional[Dict[str, str]]
