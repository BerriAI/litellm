from typing import Dict, Optional

from pydantic import BaseModel


class GetRouterSettingsResponse(BaseModel):
    """
    Response body for getting router settings

    Add any router params you want to allow UI users to get
    """

    model_group_alias: Optional[Dict[str, str]] = {}


class PatchRouterSettingsRequest(BaseModel):
    """
    Request body for patching router settings

    Add any router params you want to allow UI users to patch
    """

    model_group_alias: Optional[Dict[str, str]] = {}
