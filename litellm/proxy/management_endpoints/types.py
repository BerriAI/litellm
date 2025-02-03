"""
Types for the management endpoints

Might include fastapi/proxy requirements.txt related imports
"""

from typing import List

from fastapi_sso.sso.base import OpenID


class CustomOpenID(OpenID):
    team_ids: List[str]
