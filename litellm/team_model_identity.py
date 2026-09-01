"""Feature flag for the (team_id, model_name) deployment identity.

v1 stores a team deployment under a synthetic ``model_name_{team_id}_{uuid}`` and
translates back at every read site. v2 stores the public name verbatim and makes
``(team_id or None, model_name)`` the router's key.
"""

import os
from typing import Final


def team_model_identity_v2_enabled() -> bool:
    return os.getenv("LITELLM_TEAM_MODEL_IDENTITY_V2", "").lower() in ("1", "true", "yes")


TEAM_MODEL_NAME_PREFIX: Final = "model_name_"
