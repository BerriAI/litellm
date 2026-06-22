from typing import Optional

import litellm


def resolve_ui_session_max_budget(user_max_budget: Optional[float]) -> Optional[float]:
    if user_max_budget is not None:
        return user_max_budget
    return litellm.max_ui_session_budget
