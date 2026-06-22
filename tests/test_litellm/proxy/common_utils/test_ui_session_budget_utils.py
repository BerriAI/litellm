import litellm
from litellm.proxy.common_utils.ui_session_budget_utils import (
    resolve_ui_session_max_budget,
)


def test_resolve_ui_session_max_budget_prefers_user_budget():
    assert resolve_ui_session_max_budget(500.0) == 500.0


def test_resolve_ui_session_max_budget_falls_back_to_session_cap():
    assert resolve_ui_session_max_budget(None) == litellm.max_ui_session_budget
