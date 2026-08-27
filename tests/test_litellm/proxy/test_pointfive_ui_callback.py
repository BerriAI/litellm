from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy._types import AllCallbacks


def test_pointfive_is_offered_in_the_ui_callback_registry():
    """The proxy ui builds its form from this registry, so an absent entry is an absent form."""
    entry = AllCallbacks().pointfive

    assert entry.litellm_callback_name == "pointfive"
    assert entry.ui_callback_name == "PointFive"


def test_the_ui_offers_the_two_settings_the_plugin_reads():
    """get_callback_env_vars is what the ui renders; it must match what the logger looks up."""
    assert tuple(CustomLogger.get_callback_env_vars("pointfive")) == ("POINTFIVE_API_KEY", "POINTFIVE_API_URL")
