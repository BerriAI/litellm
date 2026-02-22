def test_ui_settings_has_disable_show_blog_field():
    """UISettings model must include disable_show_blog."""
    from litellm.proxy.ui_crud_endpoints.proxy_setting_endpoints import UISettings

    settings = UISettings()
    assert hasattr(settings, "disable_show_blog")
    assert settings.disable_show_blog is False  # default


def test_allowed_ui_settings_fields_contains_disable_show_blog():
    from litellm.proxy.ui_crud_endpoints.proxy_setting_endpoints import ALLOWED_UI_SETTINGS_FIELDS

    assert "disable_show_blog" in ALLOWED_UI_SETTINGS_FIELDS
