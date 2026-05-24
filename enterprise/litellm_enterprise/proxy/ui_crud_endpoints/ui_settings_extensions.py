"""Enterprise-only UI settings fields.

Registers additional fields onto the OSS ``UISettings`` model at import time.
Importing this module has the side effect of extending both the GET schema
and the PATCH allowlist served by ``/get/ui_settings`` and
``/update/ui_settings``.
"""

from pydantic.fields import FieldInfo

from litellm.proxy.ui_crud_endpoints.proxy_setting_endpoints import (
    register_extra_ui_setting,
)

register_extra_ui_setting(
    "enable_projects_ui",
    bool,
    FieldInfo(
        default=False,
        description=(
            "If enabled, shows the Projects feature in the UI sidebar and "
            "the project field in key management."
        ),
    ),
)
