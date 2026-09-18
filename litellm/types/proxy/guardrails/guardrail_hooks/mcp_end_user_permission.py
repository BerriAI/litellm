from .base import GuardrailConfigModel


class MCPEndUserPermissionGuardrailConfigModel(GuardrailConfigModel):
    """
    No provider-specific params required — permissions come from the end user
    object already stored in the database.
    """

    @staticmethod
    def ui_friendly_name() -> str:
        return "MCP End User Permission"
