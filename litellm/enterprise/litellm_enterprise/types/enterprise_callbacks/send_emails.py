import enum

from pydantic import Field

from litellm.proxy._types import WebhookEvent
from litellm.types.integrations.model_deprecation_email import MODEL_DEPRECATION_EMAIL_EVENT
from litellm.types.llms.base import LiteLLMBaseModel


class EmailParams(LiteLLMBaseModel):
    logo_url: str
    support_contact: str
    base_url: str
    recipient_email: str
    subject: str
    signature: str


class SendKeyCreatedEmailEvent(WebhookEvent):
    virtual_key: str
    """
    The virtual key that was created
    this will be sk-123xxx, since we will be emailing this to the user to start using the key
    """


class SendKeyRotatedEmailEvent(WebhookEvent):
    virtual_key: str
    key_alias: str | None = None
    """
    The virtual key that was rotated
    this will be sk-123xxx, since we will be emailing this to the user to start using the new key
    """


class EmailEvent(str, enum.Enum):
    model_deprecations = MODEL_DEPRECATION_EMAIL_EVENT
    performance_alerts = "Performance Alerts"
    virtual_key_created = "Virtual Key Created"
    new_user_invitation = "New User Invitation"
    virtual_key_rotated = "Virtual Key Rotated"
    soft_budget_crossed = "Soft Budget Crossed"
    max_budget_alert = "Max Budget Alert"


class EmailEventSettings(LiteLLMBaseModel):
    event: EmailEvent
    enabled: bool


class EmailEventSettingsUpdateRequest(LiteLLMBaseModel):
    settings: list[EmailEventSettings]


class EmailEventSettingsResponse(LiteLLMBaseModel):
    settings: list[EmailEventSettings]


class DefaultEmailSettings(LiteLLMBaseModel):
    """Default settings for email events"""

    settings: dict[EmailEvent, bool] = Field(
        default_factory=lambda: {
            EmailEvent.model_deprecations: False,
            EmailEvent.performance_alerts: False,
            EmailEvent.virtual_key_created: True,  # On by default
            EmailEvent.new_user_invitation: True,  # On by default
            EmailEvent.virtual_key_rotated: True,  # On by default
            EmailEvent.soft_budget_crossed: True,  # On by default
            EmailEvent.max_budget_alert: True,  # On by default
        }
    )

    def to_dict(self) -> dict[str, bool]:
        """Convert to dictionary with string keys for storage"""
        return {event.value: enabled for event, enabled in self.settings.items()}

    @classmethod
    def get_defaults(cls) -> dict[str, bool]:
        """Get the default settings as a dictionary with string keys"""
        return cls().to_dict()
