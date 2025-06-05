import enum
from typing import Dict, List

from pydantic import BaseModel, Field

from litellm.proxy._types import WebhookEvent


class EmailParams(BaseModel):
    logo_url: str
    support_contact: str
    base_url: str
    recipient_email: str


class SendKeyCreatedEmailEvent(WebhookEvent):
    virtual_key: str
    """
    The virtual key that was created

    this will be sk-123xxx, since we will be emailing this to the user to start using the key
    """


class EmailEvent(str, enum.Enum):
    virtual_key_created = "Virtual Key Created"
    new_user_invitation = "New User Invitation"


class EmailEventSettings(BaseModel):
    event: EmailEvent
    enabled: bool


class EmailEventSettingsUpdateRequest(BaseModel):
    settings: List[EmailEventSettings]


class EmailEventSettingsResponse(BaseModel):
    settings: List[EmailEventSettings]


class DefaultEmailSettings(BaseModel):
    """Default settings for email events"""

    settings: Dict[EmailEvent, bool] = Field(
        default_factory=lambda: {
            EmailEvent.virtual_key_created: False,  # Off by default
            EmailEvent.new_user_invitation: True,  # On by default
        }
    )

    def to_dict(self) -> Dict[str, bool]:
        """Convert to dictionary with string keys for storage"""
        return {event.value: enabled for event, enabled in self.settings.items()}

    @classmethod
    def get_defaults(cls) -> Dict[str, bool]:
        """Get the default settings as a dictionary with string keys"""
        return cls().to_dict()
