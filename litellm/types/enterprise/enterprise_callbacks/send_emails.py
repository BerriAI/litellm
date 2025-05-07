from pydantic import BaseModel

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
