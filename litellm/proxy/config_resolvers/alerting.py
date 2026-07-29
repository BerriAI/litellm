"""Descriptor tables for the alerting settings surfaced by /get/config/callbacks.

These reconcile the stored ``environment_variables`` blob (keyed by the
uppercase env-var names) with the process environment. SMTP_PORT and SMTP_TLS
carry the same effective defaults the mail-send path applies, so the settings
page shows the config that mail would actually use rather than a blank.
"""

from litellm.proxy.config_resolvers._descriptors import FieldDescriptor

EMAIL_DESCRIPTORS: tuple[FieldDescriptor, ...] = (
    FieldDescriptor("SMTP_HOST", "SMTP_HOST", "SMTP_HOST"),
    FieldDescriptor("SMTP_PORT", "SMTP_PORT", "SMTP_PORT", default="587"),
    FieldDescriptor("SMTP_TLS", "SMTP_TLS", "SMTP_TLS", default="True"),
    FieldDescriptor("SMTP_USERNAME", "SMTP_USERNAME", "SMTP_USERNAME", is_secret=True),
    FieldDescriptor("SMTP_PASSWORD", "SMTP_PASSWORD", "SMTP_PASSWORD", is_secret=True),
    FieldDescriptor("SMTP_SENDER_EMAIL", "SMTP_SENDER_EMAIL", "SMTP_SENDER_EMAIL"),
    FieldDescriptor("TEST_EMAIL_ADDRESS", "TEST_EMAIL_ADDRESS", "TEST_EMAIL_ADDRESS"),
    FieldDescriptor("EMAIL_LOGO_URL", "EMAIL_LOGO_URL", "EMAIL_LOGO_URL"),
    FieldDescriptor("EMAIL_SUPPORT_CONTACT", "EMAIL_SUPPORT_CONTACT", "EMAIL_SUPPORT_CONTACT"),
)

SLACK_DESCRIPTORS: tuple[FieldDescriptor, ...] = (
    FieldDescriptor("SLACK_WEBHOOK_URL", "SLACK_WEBHOOK_URL", "SLACK_WEBHOOK_URL", is_secret=True),
)
