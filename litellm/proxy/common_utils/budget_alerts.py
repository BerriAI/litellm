from typing import Final

from litellm.litellm_core_utils.core_helpers import as_object_list


def parse_budget_alert_recipients(raw: object) -> list[str]:
    if isinstance(raw, str):
        return [email.strip() for email in raw.split(",") if email.strip()]
    emails: Final = as_object_list(raw)
    if emails is not None:
        return [email.strip() for email in emails if isinstance(email, str) and email.strip()]
    return []
