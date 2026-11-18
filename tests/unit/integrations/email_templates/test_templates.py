from typing import Final

from litellm.integrations.email_templates.templates import SOFT_BUDGET_ALERT_EMAIL_TEMPLATE


def test_soft_budget_template_formats_with_the_fields_a_released_enterprise_package_passes() -> None:
    # pyproject pins litellm-enterprise separately; 0.1.86's send_soft_budget_alert_email passes exactly these fields
    html: Final = SOFT_BUDGET_ALERT_EMAIL_TEMPLATE.format(
        email_logo_url="logo",
        recipient_email="owner@example.com",
        soft_budget="$0.5",
        spend="$1.0",
        max_budget_info="",
        base_url="https://proxy.example.com",
        email_support_contact="support@example.com",
        email_footer="footer",
    )

    assert "Your LiteLLM API key has crossed its <b>soft budget limit of $0.5</b>" in html
