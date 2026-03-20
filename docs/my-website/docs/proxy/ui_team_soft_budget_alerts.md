import Image from '@theme/IdealImage';

# Team Soft Budget Alerts

Set a soft budget on a team and get email alerts when spending crosses the threshold — without blocking any requests.

## Overview

A **soft budget** is a spending threshold that triggers email notifications when exceeded, but **does not block requests**. This is different from a hard budget (`max_budget`), which rejects requests once the limit is reached.

<Image img={require('../../img/ui_team_soft_budget_alerts.png')} />

Team soft budget alerts let you:

- **Get notified early** — receive email alerts when a team's spend crosses the soft budget threshold
- **Keep requests flowing** — unlike hard budgets, soft budgets never block API calls
- **Target specific recipients** — send alerts to specific email addresses (e.g. team leads, finance), not just the team members
- **Work without global alerting** — team soft budget alerts are sent via email independently of Slack or other global alerting configuration

:::warning Email integration required
Team soft budget alerts are sent via email. You must have an active email integration (SendGrid, Resend, or SMTP) configured on your proxy for alerts to be delivered. See [Email Notifications](./email.md) for setup instructions.
:::

:::info Automatically active
Team soft budget alerts are **automatically active** once you configure a soft budget and at least one alerting email on a team. No additional proxy configuration or restart is needed — alerts are checked on every request.
:::

## How It Works

On every API request made with a key belonging to a team, the proxy checks:

1. Does the team have a `soft_budget` set?
2. Is the team's current `spend` >= the `soft_budget`?
3. Are there any emails configured in `soft_budget_alerting_emails`?

If all three conditions are met, an email alert is sent to the configured recipients. Alerts are **deduplicated** so the same alert is only sent once within a 24-hour window.

## How to Set Up Team Soft Budget Alerts

### 1. Navigate to the Admin UI

Go to the Admin UI (e.g. `http://localhost:4000/ui` or your `PROXY_BASE_URL/ui`).

![](https://colony-recorder.s3.amazonaws.com/files/2026-02-07/f06d75ad-25ef-4ee8-90c3-9604f8e46a1c/ascreenshot_1a6defaed1494d6da0001459511ecfd5_text_export.jpeg)

### 2. Go to Teams

Click **Teams** in the sidebar.

![](https://colony-recorder.s3.amazonaws.com/files/2026-02-07/f06d75ad-25ef-4ee8-90c3-9604f8e46a1c/ascreenshot_2d258fa280f6463b966bf7a05bb102d5_text_export.jpeg)

### 3. Select a team

Click on the team you want to configure soft budget alerts for.

![](https://colony-recorder.s3.amazonaws.com/files/2026-02-07/490f09fb-6bf5-45a8-a384-676889f34c88/ascreenshot_15cceb22abe64df0bf7d7c742ecb5b2f_text_export.jpeg)

### 4. Open team Settings

Click the **Settings** tab to view the team's configuration.

![](https://colony-recorder.s3.amazonaws.com/files/2026-02-07/28dd1bc5-7d07-462f-b277-33f885bdc07e/ascreenshot_12f2b762b5d24686801d93ad5b067e06_text_export.jpeg)

### 5. Edit Settings

Click **Edit Settings** to modify the team's budget configuration.

![](https://colony-recorder.s3.amazonaws.com/files/2026-02-07/30a483ea-7e01-4fdc-ac5f-a5572388d138/ascreenshot_0915eadd9e754a798489853b82de3cb5_text_export.jpeg)

### 6. Set the Soft Budget

Click the **Soft Budget (USD)** field and enter your desired threshold. For example, enter `0.01` for testing or a higher value like `500` for production.

![](https://colony-recorder.s3.amazonaws.com/files/2026-02-07/8b306d80-4943-4ad0-a51a-94b5ebdd6680/ascreenshot_5bb6e65c6428473fac2607f6a7f4b98a_text_export.jpeg)

### 7. Add alerting emails

Click the **Soft Budget Alerting Emails** field and enter one or more comma-separated email addresses that should receive the alert.

![](https://colony-recorder.s3.amazonaws.com/files/2026-02-07/a97c6efa-cc93-45d7-979e-d2a533f423b9/ascreenshot_2d8223ce8e934aa1bfadfb2f78aee5fc_text_export.jpeg)

### 8. Save Changes

Click **Save Changes**. The soft budget alert is now active — no proxy restart required.

![](https://colony-recorder.s3.amazonaws.com/files/2026-02-07/865ba6f1-3fc6-4c19-8e08-433561d6c3f7/ascreenshot_b2f0503ada3a479a83dc8b7d01c1f8da_text_export.jpeg)

### 9. Verify: email alert received

Once the team's spend crosses the soft budget, an email alert is sent to the configured recipients. Below is an example of the alert email:

<Image img={require('../../img/ui_team_soft_budget_email_example.png')} />

## Settings Reference

| Setting                         | Description                                                                                                                                   |
| ------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| **Soft Budget (USD)**           | The spending threshold that triggers an email alert. Requests are **not** blocked when this limit is exceeded.                                |
| **Soft Budget Alerting Emails** | Comma-separated email addresses that receive the alert when the soft budget is crossed. At least one email is required for alerts to be sent. |

:::tip Soft Budget vs. Max Budget

- **Soft Budget**: Advisory threshold — sends email alerts but does **not** block requests.
- **Max Budget**: Hard limit — blocks requests once the budget is exceeded.

You can set both on the same team to get early warnings (soft) and a hard stop (max).
:::

## API Configuration

You can also configure team soft budgets via the API when creating or updating a team:

```bash
curl -X POST 'http://localhost:4000/team/update' \
  --header 'Authorization: Bearer sk-1234' \
  --header 'Content-Type: application/json' \
  --data '{
    "team_id": "your-team-id",
    "soft_budget": 500.00,
    "metadata": {
      "soft_budget_alerting_emails": ["lead@example.com", "finance@example.com"]
    }
  }'
```

## Related Documentation

- [Email Notifications](./email.md) – Configure email integrations (Resend, SMTP) for LiteLLM Proxy
- [Alerting](./alerting.md) – Set up Slack and other alerting channels
- [Cost Tracking](./cost_tracking.md) – Track and manage spend across teams, keys, and users
