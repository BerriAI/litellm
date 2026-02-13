import Image from '@theme/IdealImage';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Email Notifications 

<Image 
  img={require('../../img/email_2_0.png')}
  style={{width: '70%', display: 'block', margin: '0 0 2rem 0'}}
/>
<p style={{textAlign: 'left', color: '#666'}}>
  LiteLLM Email Notifications
</p>

## Overview

Send LiteLLM Proxy users emails for specific events.

| Category | Details |
|----------|---------|
| Supported Events | • User added as a user on LiteLLM Proxy<br/>• Proxy API Key created for user<br/>• Proxy API Key rotated for user |
| Supported Email Integrations | • Resend API<br/>• SMTP |

## Usage

:::info

LiteLLM Cloud: This feature is enabled for all LiteLLM Cloud users, there's no need to configure anything.

:::

### 1. Configure email integration

<Tabs>
  <TabItem value="smtp" label="SMTP">

Get SMTP credentials to set this up

```yaml showLineNumbers title="proxy_config.yaml"
litellm_settings:
    callbacks: ["smtp_email"]
```

Add the following to your proxy env

```shell showLineNumbers
SMTP_HOST="smtp.resend.com"
SMTP_TLS="True"
SMTP_PORT="587"
SMTP_USERNAME="resend"
SMTP_SENDER_EMAIL="notifications@alerts.litellm.ai"
SMTP_PASSWORD="xxxxx"
```

  </TabItem>
  <TabItem value="resend" label="Resend API">

Add `resend_email` to your proxy config.yaml under `litellm_settings`

set the following env variables

```shell showLineNumbers
RESEND_API_KEY="re_1234"
```

```yaml showLineNumbers title="proxy_config.yaml"
litellm_settings:
    callbacks: ["resend_email"]
```

  </TabItem>
  <TabItem value="sendgrid" label="SendGrid API">

Add `sendgrid_email` to your proxy config.yaml under `litellm_settings`

set the following env variables

```shell showLineNumbers
SENDGRID_API_KEY="SG.1234"
SENDGRID_SENDER_EMAIL="notifications@your-domain.com"
```

```yaml showLineNumbers title="proxy_config.yaml"
litellm_settings:
  callbacks: ["sendgrid_email"]
```

  </TabItem>
</Tabs>

### 2. Create a new user

On the LiteLLM Proxy UI, go to users > create a new user. 

After creating a new user, they will receive an email invite a the email you specified when creating the user. 

### 3. Configure Budget Alerts (Optional)

Enable budget alert emails by adding "email" to the `alerts` list in your proxy configuration:

```yaml showLineNumbers title="proxy_config.yaml"
general_settings:
  alerts: ["email"]
```

#### Budget Alert Types

**Soft Budget Alerts**: Automatically triggered when a key exceeds its soft budget limit. These alerts help you monitor spending before reaching critical thresholds.

**Max Budget Alerts**: Automatically triggered when a key reaches a specified percentage of its maximum budget (default: 80%). These alerts warn you when you're approaching budget exhaustion.

Both alert types send a maximum of one email per 24-hour period to prevent spam.

#### Configuration Options

Customize budget alert behavior using these environment variables:

```yaml showLineNumbers title=".env"
# Percentage of max budget that triggers alerts (as decimal: 0.8 = 80%)
EMAIL_BUDGET_ALERT_MAX_SPEND_ALERT_PERCENTAGE=0.8

# Time-to-live for alert deduplication in seconds (default: 24 hours)
EMAIL_BUDGET_ALERT_TTL=86400
```

## Email Templates 


### 1. User added as a user on LiteLLM Proxy

This email is send when you create a new user on LiteLLM Proxy.

<Image 
  img={require('../../img/email_event_1.png')}
  style={{width: '70%', display: 'block', margin: '0 0 2rem 0'}}
/>

**How to trigger this event**

On the LiteLLM Proxy UI, go to Users > Create User > Enter the user's email address > Create User.

<Image 
  img={require('../../img/new_user_email.png')}
  style={{width: '70%', display: 'block', margin: '0 0 2rem 0'}}
/>

### 2. Proxy API Key created for user

This email is sent when you create a new API key for a user on LiteLLM Proxy.

<Image 
  img={require('../../img/email_event_2.png')}
  style={{width: '70%', display: 'block', margin: '0 0 2rem 0'}}
/>

**How to trigger this event**

On the LiteLLM Proxy UI, go to Virtual Keys > Create API Key > Select User ID

<Image 
  img={require('../../img/key_email.png')}
  style={{width: '70%', display: 'block', margin: '0 0 2rem 0'}}
/>

On the Create Key Modal, Select Advanced Settings > Set Send Email to True.

<Image 
  img={require('../../img/key_email_2.png')}
  style={{width: '70%', display: 'block', margin: '0 0 2rem 0'}}
/>

### 3. Proxy API Key Rotated for User

This email is sent when you rotate an API key for a user on LiteLLM Proxy.

<Image 
  img={require('../../img/email_regen2.png')}
  style={{maxHeight: '600px', width: 'auto', display: 'block', margin: '0 0 2rem 0'}}
/>

**How to trigger this event**

On the LiteLLM Proxy UI, go to Virtual Keys > Click on a key > Click "Regenerate Key"

:::info

Ensure there is a `user_id` attached to the key. This would have been set when creating the key.

:::

<Image 
  img={require('../../img/email_regen.png')}
  style={{width: '70%', display: 'block', margin: '0 0 2rem 0'}}
/>

After regenerating the key, the user will receive an email notification with:
- Security-focused messaging about the rotation
- The new API key (or a placeholder if `EMAIL_INCLUDE_API_KEY=false`)
- Instructions to update their applications
- Security best practices

## Email Customization

:::info

Customizing Email Branding is an Enterprise Feature [Get in touch with us for a Free Trial](https://calendly.com/d/4mp-gd3-k5k/litellm-1-1-onboarding-chat)

:::

LiteLLM allows you to customize various aspects of your email notifications. Below is a complete reference of all customizable fields:

| Field | Environment Variable | Type | Default Value | Example | Description |
|-------|-------------------|------|---------------|---------|-------------|
| Logo URL | `EMAIL_LOGO_URL` | string | LiteLLM logo | `"https://your-company.com/logo.png"` | Public URL to your company logo |
| Support Contact | `EMAIL_SUPPORT_CONTACT` | string | support@berri.ai | `"support@your-company.com"` | Email address for user support |
| Email Signature | `EMAIL_SIGNATURE` | string (HTML) | Standard LiteLLM footer | `"<p>Best regards,<br/>Your Team</p><p><a href='https://your-company.com'>Visit us</a></p>"` | HTML-formatted footer for all emails |
| Invitation Subject | `EMAIL_SUBJECT_INVITATION` | string | "LiteLLM: New User Invitation" | `"Welcome to Your Company!"` | Subject line for invitation emails |
| Key Creation Subject | `EMAIL_SUBJECT_KEY_CREATED` | string | "LiteLLM: API Key Created" | `"Your New API Key is Ready"` | Subject line for key creation emails |
| Key Rotation Subject | `EMAIL_SUBJECT_KEY_ROTATED` | string | "LiteLLM: API Key Rotated" | `"Your API Key Has Been Rotated"` | Subject line for key rotation emails |
| Include API Key | `EMAIL_INCLUDE_API_KEY` | boolean | true | `"false"` | Whether to include the actual API key in emails (set to false for enhanced security) |
| Proxy Base URL | `PROXY_BASE_URL` | string | http://0.0.0.0:4000 | `"https://proxy.your-company.com"` | Base URL for the LiteLLM Proxy (used in email links) |


## HTML Support in Email Signature

The `EMAIL_SIGNATURE` field supports HTML formatting for rich, branded email footers. Here's an example of what you can include:

```html
<p>Best regards,<br/>The LiteLLM Team</p>
<p>
  <a href='https://docs.litellm.ai'>Documentation</a> |
  <a href='https://github.com/BerriAI/litellm'>GitHub</a>
</p>
<p style='font-size: 12px; color: #666;'>
  This is an automated message from LiteLLM Proxy
</p>
```

Supported HTML features:
- Text formatting (bold, italic, etc.)
- Line breaks (`<br/>`)
- Links (`<a href='...'>`)
- Paragraphs (`<p>`)
- Basic inline styling
- Company information and social media links
- Legal disclaimers or terms of service links

## Environment Variables

You can customize the following aspects of emails through environment variables:

```bash
# Email Branding
EMAIL_LOGO_URL="https://your-company.com/logo.png"  # Custom logo URL
EMAIL_SUPPORT_CONTACT="support@your-company.com"     # Support contact email
EMAIL_SIGNATURE="<p>Best regards,<br/>Your Company Team</p><p><a href='https://your-company.com'>Visit our website</a></p>"  # Custom HTML footer/signature

# Email Subject Lines
EMAIL_SUBJECT_INVITATION="Welcome to Your Company!"  # Subject for invitation emails
EMAIL_SUBJECT_KEY_CREATED="Your API Key is Ready"    # Subject for key creation emails
EMAIL_SUBJECT_KEY_ROTATED="Your API Key Has Been Rotated"  # Subject for key rotation emails

# Security Settings
EMAIL_INCLUDE_API_KEY="false"  # Set to false to hide API keys in emails (default: true)

# Proxy Configuration
PROXY_BASE_URL="https://proxy.your-company.com"      # Base URL for the LiteLLM Proxy (used in email links)
```

## Security: Hiding API Keys in Emails

For enhanced security, you can configure LiteLLM to **not** include actual API keys in email notifications. This is useful when:

- You want to reduce the risk of key exposure via email interception
- Your security policy requires keys to only be retrieved from the secure dashboard
- You're concerned about email forwarding or storage security

When disabled, emails will show: `[Key hidden for security - retrieve from dashboard]` instead of the actual API key.

**Configuration:**

```bash
# Hide API keys in emails (enhanced security)
EMAIL_INCLUDE_API_KEY="false"

# Include API keys in emails (default behavior)
EMAIL_INCLUDE_API_KEY="true"  # or omit this variable
```

**Behavior:**

| Setting | Key Created Email | Key Rotated Email |
|---------|------------------|-------------------|
| `true` (default) | Shows actual `sk-xxxxx` key | Shows actual `sk-xxxxx` key |
| `false` | Shows placeholder message | Shows placeholder message |

Users can always retrieve their keys from the LiteLLM Proxy dashboard.

## HTML Support in Email Signature

The `EMAIL_SIGNATURE` environment variable supports HTML formatting, allowing you to create rich, branded email footers. You can include:

- Text formatting (bold, italic, etc.)
- Line breaks using `<br/>`
- Links using `<a href='...'>`
- Paragraphs using `<p>`
- Company information and social media links
- Legal disclaimers or terms of service links

Example HTML signature:
```html
<p>Best regards,<br/>The LiteLLM Team</p>
<p>
  <a href='https://docs.litellm.ai'>Documentation</a> |
  <a href='https://github.com/BerriAI/litellm'>GitHub</a>
</p>
<p style='font-size: 12px; color: #666;'>
  This is an automated message from LiteLLM Proxy
</p>
```

## Default Templates

If environment variables are not set, LiteLLM will use default templates:

- Default logo: LiteLLM logo
- Default support contact: support@berri.ai
- Default signature: Standard LiteLLM footer
- Default subjects: "LiteLLM: \{event_message\}" (replaced with actual event message)

## Template Variables

When setting custom email subjects, you can use template variables that will be replaced with actual values:

```bash
# Examples of template variable usage
EMAIL_SUBJECT_INVITATION="Welcome to \{company_name\}!"
EMAIL_SUBJECT_KEY_CREATED="Your \{company_name\} API Key"
```

The system will automatically replace `\{event_message\}` and other template variables with their actual values when sending emails.

## FAQ 

### Why do I see "http://0.0.0.0:4000" in the email links?

The `PROXY_BASE_URL` environment variable is used to construct email links. If you are using the LiteLLM Proxy in a local environment, you will see "http://0.0.0.0:4000" in the email links.

If you are using the LiteLLM Proxy in a production environment, you will see the actual base URL of the LiteLLM Proxy.

You can set the `PROXY_BASE_URL` environment variable to the actual base URL of the LiteLLM Proxy.

```bash
PROXY_BASE_URL="https://proxy.your-company.com"
```