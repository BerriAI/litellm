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
| Supported Events | • User added as a user on LiteLLM Proxy<br/>• Proxy API Key created for user |
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
</Tabs>

### 2. Create a new user

On the LiteLLM Proxy UI, go to users > create a new user. 

After creating a new user, they will receive an email invite a the email you specified when creating the user. 

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
```

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
