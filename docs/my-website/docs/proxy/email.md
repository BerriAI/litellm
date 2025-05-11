import Image from '@theme/IdealImage';

# Email Notifications 

<Image 
  img={require('../../img/email_2.png')}
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


## Quick Start 

Get SMTP credentials to set this up
Add the following to your proxy env

```shell
SMTP_HOST="smtp.resend.com"
SMTP_USERNAME="resend"
SMTP_PASSWORD="*******"
SMTP_SENDER_EMAIL="support@alerts.litellm.ai"  # email to send alerts from: `support@alerts.litellm.ai`
```

Add `email` to your proxy config.yaml under `general_settings`

```yaml
general_settings:
  master_key: sk-1234
  alerting: ["email"]
```

That's it ! start your proxy

## Customizing Email Branding

:::info

Customizing Email Branding is an Enterprise Feature [Get in touch with us for a Free Trial](https://calendly.com/d/4mp-gd3-k5k/litellm-1-1-onboarding-chat)

:::

LiteLLM allows you to customize the:
- Logo on the Email
- Email support contact 

Set the following in your env to customize your emails

```shell
EMAIL_LOGO_URL="https://litellm-listing.s3.amazonaws.com/litellm_logo.png"  # public url to your logo
EMAIL_SUPPORT_CONTACT="support@berri.ai"                                    # Your company support email
```
