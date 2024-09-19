import Image from '@theme/IdealImage';

# Email Notifications 

Send an Email to your users when:
- A Proxy API Key is created for them 
- Their API Key crosses it's Budget 
- All Team members of a LiteLLM Team -> when the team crosses it's budget

<Image img={require('../../img/email_notifs.png')} style={{ width: '500px' }}/>

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
