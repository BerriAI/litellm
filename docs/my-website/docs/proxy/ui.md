import Image from '@theme/IdealImage';

# [BETA] Self-serve UI 

Allow your users to create their own keys through a UI

:::info

This is in beta, so things may change. If you have feedback, [let us know](https://discord.com/invite/wuPM9dRgDw)

:::

## Quick Start

Requirements: 

- Need to a SMTP server connection to send emails (e.g. [Resend](https://resend.com/docs/send-with-smtp))

[**See code**](https://github.com/BerriAI/litellm/blob/61cd800b9ffbb02c286481d2056b65c7fb5447bf/litellm/proxy/proxy_server.py#L1782)

### Step 1. Save SMTP server credentials

```env
export SMTP_HOST="my-smtp-host"
export SMTP_USERNAME="my-smtp-password"
export SMTP_PASSWORD="my-smtp-password"
export SMTP_SENDER_EMAIL="krrish@berri.ai"
```

### Step 2. Enable user auth 

In your config.yaml, 

```yaml
general_settings:
    # other changes
    allow_user_auth: true
```

This will enable:
* Users to create keys via `/key/generate` (by default, only admin can create keys)
* The `/user/auth` endpoint to send user's emails with their login credentials (key + user id)

### Step 3. Connect to UI 

You can use our hosted UI (https://dashboard.litellm.ai/) or [self-host your own](https://github.com/BerriAI/litellm/tree/main/ui). 

If you self-host, you need to save the UI url in your proxy environment as `LITELLM_HOSTED_UI`. 

Connect your proxy to your UI, by entering: 
1. The hosted proxy URL 
2. Accepted email subdomains
3. [OPTIONAL] Allowed admin emails 

<Image img={require('../../img/admin_dashboard.png')} />  

## What users will see? 

### Auth 

<Image img={require('../../img/user_auth_screen.png')} />  

### Create Keys 

<Image img={require('../../img/user_create_key_screen.png')} />  