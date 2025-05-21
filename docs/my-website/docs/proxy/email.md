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
