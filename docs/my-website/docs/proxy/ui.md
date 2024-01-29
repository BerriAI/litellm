import Image from '@theme/IdealImage';

# [BETA] Admin UI

:::info

This is in beta, so things may change. If you have feedback, [let us know](https://discord.com/invite/wuPM9dRgDw)

:::

Allow your users to create, view their own keys through a UI

<Image img={require('../../img/admin_ui_2.png')} />  



## Quick Start

## 1. Changes to your config.yaml

Set `allow_user_auth: true` on your config

```yaml
general_settings:
    # other changes
    allow_user_auth: true
```

## 2. Setup Google SSO - Use this to Authenticate Team Members to the UI
- Create an Oauth 2.0 Client
    <Image img={require('../../img/google_oauth2.png')} />  

    - Navigate to Google `Credenentials` 
    - Create a new Oauth client ID 
    - Set the `GOOGLE_CLIENT_ID` and `GOOGLE_CLIENT_SECRET` in your Proxy .env 
- Set Redirect URL on your Oauth 2.0 Client 
    - Click on your Oauth 2.0 client on https://console.cloud.google.com/
    - Set a redirect url = `<your proxy base url>/google-callback`
    ```
    https://litellm-production-7002.up.railway.app/google-callback
    ```
    <Image img={require('../../img/google_redirect.png')} />  
## 3. Required env variables on your Proxy

```shell
PROXY_BASE_URL="<your deployed proxy endpoint>" example PROXY_BASE_URL=https://litellm-production-7002.up.railway.app/

# for Google SSO Login
GOOGLE_CLIENT_ID=
GOOGLE_CLIENT_SECRET=
```

## 4. Use UI

👉 Get Started here: https://litellm-dashboard.vercel.app/


<!-- You can use our hosted UI (https://dashboard.litellm.ai/) or [self-host your own](https://github.com/BerriAI/litellm/tree/main/ui). 

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

### Spend Per Key

<Image img={require('../../img/spend_per_api_key.png')} />  

### Spend Per User

<Image img={require('../../img/spend_per_user.png')} />   -->

