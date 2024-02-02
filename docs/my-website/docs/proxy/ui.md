import Image from '@theme/IdealImage';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# [BETA] Admin UI

:::info

This is in beta, so things may change. If you have feedback, [let us know](https://discord.com/invite/wuPM9dRgDw)

:::

Allow your users to create, view their own keys through a UI

<Image img={require('../../img/admin_ui_2.png')} />  



## Quick Start

## 1. Setup SSO/Auth for UI

<Tabs>

<TabItem value="username" label="Quick Start - Username, Password">

Set the following in your .env on the Proxy

```shell
UI_USERNAME=ishaan-litellm
UI_PASSWORD=langchain
```

On accessing the LiteLLM UI, you will be prompted to enter your username, password

</TabItem>

<TabItem value="google" label="Google SSO">

- Create a new Oauth 2.0 Client on https://console.cloud.google.com/ 

**Required .env variables on your Proxy**
```shell
# for Google SSO Login
GOOGLE_CLIENT_ID=
GOOGLE_CLIENT_SECRET=
```

- Set Redirect URL on your Oauth 2.0 Client on https://console.cloud.google.com/ 
    - Set a redirect url = `<your proxy base url>/sso/callback`
    ```shell
    https://litellm-production-7002.up.railway.app/sso/callback
    ```

</TabItem>

<TabItem value="msft" label="Microsoft SSO">

- Create a new App Registration on https://portal.azure.com/
- Create a client Secret for your App Registration

**Required .env variables on your Proxy**
```shell
MICROSOFT_CLIENT_ID="84583a4d-"
MICROSOFT_CLIENT_SECRET="nbk8Q~"
MICROSOFT_TENANT="5a39737
```
- Set Redirect URI on your App Registration on https://portal.azure.com/
    - Set a redirect url = `<your proxy base url>/sso/callback`
    ```shell
    http://localhost:4000/sso/callback
    ```

</TabItem>

</Tabs>

## 2. Start Proxy Server

```shell
litellm --config proxy_config.yaml --port 4000

# start proxy on port 4000
```

## 3. Get Admin UI Link to you on Swagger 

Your Proxy Swagger is available on the root of the Proxy: `http://localhost:4000/`

<Image img={require('../../img/ui_link.png')} />




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

