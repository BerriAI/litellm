import Image from '@theme/IdealImage';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# [BETA] Admin UI

:::info

This is in beta, so things may change. If you have feedback, [let us know](https://discord.com/invite/wuPM9dRgDw)

:::

Create + delete keys through a UI

<Image img={require('../../img/ui_3.gif')} />  



## Quick Start

- Requires proxy master key to be set 
- Requires db connected 

Follow [setup](./virtual_keys.md#setup)

### 1. Start the proxy
```bash
litellm --config /path/to/config.yaml

#INFO: Proxy running on http://0.0.0.0:8000
```

### 2. Go to UI 
```bash
http://0.0.0.0:8000/ui # <proxy_base_url>/ui
```

### 3. Create Key

<Image img={require('../../img/litellm_ui_create_key.png')} />  


## Setup SSO/Auth for UI

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