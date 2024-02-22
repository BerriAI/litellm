import Image from '@theme/IdealImage';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# 🔑 [BETA] Proxy UI 
### **Create + delete keys through a UI**

[Let users create their own keys](#setup-ssoauth-for-ui)

:::info

This is in beta, so things may change. If you have feedback, [let us know](https://discord.com/invite/wuPM9dRgDw)

:::

<Image img={require('../../img/litellm_ui_create_key.png')} />  



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


### 3. Get Admin UI Link on Swagger 
Your Proxy Swagger is available on the root of the Proxy: e.g.: `http://localhost:4000/`

<Image img={require('../../img/ui_link.png')} />

### 4. Change default username + password

Set the following in your .env on the Proxy

```shell
UI_USERNAME=ishaan-litellm
UI_PASSWORD=langchain
```

On accessing the LiteLLM UI, you will be prompted to enter your username, password

## ✨ Enterprise Features

### Setup SSO/Auth for UI

#### Step 1: Set upperbounds for keys
Control the upperbound that users can use for `max_budget`, `budget_duration` or any `key/generate` param per key. 

```yaml
litellm_settings:
  upperbound_key_generate_params:
    max_budget: 100 # upperbound of $100, for all /key/generate requests
    duration: "30d" # upperbound of 30 days for all /key/generate requests
```

** Expected Behavior **

- Send a `/key/generate` request with `max_budget=200`
- Key will be created with `max_budget=100` since 100 is the upper bound

#### Step 2: Setup Oauth Client
<Tabs>
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


<TabItem value="Generic" label="Generic SSO Provider">

A generic OAuth client that can be used to quickly create support for any OAuth provider with close to no code

**Required .env variables on your Proxy**
```shell

GENERIC_CLIENT_ID = "******"
GENERIC_CLIENT_SECRET = "G*******"
GENERIC_AUTHORIZATION_ENDPOINT = "http://localhost:9090/auth"
GENERIC_TOKEN_ENDPOINT = "http://localhost:9090/token"
GENERIC_USERINFO_ENDPOINT = "http://localhost:9090/me"
```

**Optional .env variables**
The following can be used to customize attribute names when interacting with the generic OAuth provider. We will read these attributes from the SSO Provider result

```shell
GENERIC_USER_ID_ATTRIBUTE = "given_name"
GENERIC_USER_EMAIL_ATTRIBUTE = "family_name"
GENERIC_USER_DISPLAY_NAME_ATTRIBUTE = "display_name"
GENERIC_USER_FIRST_NAME_ATTRIBUTE = "first_name"
GENERIC_USER_LAST_NAME_ATTRIBUTE = "last_name"
GENERIC_USER_ROLE_ATTRIBUTE = "given_role"
GENERIC_CLIENT_STATE = "some-state" # if the provider needs a state parameter
GENERIC_INCLUDE_CLIENT_ID = "false" # some providers enforce that the client_id is not in the body
GENERIC_SCOPE = "openid profile email" # default scope openid is sometimes not enough to retrieve basic user info like first_name and last_name located in profile scope
```

- Set Redirect URI, if your provider requires it
    - Set a redirect url = `<your proxy base url>/sso/callback`
    ```shell
    http://localhost:4000/sso/callback
    ```

</TabItem>

</Tabs>

#### Step 3. Test flow
<Image img={require('../../img/litellm_ui_3.gif')} />

### Set Admin view w/ SSO 

You just need to set Proxy Admin ID

#### Step 1: Copy your ID from the UI 

<Image img={require('../../img/litellm_ui_copy_id.png')} />

#### Step 2: Set it in your .env as the PROXY_ADMIN_ID 

```env
export PROXY_ADMIN_ID="116544810872468347480"
```

#### Step 3: See all proxy keys

<Image img={require('../../img/litellm_ui_admin.png')} />

:::info

If you don't see all your keys this could be due to a cached token. So just re-login and it should work.

:::

### Custom Branding Admin UI

Use your companies custom branding on the LiteLLM Admin UI
We allow you to 
- Customize the UI Logo
- Customize the UI color scheme
<Image img={require('../../img/litellm_custom_ai.png')} />

#### Usage
- Navigate to [/enterprise/enterprise_ui](https://github.com/BerriAI/litellm/blob/main/enterprise/enterprise_ui/_enterprise_colors.json)
- Inside the `enterprise_ui` directory, rename `_enterprise_colors.json` to `enterprise_colors.json`
- Set your companies custom color scheme in `enterprise_colors.json`
Example contents of `enterprise_colors.json` 
Set your colors to any of the following colors: https://www.tremor.so/docs/layout/color-palette#default-colors
```json
{
    "brand": {
      "DEFAULT": "teal",
      "faint": "teal",
      "muted": "teal",
      "subtle": "teal",
      "emphasis": "teal",
      "inverted": "teal"
    }
}

```

- Set the path to your custom png/jpg logo as `UI_LOGO_PATH` in your .env
- Deploy LiteLLM Proxy Server


