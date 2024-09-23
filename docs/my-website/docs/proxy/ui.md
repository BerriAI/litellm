import Image from '@theme/IdealImage';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Quick Start

Create keys, track spend, add models without worrying about the config / CRUD endpoints.

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

#INFO: Proxy running on http://0.0.0.0:4000
```

### 2. Go to UI 
```bash
http://0.0.0.0:4000/ui # <proxy_base_url>/ui
```


### 3. Get Admin UI Link on Swagger 
Your Proxy Swagger is available on the root of the Proxy: e.g.: `http://localhost:4000/`

<Image img={require('../../img/ui_link.png')} />

### 4. Change default username + password

Set the following in your .env on the Proxy

```shell
LITELLM_MASTER_KEY="sk-1234" # this is your master key for using the proxy server
UI_USERNAME=ishaan-litellm   # username to sign in on UI
UI_PASSWORD=langchain        # password to sign in on UI
```

On accessing the LiteLLM UI, you will be prompted to enter your username, password

## Invite-other users 

Allow others to create/delete their own keys. 

[**Go Here**](./self_serve.md)

## ✨ Enterprise Features

Features here are behind a commercial license in our `/enterprise` folder. [**See Code**](https://github.com/BerriAI/litellm/tree/main/enterprise)


### Setup SSO/Auth for UI

#### Step 1: Set upperbounds for keys
Control the upperbound that users can use for `max_budget`, `budget_duration` or any `key/generate` param per key. 

```yaml
litellm_settings:
  upperbound_key_generate_params:
    max_budget: 100 # Optional[float], optional): upperbound of $100, for all /key/generate requests
    budget_duration: "10d" # Optional[str], optional): upperbound of 10 days for budget_duration values
    duration: "30d" # Optional[str], optional): upperbound of 30 days for all /key/generate requests
    max_parallel_requests: 1000 # (Optional[int], optional): Max number of requests that can be made in parallel. Defaults to None.
    tpm_limit: 1000 #(Optional[int], optional): Tpm limit. Defaults to None.
    rpm_limit: 1000 #(Optional[int], optional): Rpm limit. Defaults to None.

```

** Expected Behavior **

- Send a `/key/generate` request with `max_budget=200`
- Key will be created with `max_budget=100` since 100 is the upper bound

#### Step 2: Setup Oauth Client

:::tip

Looking for how to use Oauth 2.0 for /chat, /completions API requests to the proxy? [Follow this doc](oauth2)

:::

<Tabs>
<TabItem value="okta" label="Okta SSO">

1. Add Okta credentials to your .env

```bash
GENERIC_CLIENT_ID = "<your-okta-client-id>"
GENERIC_CLIENT_SECRET = "<your-okta-client-secret>" 
GENERIC_AUTHORIZATION_ENDPOINT = "<your-okta-domain>/authorize" # https://dev-2kqkcd6lx6kdkuzt.us.auth0.com/authorize
GENERIC_TOKEN_ENDPOINT = "<your-okta-domain>/token" # https://dev-2kqkcd6lx6kdkuzt.us.auth0.com/oauth/token
GENERIC_USERINFO_ENDPOINT = "<your-okta-domain>/userinfo" # https://dev-2kqkcd6lx6kdkuzt.us.auth0.com/userinfo
GENERIC_CLIENT_STATE = "random-string" # [OPTIONAL] REQUIRED BY OKTA, if not set random state value is generated
```

You can get your domain specific auth/token/userinfo endpoints at `<YOUR-OKTA-DOMAIN>/.well-known/openid-configuration`

2. Add proxy url as callback_url on Okta

On Okta, add the 'callback_url' as `<proxy_base_url>/sso/callback`


<Image img={require('../../img/okta_callback_url.png')} />

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
GENERIC_USER_PROVIDER_ATTRIBUTE = "provider"
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

#### Step 3. Set `PROXY_BASE_URL` in your .env

Set this in your .env (so the proxy can set the correct redirect url)
```shell
PROXY_BASE_URL=https://litellm-api.up.railway.app/
```

#### Step 4. Test flow
<Image img={require('../../img/litellm_ui_3.gif')} />

### Restrict Email Subdomains w/ SSO

If you're using SSO and want to only allow users with a specific subdomain - e.g. (@berri.ai email accounts) to access the UI, do this:

```bash
export ALLOWED_EMAIL_DOMAINS="berri.ai"
```

This will check if the user email we receive from SSO contains this domain, before allowing access.

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

### Disable `Default Team` on Admin UI

Use this if you want to hide the Default Team on the Admin UI

The following logic will apply
- If team assigned don't show `Default Team`
- If no team assigned then they should see `Default Team`

Set `default_team_disabled: true` on your litellm config.yaml

```yaml
general_settings:
  master_key: sk-1234
  default_team_disabled: true # OR you can set env var PROXY_DEFAULT_TEAM_DISABLED="true"
```

### Sign in with Username, Password when SSO is on

If you need to access the UI via username/password when SSO is on navigate to `/fallback/login`. This route will allow you to sign in with your username/password credentials.

### Restrict UI Access

You can restrict UI Access to just admins - includes you (proxy_admin) and people you give view only access to (proxy_admin_viewer) for seeing global spend.

**Step 1. Set 'admin_only' access**
```yaml
general_settings:
    ui_access_mode: "admin_only"
```

**Step 2. Invite view-only users**

<Image img={require('../../img/admin_ui_viewer.png')} />

### Custom Branding Admin UI

Use your companies custom branding on the LiteLLM Admin UI
We allow you to 
- Customize the UI Logo
- Customize the UI color scheme
<Image img={require('../../img/litellm_custom_ai.png')} />

#### Set Custom Logo
We allow you to pass a local image or a an http/https url of your image

Set `UI_LOGO_PATH` on your env. We recommend using a hosted image, it's a lot easier to set up and configure / debug

Exaple setting Hosted image
```shell
UI_LOGO_PATH="https://litellm-logo-aws-marketplace.s3.us-west-2.amazonaws.com/berriai-logo-github.png"
```

Exaple setting a local image (on your container)
```shell
UI_LOGO_PATH="ui_images/logo.jpg"
```
#### Set Custom Color Theme
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
- Deploy LiteLLM Proxy Server



## Disable Admin UI

Set `DISABLE_ADMIN_UI="True"` in your environment to disable the Admin UI. 

Useful, if your security team has additional restrictions on UI usage. 


**Expected Response**

<Image img={require('../../img/admin_ui_disabled.png')}/>