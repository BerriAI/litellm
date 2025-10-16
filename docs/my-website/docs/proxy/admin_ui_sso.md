import Image from '@theme/IdealImage';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# ✨ SSO for Admin UI

:::info
From v1.76.0, SSO is now Free for up to 5 users.
:::

:::info

✨ SSO is on LiteLLM Enterprise

[Enterprise Pricing](https://www.litellm.ai/#pricing)

[Get free 7-day trial key](https://www.litellm.ai/enterprise#trial)

:::

### Usage (Google, Microsoft, Okta, etc.)

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
GENERIC_SSO_HEADERS = "Content-Type=application/json, X-Custom-Header=custom-value" # [OPTIONAL] Comma-separated list of additional headers to add to the request - e.g. Content-Type=application/json, etc.
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

**Using App Roles for User Permissions**

You can assign user roles directly from Entra ID using App Roles. LiteLLM will automatically read the app roles from the JWT token and assign the corresponding role to the user.

Supported roles:
- `proxy_admin` - Admin over the platform
- `proxy_admin_viewer` - Can login, view all keys, view all spend (read-only)
- `internal_user` - Normal user. Can login, view spend and depending on team-member permissions - view/create/delete their own keys.


To set up app roles:
1. Navigate to your App Registration on https://portal.azure.com/
2. Go to "App roles" and create a new app role
3. Use one of the supported role names above (e.g., `proxy_admin`)
4. Assign users to these roles in your Enterprise Application
5. When users sign in via SSO, LiteLLM will automatically assign them the corresponding role

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

### Default Login, Logout URLs

Some SSO providers require a specific redirect url for login and logout. You can input the following values.

- Login: `<your-proxy-base-url>/sso/key/generate`
- Logout: `<your-proxy-base-url>`

Here's the env var to set the logout url on the proxy
```bash
PROXY_LOGOUT_URL="https://www.google.com"
```

#### Step 3. Set `PROXY_BASE_URL` in your .env

Set this in your .env (so the proxy can set the correct redirect url)
```shell
PROXY_BASE_URL=https://litellm-api.up.railway.app
```

#### Step 4. Test flow
<Image img={require('../../img/litellm_ui_3.gif')} />

### Restrict Email Subdomains w/ SSO

If you're using SSO and want to only allow users with a specific subdomain - e.g. (@berri.ai email accounts) to access the UI, do this:

```bash
export ALLOWED_EMAIL_DOMAINS="berri.ai"
```

This will check if the user email we receive from SSO contains this domain, before allowing access.

### Set Proxy Admin

Set a Proxy Admin when SSO is enabled. Once SSO is enabled, the `user_id` for users is retrieved from the SSO provider. In order to set a Proxy Admin, you need to copy the `user_id` from the UI and set it in your `.env` as `PROXY_ADMIN_ID`.

#### Step 1: Copy your ID from the UI 

<Image img={require('../../img/litellm_ui_copy_id.png')} />

#### Step 2: Set it in your .env as the PROXY_ADMIN_ID 

```env
export PROXY_ADMIN_ID="116544810872468347480"
```

This will update the user role in the `LiteLLM_UserTable` to `proxy_admin`. 

If you plan to change this ID, please update the user role via API `/user/update` or UI (Internal Users page). 

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

### Use Username, Password when SSO is on

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

Example setting Hosted image
```shell
UI_LOGO_PATH="https://litellm-logo-aws-marketplace.s3.us-west-2.amazonaws.com/berriai-logo-github.png"
```

Example setting a local image (on your container)
```shell
UI_LOGO_PATH="ui_images/logo.jpg"
```

#### Or set your logo directly from Admin UI:
<div style={{ display: 'flex', gap: '12px', alignItems: 'center' }}>
  <Image img={require('../../img/admin_settings_ui_theme.png')} />
  <Image img={require('../../img/admin_settings_ui_theme_logo.png')} />
</div>

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

## Troubleshooting

### "The 'redirect_uri' parameter must be a Login redirect URI in the client app settings" Error

This error commonly occurs with Okta and other SSO providers when the redirect URI configuration is incorrect.

#### Issue
```
Your request resulted in an error. The 'redirect_uri' parameter must be a Login redirect URI in the client app settings
```

#### Solution

**1. Ensure you have set PROXY_BASE_URL in your .env and it includes protocol**

Make sure your `PROXY_BASE_URL` includes the complete URL with protocol (`http://` or `https://`):

```bash
# ✅ Correct - includes https://
PROXY_BASE_URL=https://litellm.platform.com

# ✅ Correct - includes http://
PROXY_BASE_URL=http://litellm.platform.com

# ❌ Incorrect - missing protocol
PROXY_BASE_URL=litellm.platform.com
```

**2. For Okta specifically, ensure GENERIC_CLIENT_STATE is set**

Okta requires the `GENERIC_CLIENT_STATE` parameter:

```bash
GENERIC_CLIENT_STATE="random-string" # Required for Okta
```

### Okta PKCE

If your Okta application is configured to require PKCE (Proof Key for Code Exchange), enable it by setting:

```bash
GENERIC_CLIENT_USE_PKCE="true"
```

This is required when your Okta app settings enforce PKCE for enhanced security. LiteLLM will automatically handle PKCE parameter generation and verification during the OAuth flow.

### Common Configuration Issues

#### Missing Protocol in Base URL
```bash
# This will cause redirect_uri errors
PROXY_BASE_URL=mydomain.com

# Fix: Add the protocol
PROXY_BASE_URL=https://mydomain.com
```

### Fallback Login

If you need to access the UI via username/password when SSO is on navigate to `/fallback/login`. This route will allow you to sign in with your username/password credentials.

<Image img={require('../../img/fallback_login.png')} />


### Debugging SSO JWT fields 

If you need to inspect the JWT fields received from your SSO provider by LiteLLM, follow these instructions. This guide walks you through setting up a debug callback to view the JWT data during the SSO process.


<Image img={require('../../img/debug_sso.png')}  style={{ width: '500px', height: 'auto' }} />
<br />

1. Add `/sso/debug/callback` as a redirect URL in your SSO provider 

  In your SSO provider's settings, add the following URL as a new redirect (callback) URL:

  ```bash showLineNumbers title="Redirect URL"
  http://<proxy_base_url>/sso/debug/callback
  ```


2. Navigate to the debug login page on your browser 

    Navigate to the following URL on your browser:

    ```bash showLineNumbers title="URL to navigate to"
    https://<proxy_base_url>/sso/debug/login
    ```

    This will initiate the standard SSO flow. You will be redirected to your SSO provider's login screen, and after successful authentication, you will be redirected back to LiteLLM's debug callback route.


3. View the JWT fields 

Once redirected, you should see a page called "SSO Debug Information". This page displays the JWT fields received from your SSO provider (as shown in the image above)

