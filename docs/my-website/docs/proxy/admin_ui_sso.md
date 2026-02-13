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

#### Step 1: Create an OIDC Application in Okta

In your Okta Admin Console, create a new **OIDC Web Application**. See [Okta's guide on creating OIDC app integrations](https://help.okta.com/en-us/content/topics/apps/apps_app_integration_wizard_oidc.htm) for detailed instructions.

When configuring the application:
- **Sign-in redirect URI**: `https://<your-proxy-base-url>/sso/callback`
- **Sign-out redirect URI** (optional): `https://<your-proxy-base-url>`

<Image img={require('../../img/okta_redirect_uri.png')} />

After creating the app, copy your **Client ID** and **Client Secret** from the application's General tab:

<Image img={require('../../img/okta_client_credentials.png')} />

#### Step 2: Assign Users to the Application

Ensure users are assigned to the app in the **Assignments** tab. If Federation Broker Mode is enabled, you may need to disable it to assign users manually.

#### Step 3: Configure Authorization Server Access Policy

:::warning Important
This step is required. Without an Access Policy for your app, users will get a `no_matching_policy` error when attempting to log in.
:::

1. Go to **Security** → **API**

<Image img={require('../../img/okta_security_api.png')} />

2. Select the **default** authorization server (or your custom one)

<Image img={require('../../img/okta_authorization_server.png')} />

3. Click on **Access Policies** tab, create a new policy assigned to your LiteLLM app
4. Add a rule that allows the **Authorization Code** grant type

<Image img={require('../../img/okta_access_policies.png')} />

See [Okta's Access Policy documentation](https://help.okta.com/en-us/content/topics/security/api-access-management/access-policies.htm) for more details.

#### Step 4: Configure LiteLLM Environment Variables

```bash
GENERIC_CLIENT_ID="<your-client-id>"
GENERIC_CLIENT_SECRET="<your-client-secret>"
GENERIC_AUTHORIZATION_ENDPOINT="https://<your-okta-domain>/oauth2/default/v1/authorize"
GENERIC_TOKEN_ENDPOINT="https://<your-okta-domain>/oauth2/default/v1/token"
GENERIC_USERINFO_ENDPOINT="https://<your-okta-domain>/oauth2/default/v1/userinfo"
GENERIC_CLIENT_STATE="random-string"
PROXY_BASE_URL="https://<your-proxy-base-url>"
```

:::tip
You can find all OAuth endpoints at `https://<your-okta-domain>/.well-known/openid-configuration`
:::

#### Step 5: Test the SSO Flow

1. Start your LiteLLM proxy
2. Navigate to `https://<your-proxy-base-url>/ui`
3. Click the SSO login button
4. Authenticate with Okta and verify you're redirected back to LiteLLM

#### Troubleshooting

| Error | Cause | Solution |
|-------|-------|----------|
| `redirect_uri` error | Redirect URI not configured | Add `<proxy_base_url>/sso/callback` to Sign-in redirect URIs in Okta |
| `access_denied` | User not assigned to app | Assign the user in the Assignments tab |
| `no_matching_policy` | Missing Access Policy | Create an Access Policy in the Authorization Server (see Step 3) |

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
MICROSOFT_TENANT="5a39737"
```

**Optional: Custom Microsoft SSO Endpoints**

If you need to use custom Microsoft SSO endpoints (e.g., for a custom identity provider, sovereign cloud, or proxy), you can override the default endpoints:

```shell
MICROSOFT_AUTHORIZATION_ENDPOINT="https://your-custom-url.com/oauth2/v2.0/authorize"
MICROSOFT_TOKEN_ENDPOINT="https://your-custom-url.com/oauth2/v2.0/token"
MICROSOFT_USERINFO_ENDPOINT="https://your-custom-graph-api.com/v1.0/me"
```

If these are not set, the default Microsoft endpoints are used based on your tenant.

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

**Advanced: Custom User Attribute Mapping**

For certain Microsoft Entra ID configurations, you may need to override the default user attribute field names. This is useful when your organization uses custom claims or non-standard attribute names in the SSO response.

**Step 1: Debug SSO Response**

First, inspect the JWT fields returned by your Microsoft SSO provider using the [SSO Debug Route](#debugging-sso-jwt-fields).

1. Add `/sso/debug/callback` as a redirect URL in your Azure App Registration
2. Navigate to `https://<proxy_base_url>/sso/debug/login`
3. Complete the SSO flow to see the returned user attributes

**Step 2: Identify Field Attribute Names**

From the debug response, identify the field names used for email, display name, user ID, first name, and last name.

**Step 3: Set Environment Variables**

Override the default attribute names by setting these environment variables:

| Environment Variable | Description | Default Value |
|---------------------|-------------|---------------|
| `MICROSOFT_USER_EMAIL_ATTRIBUTE` | Field name for user email | `userPrincipalName` |
| `MICROSOFT_USER_DISPLAY_NAME_ATTRIBUTE` | Field name for display name | `displayName` |
| `MICROSOFT_USER_ID_ATTRIBUTE` | Field name for user ID | `id` |
| `MICROSOFT_USER_FIRST_NAME_ATTRIBUTE` | Field name for first name | `givenName` |
| `MICROSOFT_USER_LAST_NAME_ATTRIBUTE` | Field name for last name | `surname` |

**Step 4: Restart the Proxy**

After setting the environment variables, restart the proxy:

```bash
litellm --config /path/to/config.yaml
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
GENERIC_USER_EXTRA_ATTRIBUTES = "department,employee_id,manager" # comma-separated list of additional fields to extract from SSO response
GENERIC_CLIENT_STATE = "some-state" # if the provider needs a state parameter
GENERIC_INCLUDE_CLIENT_ID = "false" # some providers enforce that the client_id is not in the body
GENERIC_SCOPE = "openid profile email" # default scope openid is sometimes not enough to retrieve basic user info like first_name and last_name located in profile scope
```

**Assigning User Roles via SSO**

Use `GENERIC_USER_ROLE_ATTRIBUTE` to specify which attribute in the SSO token contains the user's role. The role value must be one of the following supported LiteLLM roles:

- `proxy_admin` - Admin over the platform
- `proxy_admin_viewer` - Can login, view all keys, view all spend (read-only)
- `internal_user` - Can login, view/create/delete their own keys, view their spend
- `internal_user_view_only` - Can login, view their own keys, view their own spend

Nested attribute paths are supported (e.g., `claims.role` or `attributes.litellm_role`).

**Capturing Additional SSO Fields**

Use `GENERIC_USER_EXTRA_ATTRIBUTES` to extract additional fields from the SSO provider response beyond the standard user attributes (id, email, name, etc.). This is useful when you need to access custom organization-specific data (e.g., department, employee ID, groups) in your [custom SSO handler](./custom_sso.md).

```shell
# Comma-separated list of field names to extract
GENERIC_USER_EXTRA_ATTRIBUTES="department,employee_id,manager,groups"
```

**Accessing Extra Fields in Custom SSO Handler:**

```python
from litellm.proxy.management_endpoints.types import CustomOpenID

async def custom_sso_handler(userIDPInfo: CustomOpenID):
    # Access the extra fields
    extra_fields = getattr(userIDPInfo, 'extra_fields', None) or {}
    
    user_department = extra_fields.get("department")
    employee_id = extra_fields.get("employee_id")
    user_groups = extra_fields.get("groups", [])
    
    # Use these fields for custom logic (e.g., team assignment, access control)
    # ...
```

**Nested Field Paths:**

Dot notation is supported for nested fields:

```shell
GENERIC_USER_EXTRA_ATTRIBUTES="org_info.department,org_info.cost_center,metadata.employee_type"
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


## Advanced

### Manage User Roles via Azure App Roles

Centralize role management by defining user permissions in Azure Entra ID. LiteLLM will automatically assign roles based on your Azure configuration when users sign in—no need to manually manage roles in LiteLLM.

#### Step 1: Create App Roles on Azure App Registration

1. Navigate to your App Registration on https://portal.azure.com/
2. Go to **App roles** > **Create app role**
3. Configure the app role using one of the [supported LiteLLM roles](./access_control.md#global-proxy-roles):
   - **Display name**: Admin Viewer (or your preferred display name)
   - **Value**: `proxy_admin_viewer` (must match one of the LiteLLM role values exactly)
4. Click **Apply** to save the role
5. Repeat for each LiteLLM role you want to use


**Supported LiteLLM role values** (see [full role documentation](./access_control.md#global-proxy-roles)):
- `proxy_admin` - Full admin access
- `proxy_admin_viewer` - Read-only admin access
- `internal_user` - Can create/view/delete own keys
- `internal_user_viewer` - Can view own keys (read-only)

<Image img={require('../../img/app_roles.png')} style={{ width: '900px', height: 'auto' }} />

---

#### Step 2: Assign Users to App Roles

1. Navigate to **Enterprise Applications** on https://portal.azure.com/
2. Select your LiteLLM application
3. Go to **Users and groups** > **Add user/group**
4. Select the user
5. Under **Select a role**, choose the app role you created (e.g., `proxy_admin_viewer`)
6. Click **Assign** to save

<Image img={require('../../img/app_role2.png')} style={{ width: '900px', height: 'auto' }} />

---

#### Step 3: Sign in and verify

1. Sign in to the LiteLLM UI via SSO
2. LiteLLM will automatically extract the app role from the JWT token
3. The user will be assigned the corresponding role (you can verify this in the UI by checking the user profile dropdown)

<Image img={require('../../img/app_role3.png')} style={{ width: '900px', height: 'auto' }} />

**Note:** The role from Entra ID will take precedence over any existing role in the LiteLLM database. This ensures your SSO provider is the authoritative source for user roles.

