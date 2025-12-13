import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# OIDC - JWT-based Auth 

Use JWT's to auth admins / users / projects into the proxy.

:::info

✨ JWT-based Auth  is on LiteLLM Enterprise

[Enterprise Pricing](https://www.litellm.ai/#pricing)

[Contact us here to get a free trial](https://calendly.com/d/4mp-gd3-k5k/litellm-1-1-onboarding-chat)

:::


## Usage

### Step 1. Setup Proxy

- `JWT_PUBLIC_KEY_URL`: This is the public keys endpoint of your OpenID provider. Typically it's `{openid-provider-base-url}/.well-known/openid-configuration/jwks`. For Keycloak it's `{keycloak_base_url}/realms/{your-realm}/protocol/openid-connect/certs`.
- `JWT_AUDIENCE`: This is the audience used for decoding the JWT. If not set, the decode step will not verify the audience. 

```bash
export JWT_PUBLIC_KEY_URL="" # "https://demo.duendesoftware.com/.well-known/openid-configuration/jwks"
```

- `enable_jwt_auth` in your config. This will tell the proxy to check if a token is a jwt token.

```yaml
general_settings:
  master_key: sk-1234
  enable_jwt_auth: True

model_list:
- model_name: azure-gpt-3.5 
  litellm_params:
      model: azure/<your-deployment-name>
      api_base: os.environ/AZURE_API_BASE
      api_key: os.environ/AZURE_API_KEY
      api_version: "2023-07-01-preview"
```

### Step 2. Create JWT with scopes 

<Tabs>
<TabItem value="admin" label="admin">

Create a client scope called `litellm_proxy_admin` in your OpenID provider (e.g. Keycloak).

Grant your user, `litellm_proxy_admin` scope when generating a JWT. 

```bash
curl --location ' 'https://demo.duendesoftware.com/connect/token'' \
--header 'Content-Type: application/x-www-form-urlencoded' \
--data-urlencode 'client_id={CLIENT_ID}' \
--data-urlencode 'client_secret={CLIENT_SECRET}' \
--data-urlencode 'username=test-{USERNAME}' \
--data-urlencode 'password={USER_PASSWORD}' \
--data-urlencode 'grant_type=password' \
--data-urlencode 'scope=litellm_proxy_admin' # 👈 grant this scope
```
</TabItem>
<TabItem value="project" label="project">

Create a JWT for your project on your OpenID provider (e.g. Keycloak).

```bash
curl --location ' 'https://demo.duendesoftware.com/connect/token'' \
--header 'Content-Type: application/x-www-form-urlencoded' \
--data-urlencode 'client_id={CLIENT_ID}' \ # 👈 project id
--data-urlencode 'client_secret={CLIENT_SECRET}' \
--data-urlencode 'grant_type=client_credential' \
```

</TabItem>
</Tabs>

### Step 3. Test your JWT 

<Tabs>
<TabItem value="key" label="/key/generate">

```bash
curl --location '{proxy_base_url}/key/generate' \
--header 'Authorization: Bearer eyJhbGciOiJSUzI1NiI...' \
--header 'Content-Type: application/json' \
--data '{}'
```
</TabItem>
<TabItem value="llm_call" label="/chat/completions">

```bash
curl --location 'http://0.0.0.0:4000/v1/chat/completions' \
--header 'Content-Type: application/json' \
--header 'Authorization: Bearer eyJhbGciOiJSUzI1...' \
--data '{"model": "azure-gpt-3.5", "messages": [ { "role": "user", "content": "What's the weather like in Boston today?" } ]}'
```

</TabItem>
</Tabs>

## Advanced

### Multiple OIDC providers

Use this if you want LiteLLM to validate your JWT against multiple OIDC providers (e.g. Google Cloud, GitHub Auth)

Set `JWT_PUBLIC_KEY_URL` in your environment to a comma-separated list of URLs for your OIDC providers.

```bash
export JWT_PUBLIC_KEY_URL="https://demo.duendesoftware.com/.well-known/openid-configuration/jwks,https://accounts.google.com/.well-known/openid-configuration/jwks"
```

### Set Accepted JWT Scope Names 

Change the string in JWT 'scopes', that litellm evaluates to see if a user has admin access.

```yaml
general_settings:
  master_key: sk-1234
  enable_jwt_auth: True
  litellm_jwtauth:
    admin_jwt_scope: "litellm-proxy-admin"
```

### Tracking End-Users / Internal Users / Team / Org

Set the field in the jwt token, which corresponds to a litellm user / team / org.

**Note:** All JWT fields support dot notation to access nested claims (e.g., `"user.sub"`, `"resource_access.client.roles"`).

```yaml
general_settings:
  master_key: sk-1234
  enable_jwt_auth: True
  litellm_jwtauth:
    admin_jwt_scope: "litellm-proxy-admin"
    team_id_jwt_field: "client_id" # 👈 CAN BE ANY FIELD (supports dot notation for nested claims)
    user_id_jwt_field: "sub" # 👈 CAN BE ANY FIELD (supports dot notation for nested claims)
    org_id_jwt_field: "org_id" # 👈 CAN BE ANY FIELD (supports dot notation for nested claims)
    end_user_id_jwt_field: "customer_id" # 👈 CAN BE ANY FIELD (supports dot notation for nested claims)
```

Expected JWT (flat structure): 

```json
{
  "client_id": "my-unique-team",
  "sub": "my-unique-user",
  "org_id": "my-unique-org"
}
```

**Or with nested structure using dot notation:**

```json
{
  "user": {
    "sub": "my-unique-user",
    "email": "user@example.com"
  },
  "tenant": {
    "team_id": "my-unique-team"
  },
  "organization": {
    "id": "my-unique-org"
  }
}
```

**Configuration for nested example:**

```yaml
litellm_jwtauth:
  user_id_jwt_field: "user.sub"
  user_email_jwt_field: "user.email"
  team_id_jwt_field: "tenant.team_id"
  org_id_jwt_field: "organization.id"
```

Now litellm will automatically update the spend for the user/team/org in the db for each call. 

### JWT Scopes

Here's what scopes on JWT-Auth tokens look like

**Can be a list**
```
scope: ["litellm-proxy-admin",...]
```

**Can be a space-separated string**
```
scope: "litellm-proxy-admin ..."
```

### Control model access with Teams


1. Specify the JWT field that contains the team ids, that the user belongs to. 

```yaml
general_settings:
  enable_jwt_auth: True
  litellm_jwtauth:
    user_id_jwt_field: "sub"
    team_ids_jwt_field: "groups" 
    user_id_upsert: true # add user_id to the db if they don't exist
    enforce_team_based_model_access: true # don't allow users to access models unless the team has access
```

This is assuming your token looks like this:
```
{
  ...,
  "sub": "my-unique-user",
  "groups": ["team_id_1", "team_id_2"]
}
```

2. Create the teams on LiteLLM 

```bash
curl -X POST '<PROXY_BASE_URL>/team/new' \
-H 'Authorization: Bearer <PROXY_MASTER_KEY>' \
-H 'Content-Type: application/json' \
-D '{
    "team_alias": "team_1",
    "team_id": "team_id_1" # 👈 MUST BE THE SAME AS THE SSO GROUP ID
}'
```

3. Test the flow

SSO for UI: [**See Walkthrough**](https://www.loom.com/share/8959be458edf41fd85937452c29a33f3?sid=7ebd6d37-569a-4023-866e-e0cde67cb23e)

OIDC Auth for API: [**See Walkthrough**](https://www.loom.com/share/00fe2deab59a426183a46b1e2b522200?sid=4ed6d497-ead6-47f9-80c0-ca1c4b6b4814)


### Flow

- Validate if user id is in the DB (LiteLLM_UserTable)
- Validate if any of the groups are in the DB (LiteLLM_TeamTable)
- Validate if any group has model access
- If all checks pass, allow the request

### Select Team via Request Header

When a JWT token contains multiple teams (via `team_ids_jwt_field`), you can explicitly select which team to use for a request by passing the `x-litellm-team-id` header.

```bash
curl -X POST 'http://0.0.0.0:4000/v1/chat/completions' \
-H 'Content-Type: application/json' \
-H 'Authorization: Bearer <your-jwt-token>' \
-H 'x-litellm-team-id: team_id_2' \
-d '{
  "model": "gpt-4",
  "messages": [{"role": "user", "content": "Hello"}]
}'
```

**Validation:**
- The team ID in the header must exist in the JWT's `team_ids_jwt_field` list or match `team_id_jwt_field`
- If an invalid team is specified, a 403 error is returned
- If no header is provided, LiteLLM auto-selects the first team with access to the requested model


### Custom JWT Validate

Validate a JWT Token using custom logic, if you need an extra way to verify if tokens are valid for LiteLLM Proxy.

#### 1. Setup custom validate function

```python
from typing import Literal

def my_custom_validate(token: str) -> Literal[True]:
  """
  Only allow tokens with tenant-id == "my-unique-tenant", and claims == ["proxy-admin"]
  """
  allowed_tenants = ["my-unique-tenant"]
  allowed_claims = ["proxy-admin"]

  if token["tenant_id"] not in allowed_tenants:
    raise Exception("Invalid JWT token")
  if token["claims"] not in allowed_claims:
    raise Exception("Invalid JWT token")
  return True
```

#### 2. Setup config.yaml

```yaml
general_settings:
  master_key: sk-1234
  enable_jwt_auth: True
  litellm_jwtauth:
    user_id_jwt_field: "sub"
    team_id_jwt_field: "tenant_id"
    user_id_upsert: True
    custom_validate: custom_validate.my_custom_validate # 👈 custom validate function
```

#### 3. Test the flow

**Expected JWT**

```
{
  "sub": "my-unique-user",
  "tenant_id": "INVALID_TENANT",
  "claims": ["proxy-admin"]
}
```

**Expected Response**

```
{
  "error": "Invalid JWT token"
}
```



### Allowed Routes 

Configure which routes a JWT can access via the config.

By default: 

- Admins: can access only management routes (`/team/*`, `/key/*`, `/user/*`)
- Teams: can access only openai routes (`/chat/completions`, etc.)+ info routes (`/*/info`)

[**See Code**](https://github.com/BerriAI/litellm/blob/b204f0c01c703317d812a1553363ab0cb989d5b6/litellm/proxy/_types.py#L95)

**Admin Routes**
```yaml
general_settings:
  master_key: sk-1234
  enable_jwt_auth: True
  litellm_jwtauth:
    admin_jwt_scope: "litellm-proxy-admin"
    admin_allowed_routes: ["/v1/embeddings"]
```

**Team Routes**
```yaml
general_settings:
  master_key: sk-1234
  enable_jwt_auth: True
  litellm_jwtauth:
    ...
    team_id_jwt_field: "litellm-team" # 👈 Set field in the JWT token that stores the team ID
    team_allowed_routes: ["/v1/chat/completions"] # 👈 Set accepted routes
```

### Allowing other provider routes for Teams

To enable team JWT tokens to access Anthropic-style endpoints such as `/v1/messages`, update `team_allowed_routes` in your `litellm_jwtauth` configuration. `team_allowed_routes` supports the following values:

- Named route groups from `LiteLLMRoutes` (e.g., `openai_routes`, `anthropic_routes`, `info_routes`, `mapped_pass_through_routes`).

Below is a quick reference for the route groups you can use and example representative routes from each group. If you need the exhaustive list, see the `LiteLLMRoutes` enum in `litellm/proxy/_types.py` for the authoritative list.

| Route Group | What it contains | Representative routes |
|-------------|------------------|-----------------------|
| `openai_routes` | OpenAI-compatible REST endpoints (chat, completion, embeddings, images, responses, models, etc.) | `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, `/v1/images/generations`, `/v1/models` |
| `anthropic_routes` | Anthropic-style endpoints (`/v1/messages` and related) | `/v1/messages`, `/v1/messages/count_tokens`, `/v1/skills` |
| `mapped_pass_through_routes` | Provider-specific pass-through route prefixes (e.g., Anthropic when proxied via `/anthropic`). Use with `mapped_pass_through_routes` for provider wildcard mapping | `/anthropic/*`, `/vertex-ai/*`, `/bedrock/*` |
| `passthrough_routes_wildcard` | Wildcard mapping for providers (e.g., `/anthropic/*`) - precomputed wildcard list used by the proxy | `/anthropic/*`, `/vllm/*` |
| `google_routes` | Google-specific (e.g., Vertex / Batching endpoints) | `/v1beta/models/{model_name}:generateContent` |
| `mcp_routes` | Internal MCP management endpoints | `/mcp/tools`, `/mcp/tools/call` |
| `info_routes` | Read-only & info endpoints used by the UI | `/key/info`, `/team/info`, `/v1/models` |
| `management_routes` | Admin-only management endpoints (create/update/delete user/team/model) | `/team/new`, `/key/generate`, `/model/new` |
| `spend_tracking_routes` | Budget/spend related endpoints | `/spend/logs`, `/spend/keys` |
| `public_routes` | Public and unauthenticated endpoints | `/`, `/routes`, `/.well-known/litellm-ui-config` |

Note: `llm_api_routes` is the union of OpenAI, Anthropic, Google, pass-through and other LLM routes (`openai_routes + anthropic_routes + google_routes + mapped_pass_through_routes + passthrough_routes_wildcard + apply_guardrail_routes + mcp_routes + litellm_native_routes`).

Defaults (what the proxy uses if you don't override them in `litellm_jwtauth`):

- `admin_jwt_scope`: `litellm_proxy_admin`
- `admin_allowed_routes` (default): `management_routes`, `spend_tracking_routes`, `global_spend_tracking_routes`, `info_routes` 
- `team_allowed_routes` (default): `openai_routes`, `info_routes` 
- `public_allowed_routes` (default): `public_routes`


Example: Allow team JWTs to call Anthropic `/v1/messages` (either by route group or by explicit route string):

```yaml
general_settings:
  enable_jwt_auth: True
  litellm_jwtauth:
    team_ids_jwt_field: "team_ids"
    team_allowed_routes: ["openai_routes", "info_routes", "anthropic_routes"]
```

Or selectively allow the exact Anthropic message endpoint only:

```yaml
general_settings:
  enable_jwt_auth: True
  litellm_jwtauth:
    team_ids_jwt_field: "team_ids"
    team_allowed_routes: ["/v1/messages", "info_routes"]
```


### Caching Public Keys 

Control how long public keys are cached for (in seconds).

```yaml
general_settings:
  master_key: sk-1234
  enable_jwt_auth: True
  litellm_jwtauth:
    admin_jwt_scope: "litellm-proxy-admin"
    admin_allowed_routes: ["/v1/embeddings"]
    public_key_ttl: 600 # 👈 KEY CHANGE
```

### Custom JWT Field 

Set a custom field in which the team_id exists. By default, the 'client_id' field is checked. 

```yaml
general_settings:
  master_key: sk-1234
  enable_jwt_auth: True
  litellm_jwtauth:
    team_id_jwt_field: "client_id" # 👈 KEY CHANGE
```

### Block Teams 

To block all requests for a certain team id, use `/team/block`

**Block Team**

```bash
curl --location 'http://0.0.0.0:4000/team/block' \
--header 'Authorization: Bearer <admin-token>' \
--header 'Content-Type: application/json' \
--data '{
    "team_id": "litellm-test-client-id-new" # 👈 set team id
}'
```

**Unblock Team**

```bash
curl --location 'http://0.0.0.0:4000/team/unblock' \
--header 'Authorization: Bearer <admin-token>' \
--header 'Content-Type: application/json' \
--data '{
    "team_id": "litellm-test-client-id-new" # 👈 set team id
}'
```


### Upsert Users + Allowed Email Domains 

Allow users who belong to a specific email domain, automatic access to the proxy.

**Note:** `user_allowed_email_domain` is optional. If not specified, all users will be allowed regardless of their email domain.
 
```yaml
general_settings:
  master_key: sk-1234
  enable_jwt_auth: True
  litellm_jwtauth:
    user_email_jwt_field: "email" # 👈 checks 'email' field in jwt payload
    user_allowed_email_domain: "my-co.com" # 👈 OPTIONAL - allows user@my-co.com to call proxy
    user_id_upsert: true # 👈 upserts the user to db, if valid email but not in db
```

## OIDC UserInfo Endpoint

Use this when your JWT/access token doesn't contain user-identifying information. LiteLLM will call your identity provider's UserInfo endpoint to fetch user details.

### When to Use

- Your JWT is opaque (not self-contained) or lacks user claims
- You need to fetch fresh user information from your identity provider
- Your access tokens don't include email, roles, or other identifying data

### Configuration

```yaml title="config.yaml" showLineNumbers
general_settings:
  enable_jwt_auth: True
  litellm_jwtauth:
    # Enable OIDC UserInfo endpoint
    oidc_userinfo_enabled: true
    oidc_userinfo_endpoint: "https://your-idp.com/oauth2/userinfo"
    oidc_userinfo_cache_ttl: 300  # Cache for 5 minutes (default: 300)
    
    # Map fields from UserInfo response
    user_id_jwt_field: "sub"
    user_email_jwt_field: "email"
    user_roles_jwt_field: "roles"
```

### Flow Diagram

```mermaid
sequenceDiagram
    participant Client
    participant LiteLLM
    participant IdP as Identity Provider

    Client->>LiteLLM: Request with Bearer token
    Note over LiteLLM: Check cache for UserInfo
    
    LiteLLM->>IdP: GET /userinfo (if not cached)<br/>Authorization: Bearer {token}
    IdP-->>LiteLLM: User data (sub, email, roles)
    
    Note over LiteLLM: Cache response (TTL: 5min)<br/>Extract user_id, email, roles<br/>Perform RBAC checks
    
    LiteLLM-->>Client: Authorized/Denied
```

### Example: Azure AD

```yaml title="config.yaml" showLineNumbers
litellm_jwtauth:
  oidc_userinfo_enabled: true
  oidc_userinfo_endpoint: "https://graph.microsoft.com/oidc/userinfo"
  user_id_jwt_field: "sub"
  user_email_jwt_field: "email"
```

### Example: Keycloak

```yaml title="config.yaml" showLineNumbers
litellm_jwtauth:
  oidc_userinfo_enabled: true
  oidc_userinfo_endpoint: "https://keycloak.example.com/realms/your-realm/protocol/openid-connect/userinfo"
  user_id_jwt_field: "sub"
  user_roles_jwt_field: "resource_access.your-client.roles"
```

## [BETA] Control Access with OIDC Roles

Allow JWT tokens with supported roles to access the proxy.

Let users and teams access the proxy, without needing to add them to the DB.


Very important, set `enforce_rbac: true` to ensure that the RBAC system is enabled.

**Note:** This is in beta and might change unexpectedly.

```yaml
general_settings:
  enable_jwt_auth: True
  litellm_jwtauth:
    object_id_jwt_field: "oid" # can be either user / team, inferred from the role mapping
    roles_jwt_field: "roles"
    role_mappings:
      - role: litellm.api.consumer
        internal_role: "team"
    enforce_rbac: true # 👈 VERY IMPORTANT

  role_permissions: # default model + endpoint permissions for a role. 
    - role: team
      models: ["anthropic-claude"]
      routes: ["/v1/chat/completions"]

environment_variables:
  JWT_AUDIENCE: "api://LiteLLM_Proxy" # ensures audience is validated
```

- `object_id_jwt_field`: The field in the JWT token that contains the object id. This id can be either a user id or a team id. Use this instead of `user_id_jwt_field` and `team_id_jwt_field`. If the same field could be both. **Supports dot notation** for nested claims (e.g., `"profile.object_id"`).

- `roles_jwt_field`: The field in the JWT token that contains the roles. This field is a list of roles that the user has. **Supports dot notation** for nested fields - e.g., `resource_access.litellm-test-client-id.roles`.

**Additional JWT Field Configuration Options:**

- `team_ids_jwt_field`: Field containing team IDs (as a list). **Supports dot notation** (e.g., `"groups"`, `"teams.ids"`).
- `user_email_jwt_field`: Field containing user email. **Supports dot notation** (e.g., `"email"`, `"user.email"`).
- `end_user_id_jwt_field`: Field containing end-user ID for cost tracking. **Supports dot notation** (e.g., `"customer_id"`, `"customer.id"`).

- `role_mappings`: A list of role mappings. Map the received role in the JWT token to an internal role on LiteLLM.

- `JWT_AUDIENCE`: The audience of the JWT token. This is used to validate the audience of the JWT token. Set via an environment variable.

### Example Token 

```bash
{
  "aud": "api://LiteLLM_Proxy",
  "oid": "eec236bd-0135-4b28-9354-8fc4032d543e",
  "roles": ["litellm.api.consumer"] 
}
```

### Role Mapping Spec 

- `role`: The expected role in the JWT token. 
- `internal_role`: The internal role on LiteLLM that will be used to control access. 

Supported internal roles:
- `team`: Team object will be used for RBAC spend tracking. Use this for tracking spend for a 'use case'. 
- `internal_user`: User object will be used for RBAC spend tracking. Use this for tracking spend for an 'individual user'.
- `proxy_admin`: Proxy admin will be used for RBAC spend tracking. Use this for granting admin access to a token.

### [Architecture Diagram (Control Model Access)](./jwt_auth_arch)

## [BETA] Control Model Access with Scopes

Control which models a JWT can access. Set `enforce_scope_based_access: true` to enforce scope-based access control.

### 1. Setup config.yaml with scope mappings.


```yaml
model_list:
  - model_name: anthropic-claude
    litellm_params:
      model: anthropic/claude-3-5-sonnet
      api_key: os.environ/ANTHROPIC_API_KEY
  - model_name: gpt-3.5-turbo-testing
    litellm_params:
      model: gpt-3.5-turbo
      api_key: os.environ/OPENAI_API_KEY

general_settings:
  enable_jwt_auth: True
  litellm_jwtauth:
    team_id_jwt_field: "client_id" # 👈 set the field in the JWT token that contains the team id
    team_id_upsert: true # 👈 upsert the team to db, if team id is not found in db
    scope_mappings:
      - scope: litellm.api.consumer
        models: ["anthropic-claude"]
      - scope: litellm.api.gpt_3_5_turbo
        models: ["gpt-3.5-turbo-testing"]
    enforce_scope_based_access: true # 👈 enforce scope-based access control
    enforce_rbac: true # 👈 enforces only a Team/User/ProxyAdmin can access the proxy.
```

#### Scope Mapping Spec 

- `scope`: The scope to be used for the JWT token.
- `models`: The models that the JWT token can access. Value is the `model_name` in `model_list`. Note: Wildcard routes are not currently supported.

### 2. Create a JWT with the correct scopes.

Expected Token:

```bash
{
  "scope": ["litellm.api.consumer", "litellm.api.gpt_3_5_turbo"] # can be a list or a space-separated string
}
```

### 3. Test the flow.

```bash
curl -L -X POST 'http://0.0.0.0:4000/v1/chat/completions' \
-H 'Content-Type: application/json' \
-H 'Authorization: Bearer eyJhbGci...' \
-d '{
  "model": "gpt-3.5-turbo-testing",
  "messages": [
    {
      "role": "user",
      "content": "Hey, how'\''s it going 1234?"
    }
  ]
}'
```

## [BETA] Sync User Roles and Teams with IDP

Automatically sync user roles and team memberships from your Identity Provider (IDP) to LiteLLM's database. This ensures that user permissions and team memberships in LiteLLM stay in sync with your IDP.

**Note:** This is in beta and might change unexpectedly.

### Use Cases

- **Role Synchronization**: Automatically update user roles in LiteLLM when they change in your IDP
- **Team Membership Sync**: Keep team memberships in sync between your IDP and LiteLLM
- **Centralized Access Management**: Manage all user permissions through your IDP while maintaining LiteLLM functionality

### Setup

#### 1. Configure JWT Role Mapping

Map roles from your JWT token to LiteLLM user roles:

```yaml
general_settings:
  enable_jwt_auth: True
  litellm_jwtauth:
    user_id_jwt_field: "sub"
    team_ids_jwt_field: "groups"
    roles_jwt_field: "roles"
    user_id_upsert: true
    sync_user_role_and_teams: true # 👈 Enable sync functionality
    jwt_litellm_role_map: # 👈 Map JWT roles to LiteLLM roles
      - jwt_role: "ADMIN"
        litellm_role: "proxy_admin"
      - jwt_role: "USER"
        litellm_role: "internal_user"
      - jwt_role: "VIEWER"
        litellm_role: "internal_user"
```

#### 2. JWT Role Mapping Spec

- `jwt_role`: The role name as it appears in your JWT token. Supports wildcard patterns using `fnmatch` (e.g., `"ADMIN_*"` matches `"ADMIN_READ"`, `"ADMIN_WRITE"`, etc.)
- `litellm_role`: The corresponding LiteLLM user role

**Supported LiteLLM Roles:**
- `proxy_admin`: Full administrative access
- `internal_user`: Standard user access
- `internal_user_view_only`: Read-only access

#### 3. Example JWT Token

```json
{
  "sub": "user-123",
  "roles": ["ADMIN"],
  "groups": ["team-alpha", "team-beta"],
  "iat": 1234567890,
  "exp": 1234567890
}
```

### How It Works

When a user makes a request with a JWT token:

1. **Role Sync**: 
   - LiteLLM checks if the user's role in the JWT matches their role in the database
   - If different, the user's role is updated in LiteLLM's database
   - Uses the `jwt_litellm_role_map` to convert JWT roles to LiteLLM roles

2. **Team Membership Sync**:
   - Compares team memberships from the JWT token with the user's current teams in LiteLLM
   - Adds the user to new teams found in the JWT
   - Removes the user from teams not present in the JWT

3. **Database Updates**:
   - Updates happen automatically during the authentication process
   - No manual intervention required

### Configuration Options

```yaml
general_settings:
  enable_jwt_auth: True
  litellm_jwtauth:
    # Required fields
    user_id_jwt_field: "sub"
    team_ids_jwt_field: "groups"
    roles_jwt_field: "roles"
    
    # Sync configuration
    sync_user_role_and_teams: true
    user_id_upsert: true
    
    # Role mapping
    jwt_litellm_role_map:
      - jwt_role: "AI_ADMIN_*"  # Wildcard pattern
        litellm_role: "proxy_admin"
      - jwt_role: "AI_USER"
        litellm_role: "internal_user"
```

### Important Notes

- **Performance**: Sync operations happen during authentication, which may add slight latency
- **Database Access**: Requires database access for user and team updates
- **Team Creation**: Teams mentioned in JWT tokens must exist in LiteLLM before sync can assign users to them
- **Wildcard Support**: JWT role patterns support wildcard matching using `fnmatch`

### Testing the Sync Feature

1. **Create a test user with initial role**:

```bash
curl -X POST 'http://0.0.0.0:4000/user/new' \
-H 'Authorization: Bearer <PROXY_MASTER_KEY>' \
-H 'Content-Type: application/json' \
-d '{
    "user_id": "user-123",
    "user_role": "internal_user"
}'
```

2. **Make a request with JWT containing different role**:

```bash
curl -X POST 'http://0.0.0.0:4000/v1/chat/completions' \
-H 'Content-Type: application/json' \
-H 'Authorization: Bearer <JWT_WITH_ADMIN_ROLE>' \
-d '{
  "model": "claude-sonnet-4-20250514",
  "messages": [{"role": "user", "content": "Hello"}]
}'
```

3. **Verify the role was updated**:

```bash
curl -X GET 'http://0.0.0.0:4000/user/info?user_id=user-123' \
-H 'Authorization: Bearer <PROXY_MASTER_KEY>'
```

## All JWT Params

[**See Code**](https://github.com/BerriAI/litellm/blob/b204f0c01c703317d812a1553363ab0cb989d5b6/litellm/proxy/_types.py#L95)


