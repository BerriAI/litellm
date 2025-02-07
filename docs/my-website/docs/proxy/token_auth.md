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

## Advanced - Set Accepted JWT Scope Names 

Change the string in JWT 'scopes', that litellm evaluates to see if a user has admin access.

```yaml
general_settings:
  master_key: sk-1234
  enable_jwt_auth: True
  litellm_jwtauth:
    admin_jwt_scope: "litellm-proxy-admin"
```

## Tracking End-Users / Internal Users / Team / Org

Set the field in the jwt token, which corresponds to a litellm user / team / org.

```yaml
general_settings:
  master_key: sk-1234
  enable_jwt_auth: True
  litellm_jwtauth:
    admin_jwt_scope: "litellm-proxy-admin"
    team_id_jwt_field: "client_id" # 👈 CAN BE ANY FIELD
    user_id_jwt_field: "sub" # 👈 CAN BE ANY FIELD
    org_id_jwt_field: "org_id" # 👈 CAN BE ANY FIELD
    end_user_id_jwt_field: "customer_id" # 👈 CAN BE ANY FIELD
```

Expected JWT: 

```
{
  "client_id": "my-unique-team",
  "sub": "my-unique-user",
  "org_id": "my-unique-org",
}
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

## Control model access with Teams


1. Specify the JWT field that contains the team ids, that the user belongs to. 

```yaml
general_settings:
  master_key: sk-1234
  litellm_jwtauth:
    user_id_jwt_field: "sub"
    team_ids_jwt_field: "groups" 
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


## Advanced - Allowed Routes 

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

## Advanced - Caching Public Keys 

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

## Advanced - Custom JWT Field 

Set a custom field in which the team_id exists. By default, the 'client_id' field is checked. 

```yaml
general_settings:
  master_key: sk-1234
  enable_jwt_auth: True
  litellm_jwtauth:
    team_id_jwt_field: "client_id" # 👈 KEY CHANGE
```

## All Params

[**See Code**](https://github.com/BerriAI/litellm/blob/b204f0c01c703317d812a1553363ab0cb989d5b6/litellm/proxy/_types.py#L95)




## Advanced - Block Teams 

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


## Advanced - Upsert Users + Allowed Email Domains 

Allow users who belong to a specific email domain, automatic access to the proxy.
 
```yaml
general_settings:
  master_key: sk-1234
  enable_jwt_auth: True
  litellm_jwtauth:
    user_email_jwt_field: "email" # 👈 checks 'email' field in jwt payload
    user_allowed_email_domain: "my-co.com" # allows user@my-co.com to call proxy
    user_id_upsert: true # 👈 upserts the user to db, if valid email but not in db
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

- `object_id_jwt_field`: The field in the JWT token that contains the object id. This id can be either a user id or a team id. Use this instead of `user_id_jwt_field` and `team_id_jwt_field`. If the same field could be both. 

- `roles_jwt_field`: The field in the JWT token that contains the roles. This field is a list of roles that the user has. To index into a nested field, use dot notation - eg. `resource_access.litellm-test-client-id.roles`.

- `role_mappings`: A list of role mappings. Map the received role in the JWT token to an internal role on LiteLLM.

- `JWT_AUDIENCE`: The audience of the JWT token. This is used to validate the audience of the JWT token. Set via an environment variable.

### Example Token 

```
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

```
{
  "scope": ["litellm.api.consumer", "litellm.api.gpt_3_5_turbo"]
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