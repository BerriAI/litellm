import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# JWT-based Auth 

Use JWT's to auth admins / projects into the proxy.

:::info

✨ JWT-based Auth  is on LiteLLM Enterprise starting at $250/mo

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

## Advanced - Spend Tracking (End-Users / Internal Users / Team / Org)

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