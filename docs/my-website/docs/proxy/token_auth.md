# [BETA] JWT-based Auth 

Use JWT's to auth admin's into the proxy.

:::info

This is a new feature, and subject to changes based on feedback.

:::

## Step 1. Set env's 

```bash
export JWT_PUBLIC_KEY_URL="" # "http://localhost:8080/realms/test-litellm-proxy/protocol/openid-connect/certs"
export JWT_ISSUER="" # http://localhost:8080/realms/test-litellm-proxy
```

## Step 2. Create JWT with scopes 

Create a client scope called `litellm_proxy_admin` in your OpenID provider (e.g. Keycloak).

Grant your user, `litellm_proxy_admin` scope when generating a JWT. 

```bash
curl --location 'http://{base_url}/realms/{your-realm}/protocol/openid-connect/token' \
--header 'Content-Type: application/x-www-form-urlencoded' \
--data-urlencode 'client_id={CLIENT_ID}' \
--data-urlencode 'client_secret={CLIENT_SECRET}' \
--data-urlencode 'username=test-{USERNAME}' \
--data-urlencode 'password={USER_PASSWORD}' \
--data-urlencode 'grant_type=password' \
--data-urlencode 'scope=litellm_proxy_admin' # 👈 grant this scope
```

## Step 3. Create a proxy key with JWT 

```bash
curl --location '{proxy_base_url}/key/generate' \
--header 'Authorization: Bearer eyJhbGciOiJSUzI1NiI...' \
--header 'Content-Type: application/json' \
--data '{}'
```