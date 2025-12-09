import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Control Public & Private Routes

:::info

Requires a LiteLLM Enterprise License. [Get a free trial](https://calendly.com/d/4mp-gd3-k5k/litellm-1-1-onboarding-chat).

:::

Control which routes require authentication and which routes are publicly accessible.

## Route Types

| Route Type | Requires Auth | Description |
|------------|---------------|-------------|
| `public_routes` | No | Routes accessible without any authentication |
| `admin_only_routes` | Yes (Admin only) | Routes only accessible by [Proxy Admin](./self_serve#available-roles) |
| `allowed_routes` | Yes | Routes exposed on the proxy. If not set, all routes are exposed |

## Quick Start

### Make Routes Public

Allow specific routes to be accessed without authentication:

```yaml
general_settings:
  master_key: sk-1234
  public_routes: ["LiteLLMRoutes.public_routes", "/spend/calculate"]
```

### Restrict Routes to Admin Only

Restrict certain routes to only be accessible by Proxy Admin:

```yaml
general_settings:
  master_key: sk-1234
  admin_only_routes: ["/key/generate", "/key/delete"]
```

### Limit Available Routes

Only expose specific routes on the proxy:

```yaml
general_settings:
  master_key: sk-1234
  allowed_routes: ["/chat/completions", "/embeddings", "LiteLLMRoutes.public_routes"]
```

## Usage Examples

### Define Public, Admin Only, and Allowed Routes

```yaml
general_settings:
  master_key: sk-1234
  public_routes: ["LiteLLMRoutes.public_routes", "/spend/calculate"]
  admin_only_routes: ["/key/generate"]
  allowed_routes: ["/chat/completions", "/spend/calculate", "LiteLLMRoutes.public_routes"]
```

`LiteLLMRoutes.public_routes` is an ENUM corresponding to the default public routes on LiteLLM. [View the source](https://github.com/BerriAI/litellm/blob/main/litellm/proxy/_types.py).

### Testing

<Tabs>

<TabItem value="public" label="Test public_routes">

```shell
curl --request POST \
  --url 'http://localhost:4000/spend/calculate' \
  --header 'Content-Type: application/json' \
  --data '{
    "model": "gpt-4",
    "messages": [{"role": "user", "content": "Hey, how'\''s it going?"}]
  }'
```

This endpoint works without an `Authorization` header.

</TabItem>

<TabItem value="admin_only_routes" label="Test admin_only_routes">

**Successful Request (Admin)**

```shell
curl --location 'http://0.0.0.0:4000/key/generate' \
--header 'Authorization: Bearer <your-master-key>' \
--header 'Content-Type: application/json' \
--data '{}'
```

**Unsuccessful Request (Non-Admin)**

```shell
curl --location 'http://0.0.0.0:4000/key/generate' \
--header 'Authorization: Bearer <virtual-key-from-non-admin>' \
--header 'Content-Type: application/json' \
--data '{"user_role": "internal_user"}'
```

**Expected Response**

```json
{
  "error": {
    "message": "user not allowed to access this route. Route=/key/generate is an admin only route",
    "type": "auth_error",
    "param": "None",
    "code": "403"
  }
}
```

</TabItem>

<TabItem value="allowed_routes" label="Test allowed_routes">

**Successful Request**

```shell
curl http://localhost:4000/chat/completions \
-H "Content-Type: application/json" \
-H "Authorization: Bearer sk-1234" \
-d '{
"model": "fake-openai-endpoint",
"messages": [
    {"role": "user", "content": "Hello, Claude"}
]
}'
```

**Unsuccessful Request (Route Not Allowed)**

```shell
curl --location 'http://0.0.0.0:4000/embeddings' \
--header 'Content-Type: application/json' \
-H "Authorization: Bearer sk-1234" \
--data '{
"model": "text-embedding-ada-002",
"input": ["write a litellm poem"]
}'
```

**Expected Response**

```json
{
  "error": {
    "message": "Route /embeddings not allowed",
    "type": "auth_error",
    "param": "None",
    "code": "403"
  }
}
```

</TabItem>

</Tabs>

## Advanced: Wildcard Patterns

Use wildcard patterns to match multiple routes at once.

### Syntax

| Pattern | Description | Example |
|---------|-------------|---------|
| `/path/*` | Matches any route starting with `/path/` | `/api/*` matches `/api/users`, `/api/users/123` |


### Examples

#### Make All Routes Under a Path Public

```yaml
general_settings:
  master_key: sk-1234
  public_routes:
    - "LiteLLMRoutes.public_routes"
    - "/api/v1/*"      # All routes under /api/v1/
    - "/health/*"       # All health check routes
```

#### Restrict Admin Routes with Wildcards

```yaml
general_settings:
  master_key: sk-1234
  admin_only_routes:
    - "/admin/*"        # All admin routes
    - "/internal/*"     # All internal routes
```

### Testing Wildcard Routes

**Config:**
```yaml
general_settings:
  master_key: sk-1234
  public_routes:
    - "/public/*"
```

**Test:**
```shell
# This works without auth (matches /public/*)
curl http://localhost:4000/public/status

# This also works without auth (matches /public/*)
curl http://localhost:4000/public/health/detailed

# This requires auth (doesn't match /public/*)
curl http://localhost:4000/private/data
```

