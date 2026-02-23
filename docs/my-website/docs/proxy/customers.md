import Image from '@theme/IdealImage';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Customers / End-Users

Track spend, set budgets and permissions for your customers.

## Tracking Customer Spend + Permissions

### 1. Make LLM API call w/ Customer ID

LiteLLM checks for a customer/end-user ID in the following order (first match wins):

| Priority | Method | Where | Notes |
|----------|--------|-------|-------|
| 1 | `x-litellm-customer-id` header | Request headers | Standard header, always checked |
| 2 | `x-litellm-end-user-id` header | Request headers | Standard header, always checked |
| 3 | Custom header via `user_header_mappings` | Request headers | Configured in `general_settings` |
| 4 | Custom header via `user_header_name` | Request headers | Deprecated — use `user_header_mappings` |
| 5 | `user` field | Request body | Standard OpenAI field |
| 6 | `litellm_metadata.user` field | Request body | Anthropic-style metadata |
| 7 | `metadata.user_id` field | Request body | Generic metadata pattern |
| 8 | `safety_identifier` field | Request body | Responses API |

**Option 1: Standard headers** (recommended — no request body modification needed)

```bash showLineNumbers title="Make request with customer ID in header"
curl -X POST 'http://0.0.0.0:4000/chat/completions' \
        --header 'Content-Type: application/json' \
        --header 'Authorization: Bearer sk-1234' \
        --header 'x-litellm-end-user-id: ishaan3' \
        --data '{
        "model": "azure-gpt-3.5",
        "messages": [{"role": "user", "content": "what time is it"}]
        }'
```

Both `x-litellm-customer-id` and `x-litellm-end-user-id` are supported and always checked without any configuration.

**Option 2: `user` field in request body** (OpenAI-compatible)

```bash showLineNumbers title="Make request with customer ID in body"
curl -X POST 'http://0.0.0.0:4000/chat/completions' \
        --header 'Content-Type: application/json' \
        --header 'Authorization: Bearer sk-1234' \
        --data '{
        "model": "azure-gpt-3.5",
        "user": "ishaan3",
        "messages": [{"role": "user", "content": "what time is it"}]
        }'
```

**Option 3: Custom header via `user_header_mappings`** (configurable)

```yaml showLineNumbers title="config.yaml"
general_settings:
  user_header_mappings:
    - header_name: "x-my-app-user-id"
      litellm_user_role: "customer"
```

```bash showLineNumbers title="Make request with custom header"
curl -X POST 'http://0.0.0.0:4000/chat/completions' \
        --header 'Content-Type: application/json' \
        --header 'Authorization: Bearer sk-1234' \
        --header 'x-my-app-user-id: ishaan3' \
        --data '{
        "model": "azure-gpt-3.5",
        "messages": [{"role": "user", "content": "what time is it"}]
        }'
```

**Option 4: `litellm_metadata.user`** (Anthropic-style)

```bash showLineNumbers title="Make request with litellm_metadata.user"
curl -X POST 'http://0.0.0.0:4000/chat/completions' \
        --header 'Content-Type: application/json' \
        --header 'Authorization: Bearer sk-1234' \
        --data '{
        "model": "claude-3-5-sonnet",
        "messages": [{"role": "user", "content": "what time is it"}],
        "litellm_metadata": {"user": "ishaan3"}
        }'
```

**Option 5: `metadata.user_id`**

```bash showLineNumbers title="Make request with metadata.user_id"
curl -X POST 'http://0.0.0.0:4000/chat/completions' \
        --header 'Content-Type: application/json' \
        --header 'Authorization: Bearer sk-1234' \
        --data '{
        "model": "azure-gpt-3.5",
        "messages": [{"role": "user", "content": "what time is it"}],
        "metadata": {"user_id": "ishaan3"}
        }'
```

The customer_id will be upserted into the DB with the new spend.

If the customer_id already exists, spend will be incremented.

### 2. Get Customer Spend 

<Tabs>
<TabItem value="all-up" label="All-up spend">

Call `/customer/info` to get a customer's all up spend

```bash showLineNumbers title="Get customer spend"
curl -X GET 'http://0.0.0.0:4000/customer/info?end_user_id=ishaan3' \ # 👈 CUSTOMER ID
        -H 'Authorization: Bearer sk-1234' \ # 👈 YOUR PROXY KEY
```

Expected Response:

```json showLineNumbers title="Response"
{
    "user_id": "ishaan3",
    "blocked": false,
    "alias": null,
    "spend": 0.001413,
    "allowed_model_region": null,
    "default_model": null,
    "litellm_budget_table": null
}
```

</TabItem>
<TabItem value="event-webhook" label="Event Webhook">

To update spend in your client-side DB, point the proxy to your webhook. 

E.g. if your server is `https://webhook.site` and your listening on `6ab090e8-c55f-4a23-b075-3209f5c57906`

1. Add webhook url to your proxy environment: 

```bash showLineNumbers title="Set webhook URL"
export WEBHOOK_URL="https://webhook.site/6ab090e8-c55f-4a23-b075-3209f5c57906"
```

2. Add 'webhook' to config.yaml

```yaml showLineNumbers title="config.yaml"
general_settings: 
  alerting: ["webhook"] # 👈 KEY CHANGE
```

3. Test it! 

```bash showLineNumbers title="Test webhook"
curl -X POST 'http://localhost:4000/chat/completions' \
-H 'Content-Type: application/json' \
-H 'Authorization: Bearer sk-1234' \
-D '{
    "model": "mistral",
    "messages": [
        {
        "role": "user",
        "content": "What's the weather like in Boston today?"
        }
    ],
    "user": "krrish12"
}
'
```

Expected Response 

```json showLineNumbers title="Webhook event payload"
{
  "spend": 0.0011120000000000001, # 👈 SPEND
  "max_budget": null,
  "token": "example-api-key-123",
  "customer_id": "krrish12",  # 👈 CUSTOMER ID
  "user_id": null,
  "team_id": null,
  "user_email": null,
  "key_alias": null,
  "projected_exceeded_date": null,
  "projected_spend": null,
  "event": "spend_tracked",
  "event_group": "customer",
  "event_message": "Customer spend tracked. Customer=krrish12, spend=0.0011120000000000001"
}
```

[See Webhook Spec](./alerting.md#api-spec-for-webhook-event)

</TabItem>
</Tabs>


## Setting Customer Object Permissions

Control which resources (MCP servers, vector stores, agents) a customer can access.

### What are Object Permissions?

Object permissions allow you to restrict customer access to specific:
- **MCP Servers**: Limit which MCP servers the customer can call
- **MCP Access Groups**: Assign customers to predefined groups of MCP servers
- **MCP Tool Permissions**: Granular control over which tools within an MCP server the customer can use
- **Vector Stores**: Control which vector stores the customer can query
- **Agents**: Restrict which agents the customer can interact with
- **Agent Access Groups**: Assign customers to predefined groups of agents

### Creating a Customer with Object Permissions

```bash showLineNumbers title="Create customer with object permissions"
curl -L -X POST 'http://localhost:4000/customer/new' \
-H 'Authorization: Bearer sk-1234' \
-H 'Content-Type: application/json' \
-d '{
    "user_id": "user_1",
    "object_permission": {
      "mcp_servers": ["server_1", "server_2"],
      "mcp_access_groups": ["public_group"],
      "mcp_tool_permissions": {
        "server_1": ["tool_a", "tool_b"]
      },
      "vector_stores": ["vector_store_1"],
      "agents": ["agent_1"],
      "agent_access_groups": ["basic_agents"]
    }
  }'
```

**Parameters:**
- `mcp_servers` (Optional[List[str]]): List of allowed MCP server IDs
- `mcp_access_groups` (Optional[List[str]]): List of MCP access group names
- `mcp_tool_permissions` (Optional[Dict[str, List[str]]]): Map of server ID to allowed tool names
- `vector_stores` (Optional[List[str]]): List of allowed vector store IDs
- `agents` (Optional[List[str]]): List of allowed agent IDs
- `agent_access_groups` (Optional[List[str]]): List of agent access group names

**Note:** If `object_permission` is `null` or `{}`, the customer has no object-level restrictions.

### Updating Customer Object Permissions

You can update object permissions for existing customers:

```bash showLineNumbers title="Update customer object permissions"
curl -L -X POST 'http://localhost:4000/customer/update' \
-H 'Authorization: Bearer sk-1234' \
-H 'Content-Type: application/json' \
-d '{
    "user_id": "user_1",
    "object_permission": {
      "mcp_servers": ["server_3"],
      "vector_stores": ["vector_store_2", "vector_store_3"]
    }
  }'
```

### Viewing Customer Object Permissions

When you query customer info, object permissions are included in the response:

```bash showLineNumbers title="Get customer info with object permissions"
curl -X GET 'http://0.0.0.0:4000/customer/info?end_user_id=user_1' \
    -H 'Authorization: Bearer sk-1234'
```

**Response:**
```json showLineNumbers title="Response with object permissions"
{
  "user_id": "user_1",
  "blocked": false,
  "alias": "John Doe",
  "spend": 0.0,
  "object_permission": {
    "object_permission_id": "perm_abc123",
    "mcp_servers": ["server_1", "server_2"],
    "mcp_access_groups": ["public_group"],
    "mcp_tool_permissions": {
      "server_1": ["tool_a", "tool_b"]
    },
    "vector_stores": ["vector_store_1"],
    "agents": ["agent_1"],
    "agent_access_groups": ["basic_agents"]
  },
  "litellm_budget_table": null
}
```

### Use Cases

**1. Tiered Access Control**
Create different permission tiers for your customers:

```bash showLineNumbers title="Free tier customer"
# Free tier - limited access
curl -L -X POST 'http://localhost:4000/customer/new' \
-H 'Authorization: Bearer sk-1234' \
-H 'Content-Type: application/json' \
-d '{
    "user_id": "free_user",
    "budget_id": "free_tier",
    "object_permission": {
      "mcp_access_groups": ["public_group"],
      "agent_access_groups": ["basic_agents"]
    }
  }'
```

```bash showLineNumbers title="Premium tier customer"
# Premium tier - full access
curl -L -X POST 'http://localhost:4000/customer/new' \
-H 'Authorization: Bearer sk-1234' \
-H 'Content-Type: application/json' \
-d '{
    "user_id": "premium_user",
    "budget_id": "premium_tier",
    "object_permission": {
      "mcp_servers": ["server_1", "server_2", "server_3"],
      "vector_stores": ["vector_store_1", "vector_store_2"],
      "agents": ["agent_1", "agent_2", "agent_3"]
    }
  }'
```

**2. Department-Specific Access**
Restrict customers to resources relevant to their department:

```bash showLineNumbers title="Sales team customer"
curl -L -X POST 'http://localhost:4000/customer/new' \
-H 'Authorization: Bearer sk-1234' \
-H 'Content-Type: application/json' \
-d '{
    "user_id": "sales_user",
    "object_permission": {
      "mcp_servers": ["crm_server", "email_server"],
      "agents": ["sales_assistant"],
      "vector_stores": ["sales_knowledge_base"]
    }
  }'
```

**3. Tool-Level Restrictions**
Grant access to specific tools within an MCP server:

```bash showLineNumbers title="Limited tool access"
curl -L -X POST 'http://localhost:4000/customer/new' \
-H 'Authorization: Bearer sk-1234' \
-H 'Content-Type: application/json' \
-d '{
    "user_id": "restricted_user",
    "object_permission": {
      "mcp_servers": ["database_server"],
      "mcp_tool_permissions": {
        "database_server": ["read_only_query", "get_table_schema"]
      }
    }
  }'
```

## Setting Customer Budgets

Set customer budgets (e.g. monthly budgets, tpm/rpm limits) on LiteLLM Proxy 

### Default Budget for All Customers

Apply budget limits to all customers without explicit budgets. This is useful for rate limiting and spending controls across all end users.

**Step 1: Create a default budget**

```bash showLineNumbers title="Create default budget"
curl -X POST 'http://localhost:4000/budget/new' \
-H 'Content-Type: application/json' \
-H 'Authorization: Bearer sk-1234' \
-d '{
    "max_budget": 10,
    "rpm_limit": 2,
    "tpm_limit": 1000
}'
```

**Step 2: Configure the default budget ID**

```yaml showLineNumbers title="config.yaml"
litellm_settings:
  max_end_user_budget_id: "budget_id_from_step_1"
```

**Step 3: Test it**

```bash showLineNumbers title="Make request with customer ID"
curl -X POST 'http://localhost:4000/chat/completions' \
-H 'Content-Type: application/json' \
-H 'Authorization: Bearer sk-1234' \
-d '{
    "model": "gpt-3.5-turbo",
    "messages": [{"role": "user", "content": "Hello"}],
    "user": "my-customer-id"
}'
```

The customer will be subject to the default budget limits (RPM, TPM, and $ budget). Customers with explicit budgets are unaffected.

### Quick Start 

Create / Update a customer with budget

**Create New Customer w/ budget**
```bash showLineNumbers title="Create customer with budget"
curl -X POST 'http://0.0.0.0:4000/customer/new'         
    -H 'Authorization: Bearer sk-1234'         
    -H 'Content-Type: application/json'         
    -d '{
        "user_id" : "my-customer-id",
        "max_budget": "0", # 👈 CAN BE FLOAT
    }'
```

**Test it!**

```bash showLineNumbers title="Test customer budget"
curl -X POST 'http://localhost:4000/chat/completions' \
-H 'Content-Type: application/json' \
-H 'Authorization: Bearer sk-1234' \
-D '{
    "model": "mistral",
    "messages": [
        {
        "role": "user",
        "content": "What'\''s the weather like in Boston today?"
        }
    ],
    "user": "ishaan-jaff-48"
}
```

### Assign Pricing Tiers

Create and assign customers to pricing tiers.

#### 1. Create a budget

<Tabs>
<TabItem value="ui" label="UI">

- Go to the 'Budgets' tab on the UI. 
- Click on '+ Create Budget'.
- Create your pricing tier (e.g. 'my-free-tier' with budget $4). This means each user on this pricing tier will have a max budget of $4. 

<Image img={require('../../img/create_budget_modal.png')} />

</TabItem>
<TabItem value="api" label="API">

Use the `/budget/new` endpoint for creating a new budget. [API Reference](https://litellm-api.up.railway.app/#/budget%20management/new_budget_budget_new_post)

```bash showLineNumbers title="Create budget via API"
curl -X POST 'http://localhost:4000/budget/new' \
-H 'Content-Type: application/json' \
-H 'Authorization: Bearer sk-1234' \
-D '{
    "budget_id": "my-free-tier", 
    "max_budget": 4 
}
```

</TabItem>
</Tabs>


#### 2. Assign Budget to Customer 

In your application code, assign budget when creating a new customer. 

Just use the `budget_id` used when creating the budget. In our example, this is `my-free-tier`.

```bash showLineNumbers title="Assign budget to customer"
curl -X POST 'http://localhost:4000/customer/new' \
-H 'Content-Type: application/json' \
-H 'Authorization: Bearer sk-1234' \
-D '{
    "user_id": "my-customer-id",
    "budget_id": "my-free-tier" # 👈 KEY CHANGE
}
```

#### 3. Test it! 

<Tabs>
<TabItem value="curl" label="curl">

```bash showLineNumbers title="Test with curl"
curl -X POST 'http://localhost:4000/customer/new' \
-H 'Content-Type: application/json' \
-H 'Authorization: Bearer sk-1234' \
-D '{
    "user_id": "my-customer-id",
    "budget_id": "my-free-tier" # 👈 KEY CHANGE
}
```

</TabItem>
<TabItem value="openai" label="OpenAI">

```python showLineNumbers title="Test with OpenAI SDK"
from openai import OpenAI
client = OpenAI(
  base_url="<your_proxy_base_url>",
  api_key="<your_proxy_key>"
)

completion = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ],
  user="my-customer-id"
)

print(completion.choices[0].message)
```

</TabItem>
</Tabs>