import Image from '@theme/IdealImage';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Setting Tag Budgets

Track spend and set budgets for your API requests using tags. Tags allow you to categorize and monitor costs across different cost centers, projects, and departments.

## Pre-Requisites

- You must set up a Postgres database (e.g. Supabase, Neon, etc.)

## What are Tags?

Tags are labels you can attach to your LLM requests to track and limit spending by category. 

**Common Use Cases:**
- **Cost Center Tracking**: Allocate LLM costs to specific departments or business units (e.g., "engineering", "marketing", "customer-support")
- **Project-based Budgeting**: Set budgets for different projects or initiatives (e.g., "project-alpha", "chatbot-v2")
- **Customer Attribution**: Track spend per customer or client (e.g., "customer-acme", "customer-techcorp")
- **Feature Monitoring**: Monitor costs for specific features (e.g., "feature-chat", "feature-summarization")

Tags can be set on API keys, so all requests made with that key are automatically tracked under the specified cost center or budget category.

## Setting Tag Budgets

### 1. Create a tag with budget

Create a tag to represent a cost center, project, or any budget category. Set `max_budget` ($ value allowed) and `budget_duration` (how frequently the budget resets).

**Example:** Create a tag for your Engineering department with a monthly $500 budget

<Tabs>

<TabItem value="API" label="API">

Create a new tag and set `max_budget` and `budget_duration`

```shell
curl -X POST 'http://0.0.0.0:4000/tag/new' \
     -H 'Authorization: Bearer sk-1234' \
     -H 'Content-Type: application/json' \
     -d '{
            "name": "engineering", 
            "description": "Engineering department cost center",
            "max_budget": 500.0, 
            "budget_duration": "30d"
        }' 
```

**Request Body Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `name` | string | Yes | Unique name for the tag (e.g., cost center name) |
| `description` | string | No | Description of what this tag tracks |
| `models` | list[string] | No | Restrict tag to specific models |
| `max_budget` | float | No | Maximum budget in USD |
| `budget_duration` | string | No | How often budget resets (e.g., "30d", "1d") |
| `soft_budget` | float | No | Soft budget limit for warnings |

**Response:**

```json
{
  "name": "engineering",
  "description": "Engineering department cost center",
  "max_budget": 500.0,
  "budget_duration": "30d",
  "budget_reset_at": "2025-11-10T00:00:00Z",
  "created_at": "2025-10-11T00:00:00Z"
}  
```

</TabItem>

<TabItem value="UI" label="Admin UI">

Navigate to the **Tag Management** page and click **Create New Tag**. Fill in the tag details and set your budget:

<Image 
  img={require('../../img/tag_budget1.png')}
  style={{width: '80%', display: 'block', margin: '0'}}
/>

</TabItem>

</Tabs>

**Possible values for `budget_duration`:**

| `budget_duration` | When Budget will reset |
| --- | --- |
| `budget_duration="1s"` | every 1 second |
| `budget_duration="1m"` | every 1 minute |
| `budget_duration="1h"` | every 1 hour |
| `budget_duration="1d"` | every 1 day |
| `budget_duration="7d"` | every 1 week |
| `budget_duration="30d"` | every 1 month |

### 2. Use the tag in your requests

There are two ways to use tags:

**Option A: Set tags on API keys** (Recommended for cost center tracking)

When you create or update a key, add tags to it. All requests made with that key will automatically be tracked under those tags:

```shell
curl -X POST 'http://0.0.0.0:4000/key/generate' \
     -H 'Authorization: Bearer sk-1234' \
     -H 'Content-Type: application/json' \
     -d '{
           "metadata": {
               "tags": ["engineering"]
           }
         }'
```

Now any request made with this key will be tracked under the "engineering" tag and count against its budget.

**Option B: Add tags per request**

Alternatively, add the tag to individual API requests in the `metadata` field:

<Tabs>

<TabItem value="openai" label="OpenAI SDK">

```python
import openai

client = openai.OpenAI(
    api_key="sk-1234",  # Your LiteLLM proxy key
    base_url="http://0.0.0.0:4000"
)

response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello"}],
    extra_body={
        "metadata": {
            "tags": ["engineering"]
        }
    }
)
```

</TabItem>

<TabItem value="curl" label="cURL">

```shell
curl -X POST 'http://0.0.0.0:4000/chat/completions' \
     -H 'Authorization: Bearer sk-1234' \
     -H 'Content-Type: application/json' \
     -d '{
           "model": "gpt-4",
           "messages": [{"role": "user", "content": "Hello"}],
           "metadata": {
               "tags": ["engineering"]
           }
         }'
```

</TabItem>

<TabItem value="litellm" label="LiteLLM SDK">

```python
import litellm

response = litellm.completion(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello"}],
    api_key="sk-1234",
    api_base="http://0.0.0.0:4000",
    metadata={
        "tags": ["engineering"]
    }
)
```

</TabItem>

</Tabs>

### 3. Test It

Make requests until the budget is exceeded:

```shell
curl -X POST 'http://0.0.0.0:4000/chat/completions' \
     -H 'Authorization: Bearer sk-1234' \
     -H 'Content-Type: application/json' \
     -d '{
           "model": "gpt-4",
           "messages": [{"role": "user", "content": "Hello"}],
           "metadata": {
               "tags": ["engineering"]
           }
         }'
```

**When budget is exceeded, you'll see:**

```json
{
  "error": {
    "message": "Budget has been exceeded! Tag=engineering Current cost: 505.50, Max budget: 500.0",
    "type": "budget_exceeded",
    "param": null,
    "code": "400"
  }
}
```

## Managing Tags

### View Tag Information

<Tabs>

<TabItem value="API" label="API">

Get information about specific tags:

```shell
curl -X POST 'http://0.0.0.0:4000/tag/info' \
     -H 'Authorization: Bearer sk-1234' \
     -H 'Content-Type: application/json' \
     -d '{
           "names": ["engineering", "marketing"]
         }'
```

**Response:**

```json
{
  "engineering": {
    "name": "engineering",
    "description": "Engineering department cost center",
    "spend": 245.50,
    "max_budget": 500.0,
    "budget_duration": "30d",
    "budget_reset_at": "2025-11-10T00:00:00Z",
    "created_at": "2025-10-11T00:00:00Z",
    "updated_at": "2025-10-11T12:30:00Z"
  },
  "marketing": {
    "name": "marketing",
    "description": "Marketing department cost center",
    "spend": 89.20,
    "max_budget": 300.0,
    "budget_duration": "30d",
    "budget_reset_at": "2025-11-10T00:00:00Z",
    "created_at": "2025-10-11T00:00:00Z",
    "updated_at": "2025-10-11T12:30:00Z"
  }
}
```

</TabItem>

<TabItem value="UI" label="Admin UI">

Click on any tag name in the Tag Management page to view detailed information, including current spend and budget status.

</TabItem>

</Tabs>

### Update Tag Budget

<Tabs>

<TabItem value="API" label="API">

Update an existing tag's budget:

```shell
curl -X POST 'http://0.0.0.0:4000/tag/update' \
     -H 'Authorization: Bearer sk-1234' \
     -H 'Content-Type: application/json' \
     -d '{
           "name": "engineering",
           "max_budget": 750.0,
           "budget_duration": "30d"
         }'
```

</TabItem>

<TabItem value="UI" label="Admin UI">

Click the edit icon next to any tag to modify its budget settings.

</TabItem>

</Tabs>

### Delete Tag

<Tabs>

<TabItem value="API" label="API">

```shell
curl -X POST 'http://0.0.0.0:4000/tag/delete' \
     -H 'Authorization: Bearer sk-1234' \
     -H 'Content-Type: application/json' \
     -d '{
           "name": "engineering"
         }'
```

</TabItem>

<TabItem value="UI" label="Admin UI">

Click the delete icon next to any tag to remove it.

</TabItem>

</Tabs>

## Multiple Tags

You can apply multiple tags to track costs across different dimensions simultaneously. For example, track both the cost center and the specific project:

**Set multiple tags on a key:**

```shell
curl -X POST 'http://0.0.0.0:4000/key/generate' \
     -H 'Authorization: Bearer sk-1234' \
     -H 'Content-Type: application/json' \
     -d '{
           "metadata": {
               "tags": ["engineering", "project-alpha", "customer-acme"]
           }
         }'
```

**Or add multiple tags per request:**

```python
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello"}],
    extra_body={
        "metadata": {
            "tags": ["engineering", "project-alpha", "customer-acme"]
        }
    }
)
```

**Budget Enforcement:** If any tag exceeds its budget, the request will be rejected.
