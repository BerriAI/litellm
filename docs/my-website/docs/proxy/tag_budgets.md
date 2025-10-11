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

Tags are added to each request in the `metadata` field to track and enforce budget limits.

## Setting Tag Budgets

### 1. Create a tag with budget

Create a tag to represent a cost center, project, or any budget category. Set `max_budget` ($ value allowed) and `budget_duration` (how frequently the budget resets).

**Example:** Create a tag for your Engineering department with a monthly $500 budget

#### API

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

#### LiteLLM Admin UI

Navigate to the **Tag Management** page and click **Create New Tag**. Fill in the tag details and set your budget:

<Image 
  img={require('../../img/tag_budget1.png')}
  style={{width: '80%', display: 'block', margin: '0'}}
/>

<br />


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

Add tags to your API requests in the `metadata` field:

:::info Tags Budgets on API Keys

Currently, tag budget enforcement is only supported per request. If you'd like to set tags on API keys so all requests automatically inherit the tags budgets, please [create a feature request on GitHub](https://github.com/BerriAI/litellm/issues/new?assignees=&labels=enhancement&projects=&template=feature_request.yml&title=%5BFeat%5D%3A).

:::

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

### Update Tag Budget

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

### Delete Tag

```shell
curl -X POST 'http://0.0.0.0:4000/tag/delete' \
     -H 'Authorization: Bearer sk-1234' \
     -H 'Content-Type: application/json' \
     -d '{
           "name": "engineering"
         }'
```

## Multiple Tags per Request

You can apply multiple tags to a single request to track costs across different dimensions simultaneously. For example, track both the cost center and the specific project:

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

```shell
curl -X POST 'http://0.0.0.0:4000/chat/completions' \
     -H 'Authorization: Bearer sk-1234' \
     -H 'Content-Type: application/json' \
     -d '{
           "model": "gpt-4",
           "messages": [{"role": "user", "content": "Hello"}],
           "metadata": {
               "tags": ["engineering", "project-alpha", "customer-acme"]
           }
         }'
```

**Budget Enforcement:** If any tag exceeds its budget, the request will be rejected.
