import Image from '@theme/IdealImage';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# AI Hub

Share models and agents with your organization. Show developers what's available without needing to rebuild them.

This feature is **available in v1.74.3-stable and above**.

## Overview

Admin can select models/agents to expose on public AI hub → Users go to the public url and see what's available. 

<Image img={require('../../img/final_public_model_hub_view.png')} />  

## Models

### How to use

#### 1. Go to the Admin UI

Navigate to the Model Hub page in the Admin UI (`PROXY_BASE_URL/ui/?login=success&page=model-hub-table`)

<Image img={require('../../img/model_hub_admin_view.png')} />  

#### 2. Select the models you want to expose

Click on `Make Public` and select the models you want to expose.

<Image img={require('../../img/make_public_modal.png')} />  

#### 3. Confirm the changes

<Image img={require('../../img/make_public_modal_confirmation.png')} />  

#### 4. Success! 

Go to the public url (`PROXY_BASE_URL/ui/model_hub_table`) and see available models. 

<Image img={require('../../img/final_public_model_hub_view.png')} />  

### API Endpoints

- `GET /public/model_hub` – returns the list of public model groups. Requires a valid user API key.
- `GET /public/model_hub/info` – returns metadata (docs title, version, useful links) for the public model hub.

## Agents

Share pre-built agents (A2A spec) across your organization. Users can discover and use agents without rebuilding them.

### How to use

#### Step 1. Create an agent

Create an agent that follows the [A2A spec](https://a2a.dev/).

```bash
curl -X POST 'http://0.0.0.0:4000/v1/agents' \
--header 'Authorization: Bearer <your-master-key>' \
--header 'Content-Type: application/json' \
--data '{
  "agent_name": "hello-world-agent",
  "agent_card_params": {
    "protocolVersion": "1.0",
    "name": "Hello World Agent",
    "description": "Just a hello world agent",
    "url": "http://localhost:9999/",
    "version": "1.0.0",
    "defaultInputModes": ["text"],
    "defaultOutputModes": ["text"],
    "capabilities": {
      "streaming": true
    },
    "skills": [
      {
        "id": "hello_world",
        "name": "Returns hello world",
        "description": "just returns hello world",
        "tags": ["hello world"],
        "examples": ["hi", "hello world"]
      }
    ]
  }
}'
```

**Expected Response**

```json
{
  "agent_id": "123e4567-e89b-12d3-a456-426614174000",
  "agent_name": "hello-world-agent",
  "agent_card_params": {
    "protocolVersion": "1.0",
    "name": "Hello World Agent",
    "description": "Just a hello world agent",
    "url": "http://localhost:9999/",
    "version": "1.0.0",
    "defaultInputModes": ["text"],
    "defaultOutputModes": ["text"],
    "capabilities": {
      "streaming": true
    },
    "skills": [
      {
        "id": "hello_world",
        "name": "Returns hello world",
        "description": "just returns hello world",
        "tags": ["hello world"],
        "examples": ["hi", "hello world"]
      }
    ]
  },
  "created_at": "2025-11-15T10:30:00Z",
  "created_by": "user123"
}
```

#### Step 2. Make agent public

Make the agent discoverable on the AI Hub.

**Option 1: Make single agent public**

```bash
curl -X POST 'http://0.0.0.0:4000/v1/agents/123e4567-e89b-12d3-a456-426614174000/make_public' \
--header 'Authorization: Bearer <your-master-key>' \
--header 'Content-Type: application/json'
```

**Option 2: Make multiple agents public**

```bash
curl -X POST 'http://0.0.0.0:4000/v1/agents/make_public' \
--header 'Authorization: Bearer <your-master-key>' \
--header 'Content-Type: application/json' \
--data '{
  "agent_ids": [
    "123e4567-e89b-12d3-a456-426614174000",
    "123e4567-e89b-12d3-a456-426614174001"
  ]
}'
```

**Expected Response**

```json
{
  "message": "Successfully updated public agent groups",
  "public_agent_groups": [
    "123e4567-e89b-12d3-a456-426614174000"
  ],
  "updated_by": "user123"
}
```

#### Step 3. View public agents

Users can now discover the agent via the public endpoint.

```bash
curl -X GET 'http://0.0.0.0:4000/public/agent_hub' \
--header 'Authorization: Bearer <user-api-key>'
```

**Expected Response**

```json
[
  {
    "protocolVersion": "1.0",
    "name": "Hello World Agent",
    "description": "Just a hello world agent",
    "url": "http://localhost:9999/",
    "version": "1.0.0",
    "defaultInputModes": ["text"],
    "defaultOutputModes": ["text"],
    "capabilities": {
      "streaming": true
    },
    "skills": [
      {
        "id": "hello_world",
        "name": "Returns hello world",
        "description": "just returns hello world",
        "tags": ["hello world"],
        "examples": ["hi", "hello world"]
      }
    ]
  }
]
```

### List all agents

View all agents in your organization (admin only).

```bash
curl -X GET 'http://0.0.0.0:4000/v1/agents' \
--header 'Authorization: Bearer <your-master-key>'
```

### Get specific agent

Get details for a specific agent.

```bash
curl -X GET 'http://0.0.0.0:4000/v1/agents/123e4567-e89b-12d3-a456-426614174000' \
--header 'Authorization: Bearer <your-api-key>'
```

### Update an agent

Update an existing agent.

```bash
curl -X PUT 'http://0.0.0.0:4000/v1/agents/123e4567-e89b-12d3-a456-426614174000' \
--header 'Authorization: Bearer <your-master-key>' \
--header 'Content-Type: application/json' \
--data '{
  "agent_name": "hello-world-agent",
  "agent_card_params": {
    "protocolVersion": "1.0",
    "name": "Hello World Agent v2",
    "description": "Updated hello world agent",
    "url": "http://localhost:9999/",
    "version": "2.0.0",
    "defaultInputModes": ["text"],
    "defaultOutputModes": ["text"],
    "capabilities": {
      "streaming": true
    },
    "skills": []
  }
}'
```

### Delete an agent

Delete an agent from your organization.

```bash
curl -X DELETE 'http://0.0.0.0:4000/v1/agents/123e4567-e89b-12d3-a456-426614174000' \
--header 'Authorization: Bearer <your-master-key>'
```

**Expected Response**

```json
{
  "message": "Agent 123e4567-e89b-12d3-a456-426614174000 deleted successfully"
}
```

## Additional Endpoints

### Get supported providers

Get a list of all providers supported by LiteLLM. No authentication required.

```bash
curl -s 'http://0.0.0.0:4000/public/providers' | jq
```

