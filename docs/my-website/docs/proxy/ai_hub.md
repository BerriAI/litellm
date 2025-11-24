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

Click on `Select Models to Make Public` and select the models you want to expose.

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

:::info
Agents are only available in v1.79.4-stable and above.
:::

Share pre-built agents (A2A spec) across your organization. Users can discover and use agents without rebuilding them.

[**Demo Video**](https://drive.google.com/file/d/1r-_Rtiu04RW5Fwwu3_eshtA1oZtC3_DH/view?usp=sharing)

### 1. Create an agent

Create an agent that follows the [A2A spec](https://a2a.dev/).

<Tabs>
<TabItem value="ui" label="UI">

<Image img={require('../../img/add_agent.png')} />  

</TabItem>
<TabItem value="api" label="API">
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

</TabItem>
</Tabs>

### 2. Make agent public

Make the agent discoverable on the AI Hub.

<Tabs>
<TabItem value="ui" label="UI">

Navigate to the Agents Tab on the AI Hub page 

<Image img={require('../../img/ai_hub_with_agents.png')} />  

Select the agents you want to make public and click on `Make Public` button.

<Image img={require('../../img/make_agents_public.png')} />  

</TabItem>
<TabItem value="api" label="API">

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

</TabItem>

</Tabs>



### 3. View public agents

Users can now discover the agent via the public endpoint.

<Tabs>
<TabItem value="ui" label="UI">

<Image img={require('../../img/public_agent_hub.png')} />  

</TabItem>
<TabItem value="api" label="API">

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

</TabItem>
</Tabs>


## MCP Servers

### How to use

#### 1. Add MCP Server

Go here for instructions: [MCP Overview](../mcp#adding-your-mcp)


#### 2. Make MCP server public

<Tabs>
<TabItem value="ui" label="UI">

Navigate to AI Hub page, and select the MCP tab (`PROXY_BASE_URL/ui/?login=success&page=mcp-server-table`)

<Image img={require('../../img/mcp_server_on_ai_hub.png')} />  

</TabItem>
<TabItem value="api" label="API">

```bash
curl -L -X POST 'http://localhost:4000/v1/mcp/make_public' \
-H 'Authorization: Bearer sk-1234' \ 
-H 'Content-Type: application/json' \
-d '{"mcp_server_ids":["e856f9a3-abc6-45b1-9d06-62fa49ac293d"]}'
```

</TabItem>
</Tabs>


#### 3. View public MCP servers

Users can now discover the MCP server via the public endpoint (`PROXY_BASE_URL/ui/model_hub_table`)

<Tabs>
<TabItem value="ui" label="UI">

<Image img={require('../../img/mcp_on_public_ai_hub.png')} />  

</TabItem>
<TabItem value="api" label="API">

```bash
curl -L -X GET 'http://0.0.0.0:4000/public/mcp_hub' \
-H 'Authorization: Bearer sk-1234'
```

**Expected Response**

```json
[
    {
        "server_id": "e856f9a3-abc6-45b1-9d06-62fa49ac293d",
        "name": "deepwiki-mcp",
        "alias": null,
        "server_name": "deepwiki-mcp",
        "url": "https://mcp.deepwiki.com/mcp",
        "transport": "http",
        "spec_path": null,
        "auth_type": "none",
        "mcp_info": {
            "server_name": "deepwiki-mcp",
            "description": "free mcp server "
        }
    },
    {
        "server_id": "a634819f-3f93-4efc-9108-e49c5b83ad84",
        "name": "deepwiki_2",
        "alias": "deepwiki_2",
        "server_name": "deepwiki_2",
        "url": "https://mcp.deepwiki.com/mcp",
        "transport": "http",
        "spec_path": null,
        "auth_type": "none",
        "mcp_info": {
            "server_name": "deepwiki_2",
            "mcp_server_cost_info": null
        }
    },
    {
        "server_id": "33f950e4-2edb-41fa-91fc-0b9581269be6",
        "name": "edc_mcp_server",
        "alias": "edc_mcp_server",
        "server_name": "edc_mcp_server",
        "url": "http://lelvdckdputildev.itg.ti.com:8085/api/mcp",
        "transport": "http",
        "spec_path": null,
        "auth_type": "none",
        "mcp_info": {
            "server_name": "edc_mcp_server",
            "mcp_server_cost_info": null
        }
    }
]
```

</TabItem>
</Tabs>