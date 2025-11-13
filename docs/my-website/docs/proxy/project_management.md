# [Beta] Project Management

Projects in LiteLLM sit between teams and keys in the organizational hierarchy, enabling fine-grained access control and budget management for specific use cases or applications.

```mermaid
graph TD
    A[Organization] --> B[Team 1]
    A --> C[Team 2]
    B --> D[Project A]
    B --> E[Project B]
    C --> F[Project C]
    D --> G[API Key 1]
    D --> H[API Key 2]
    E --> I[API Key 3]
    F --> J[API Key 4]
    
    style A fill:#e1f5ff
    style B fill:#fff4e6
    style C fill:#fff4e6
    style D fill:#f3e5f5
    style E fill:#f3e5f5
    style F fill:#f3e5f5
    style G fill:#e8f5e9
    style H fill:#e8f5e9
    style I fill:#e8f5e9
    style J fill:#e8f5e9
```

**Hierarchy**: `Organizations > Teams > Projects > Keys`

## Quick Start

This walkthrough shows how to create a project, generate an API key, make requests, and view project-level spend tracking in the UI.

### Step 1: Create a Project

```bash showLineNumbers
curl --location 'http://0.0.0.0:4000/project/new' \
--header 'Authorization: Bearer sk-1234' \
--header 'Content-Type: application/json' \
--data '{
    "project_alias": "flight-search-assistant",
    "team_id": "ad898803-c8a3-4f4a-976a-a3c372cffa45",
    "models": ["gpt-4", "gpt-3.5-turbo"],
    "max_budget": 100,
    "metadata": {
        "use_case_id": "SNOW-12345",
        "responsible_ai_id": "RAI-67890"
    }
}' | jq
```

**Response:**
```json
{
  "project_id": "e402a141-725a-4437-bff5-d47459189716",
  "project_alias": "flight-search-assistant",
  "team_id": "ad898803-c8a3-4f4a-976a-a3c372cffa45",
  "models": ["gpt-4", "gpt-3.5-turbo"],
  "max_budget": 100,
  ...
}
```

### Step 2: Generate API Key for Project

```bash showLineNumbers
curl 'http://0.0.0.0:4000/key/generate' \
--header 'Authorization: Bearer sk-1234' \
--header 'Content-Type: application/json' \
--data-raw '{
    "models": ["gpt-3.5-turbo", "gpt-4"],
    "metadata": {"user": "ishaan@berri.ai"},
    "project_id": "e402a141-725a-4437-bff5-d47459189716"
}' | jq
```

**Response:**
```json
{
  "key": "sk-W8VbscpfuyvHm5TkxRYiXA",
  "key_name": "sk-...YiXA",
  "project_id": "e402a141-725a-4437-bff5-d47459189716",
  ...
}
```

### Step 3: Use API Key in Chat Completions

```bash showLineNumbers
curl http://localhost:4000/v1/chat/completions \
--header 'Content-Type: application/json' \
--header 'Authorization: Bearer sk-W8VbscpfuyvHm5TkxRYiXA' \
--data '{
    "model": "gpt-4",
    "messages": [{"role": "user", "content": "What is litellm?"}]
}' | jq
```

### Step 4: View Project Spend in UI

Navigate to the **Logs** page in the LiteLLM Admin UI. You'll see the `user_api_key_project_id` tracked in the request metadata:

![Project Spend Tracking](/img/project_spend.png)

As shown above, the spend logs metadata includes:
- `"user_api_key_project_id": "e402a141-725a-4437-bff5-d47459189716"` - Links the request to your project
- All costs and token usage are automatically attributed to the project
- You can query and filter logs by project ID for detailed reporting

## API Endpoints

### POST /project/new

Create a new project.

**Who can call**: Admins or Team Admins

**Parameters**:
- `project_alias` (string, optional): Human-readable name for the project
- `team_id` (string, required): The team this project belongs to
- `models` (array, optional): List of models the project can access
- `max_budget` (float, optional): Maximum spend budget for the project
- `tpm_limit` (int, optional): Tokens per minute limit
- `rpm_limit` (int, optional): Requests per minute limit
- `budget_duration` (string, optional): Budget reset period (e.g., "30d", "1mo")
- `metadata` (object, optional): Custom metadata for the project
- `blocked` (boolean, optional): Block all API calls for this project

**Example**:

```bash
curl --location 'http://0.0.0.0:4000/project/new' \
--header 'Authorization: Bearer sk-1234' \
--header 'Content-Type: application/json' \
--data '{
    "project_alias": "hotel-recommendations",
    "team_id": "team-123",
    "models": ["claude-3-sonnet"],
    "max_budget": 200,
    "tpm_limit": 100000,
    "metadata": {
        "use_case_id": "SNOW-12346",
        "cost_center": "travel-products"
    }
}'
```

**Response**:

```json
{
    "project_id": "project-def",
    "project_alias": "hotel-recommendations",
    "team_id": "team-123",
    "models": ["claude-3-sonnet"],
    "spend": 0.0,
    "budget_id": "budget-xyz",
    "metadata": {
        "use_case_id": "SNOW-12346",
        "cost_center": "travel-products"
    },
    "created_at": "2025-01-15T10:00:00Z",
    "updated_at": "2025-01-15T10:00:00Z"
}
```

### POST /project/update

Update an existing project.

**Who can call**: Admins or Team Admins

**Parameters**:
- `project_id` (string, required): The project to update
- `project_alias` (string, optional): Updated project name
- `team_id` (string, optional): Move project to different team
- `models` (array, optional): Updated list of allowed models
- `max_budget` (float, optional): Updated budget
- `tpm_limit` (int, optional): Updated TPM limit
- `rpm_limit` (int, optional): Updated RPM limit
- `metadata` (object, optional): Updated metadata
- `blocked` (boolean, optional): Updated blocked status

**Example**:

```bash
curl --location 'http://0.0.0.0:4000/project/update' \
--header 'Authorization: Bearer sk-1234' \
--header 'Content-Type: application/json' \
--data '{
    "project_id": "project-abc",
    "max_budget": 200,
    "tpm_limit": 200000,
    "metadata": {
        "status": "production"
    }
}'
```

### GET /project/info

Get information about a specific project.

**Parameters**:
- `project_id` (string, required): Query parameter

**Example**:

```bash
curl --location 'http://0.0.0.0:4000/project/info?project_id=project-abc' \
--header 'Authorization: Bearer sk-1234'
```

**Response**:

```json
{
    "project_id": "project-abc",
    "project_alias": "flight-search-assistant",
    "team_id": "team-123",
    "models": ["gpt-4", "gpt-3.5-turbo"],
    "spend": 45.67,
    "model_spend": {
        "gpt-4": 42.30,
        "gpt-3.5-turbo": 3.37
    },
    "litellm_budget_table": {
        "budget_id": "budget-xyz",
        "max_budget": 100.0,
        "tpm_limit": 100000,
        "rpm_limit": 100
    },
    "metadata": {
        "use_case_id": "SNOW-12345"
    }
}
```

### GET /project/list

List all projects the user has access to.

**Example**:

```bash
curl --location 'http://0.0.0.0:4000/project/list' \
--header 'Authorization: Bearer sk-1234'
```

**Response**:

```json
[
    {
        "project_id": "project-abc",
        "project_alias": "flight-search-assistant",
        "team_id": "team-123",
        "spend": 45.67
    },
    {
        "project_id": "project-def",
        "project_alias": "hotel-recommendations",
        "team_id": "team-123",
        "spend": 23.45
    }
]
```

### DELETE /project/delete

Delete one or more projects.

**Who can call**: Admins only

**Parameters**:
- `project_ids` (array, required): List of project IDs to delete

**Example**:

```bash
curl --location --request DELETE 'http://0.0.0.0:4000/project/delete' \
--header 'Authorization: Bearer sk-1234' \
--header 'Content-Type: application/json' \
--data '{
    "project_ids": ["project-abc", "project-def"]
}'
```

**Note**: Projects with associated API keys cannot be deleted. Delete or reassign the keys first.

## Model-Specific Quotas

You can set different quotas for different models within a project:

```bash
curl --location 'http://0.0.0.0:4000/project/new' \
--header 'Authorization: Bearer sk-1234' \
--header 'Content-Type: application/json' \
--data '{
    "project_alias": "multi-model-project",
    "team_id": "team-123",
    "models": ["gpt-4", "gpt-3.5-turbo", "claude-3-sonnet"],
    "max_budget": 500,
    "metadata": {
        "model_tpm_limit": {
            "gpt-4": 50000,
            "gpt-3.5-turbo": 200000,
            "claude-3-sonnet": 100000
        },
        "model_rpm_limit": {
            "gpt-4": 50,
            "gpt-3.5-turbo": 500,
            "claude-3-sonnet": 100
        }
    }
}'
```
