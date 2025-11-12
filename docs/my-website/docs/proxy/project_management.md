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

### Example: Team with Multiple Use Cases

```
Organization: Acme Corp
└── Team A [no models] → Uses SCIM for onboarding team members
    ├── Project: Flight Search Use Case [gpt-4, gpt-3.5-turbo]
    │   ├── API Key 1 (Production)
    │   └── API Key 2 (Staging)
    ├── Project: Hotel Recommendations [claude-3-sonnet, gpt-4]
    │   └── API Key 3
    └── Project: Customer Support Bot [gpt-4, claude-3-haiku]
        ├── API Key 4
        └── API Key 5
```

**Key Points:**
- Teams have no models assigned (model access is controlled at the project level)
- Each project (use case) has specific models it can access
- Projects can have different budgets and rate limits per model
- Multiple API keys can belong to the same project

## Use Cases

- **Use Case Management**: Create separate projects for different applications or use cases within a team
- **Budget Control**: Set model-specific quotas and budgets per project
- **Spend Tracking**: Track costs and usage at the project level for reporting
- **Access Control**: Control which models and resources each project can access

## Quick Start

### 1. Create a Project

```bash
curl --location 'http://0.0.0.0:4000/project/new' \
--header 'Authorization: Bearer sk-1234' \
--header 'Content-Type: application/json' \
--data '{
    "project_alias": "flight-search-assistant",
    "team_id": "team-123",
    "models": ["gpt-4", "gpt-3.5-turbo"],
    "max_budget": 100,
    "metadata": {
        "use_case_id": "SNOW-12345",
        "responsible_ai_id": "RAI-67890"
    }
}'
```

### 2. Generate API Key for Project

```bash
curl --location 'http://0.0.0.0:4000/key/generate' \
--header 'Authorization: Bearer sk-1234' \
--header 'Content-Type: application/json' \
--data '{
    "project_id": "project-abc",
    "key_alias": "flight-search-prod-key",
    "models": ["gpt-4", "gpt-3.5-turbo"],
    "max_budget": 50
}'
```

### 3. Make API Calls

```bash
curl --location 'http://0.0.0.0:4000/chat/completions' \
--header 'Authorization: Bearer sk-proj-xyz-abc123...' \
--header 'Content-Type: application/json' \
--data '{
    "model": "gpt-4",
    "messages": [{"role": "user", "content": "Find flights to Paris"}]
}'
```

The request will be tracked under the project for spend reporting.

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

## Spend Tracking

Project spend is automatically tracked in the SpendLogs table with the `project_id` stored in the metadata field. You can view project-level spending in the LiteLLM UI or query it directly:

```sql
SELECT 
    metadata->>'project_id' as project_id,
    SUM(spend) as total_spend,
    COUNT(*) as request_count
FROM "LiteLLM_SpendLogs"
WHERE metadata->>'project_id' = 'project-abc'
GROUP BY metadata->>'project_id';
```

## Best Practices

1. **Use Meaningful Names**: Use descriptive `project_alias` values that clearly identify the use case
2. **Set Budgets**: Always set `max_budget` to prevent runaway costs
3. **Use Metadata**: Store important information like use case IDs, cost centers, and approvals in metadata
4. **Monitor Spend**: Regularly check project spend using `/project/info`
5. **Separate Environments**: Create separate projects for dev, staging, and production
6. **Model Restrictions**: Only grant access to models that are actually needed for the use case

## Example: Expedia Use Case

This example shows how to implement a use-case approval workflow similar to Expedia's requirements:

```bash
# 1. Employee gets approved use-case ID from ServiceNow (e.g., SNOW-12345)

# 2. Create project with mandatory metadata
curl --location 'http://0.0.0.0:4000/project/new' \
--header 'Authorization: Bearer sk-1234' \
--header 'Content-Type: application/json' \
--data '{
    "project_alias": "travel-recommendation-engine",
    "team_id": "team-travel-products",
    "models": ["gpt-4", "claude-3-sonnet"],
    "max_budget": 1000,
    "metadata": {
        "use_case_id": "SNOW-12345",
        "responsible_ai_id": "RAI-67890",
        "cost_center": "travel-products",
        "approved_by": "jane.smith@company.com",
        "approval_date": "2025-01-15"
    }
}'

# 3. Set model-specific quotas
curl --location 'http://0.0.0.0:4000/project/update' \
--header 'Authorization: Bearer sk-1234' \
--header 'Content-Type: application/json' \
--data '{
    "project_id": "project-abc",
    "metadata": {
        "model_tpm_limit": {
            "gpt-4": 100000,
            "claude-3-sonnet": 200000
        }
    }
}'

# 4. Generate API keys for the project
curl --location 'http://0.0.0.0:4000/key/generate' \
--header 'Authorization: Bearer sk-1234' \
--header 'Content-Type: application/json' \
--data '{
    "project_id": "project-abc",
    "key_alias": "travel-rec-prod",
    "models": ["gpt-4", "claude-3-sonnet"]
}'
```

## Permissions

- **Proxy Admins**: Can create, update, delete, and view all projects
- **Team Admins**: Can create, update, and view projects for their teams
- **Team Members**: Can view projects for their teams
- **Other Users**: No access to projects outside their teams

