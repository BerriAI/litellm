# Anthropic Skills API Integration

This module provides comprehensive support for the Anthropic Skills API through LiteLLM.

## Features

The Skills API allows you to:
- **Create skills**: Define reusable AI capabilities
- **List skills**: Browse all available skills
- **Get skills**: Retrieve detailed information about a specific skill
- **Delete skills**: Remove skills that are no longer needed

## Quick Start

### Prerequisites

Set your Anthropic API key:
```python
import os
os.environ["ANTHROPIC_API_KEY"] = "your-api-key-here"
```

### Basic Usage

#### Create a Skill

```python
import litellm

# Create a skill with files
# Note: All files must be in the same top-level directory
# and must include a SKILL.md file at the root
skill = litellm.create_skill(
    files=[
        # List of file objects to upload
        # Must include SKILL.md
    ],
    display_title="Python Code Generator",
    custom_llm_provider="anthropic"
)
print(f"Created skill: {skill.id}")

# Asynchronous version
skill = await litellm.acreate_skill(
    files=[...],  # Your files here
    display_title="Python Code Generator",
    custom_llm_provider="anthropic"
)
```

#### List Skills

```python
# List all skills
skills = litellm.list_skills(
    custom_llm_provider="anthropic"
)

for skill in skills.data:
    print(f"{skill.display_title}: {skill.id}")

# With pagination and filtering
skills = litellm.list_skills(
    limit=20,
    source="custom",  # Filter by 'custom' or 'anthropic'
    custom_llm_provider="anthropic"
)

# Get next page if available
if skills.has_more:
    next_page = litellm.list_skills(
        page=skills.next_page,
        custom_llm_provider="anthropic"
    )
```

#### Get a Skill

```python
skill = litellm.get_skill(
    skill_id="skill_abc123",
    custom_llm_provider="anthropic"
)

print(f"Skill: {skill.display_title}")
print(f"Created: {skill.created_at}")
print(f"Latest version: {skill.latest_version}")
print(f"Source: {skill.source}")
```

#### Delete a Skill

```python
result = litellm.delete_skill(
    skill_id="skill_abc123",
    custom_llm_provider="anthropic"
)

print(f"Deleted skill {result.id}, type: {result.type}")
```

## API Reference

### `create_skill()`

Create a new skill.

**Parameters:**
- `files` (List[Any], optional): Files to upload for the skill. All files must be in the same top-level directory and must include a SKILL.md file at the root.
- `display_title` (str, optional): Display title for the skill
- `custom_llm_provider` (str, optional): Provider name (default: "anthropic")
- `extra_headers` (dict, optional): Additional HTTP headers
- `timeout` (float, optional): Request timeout

**Returns:**
- `Skill`: The created skill object

**Async version:** `acreate_skill()`

### `list_skills()`

List all skills.

**Parameters:**
- `limit` (int, optional): Number of results to return per page (max 100, default 20)
- `page` (str, optional): Pagination token for fetching a specific page of results
- `source` (str, optional): Filter skills by source ('custom' or 'anthropic')
- `custom_llm_provider` (str, optional): Provider name (default: "anthropic")
- `extra_headers` (dict, optional): Additional HTTP headers
- `timeout` (float, optional): Request timeout

**Returns:**
- `ListSkillsResponse`: Object containing a list of skills and pagination info

**Async version:** `alist_skills()`

### `get_skill()`

Get a specific skill by ID.

**Parameters:**
- `skill_id` (str, required): The skill ID
- `custom_llm_provider` (str, optional): Provider name (default: "anthropic")
- `extra_headers` (dict, optional): Additional HTTP headers
- `timeout` (float, optional): Request timeout

**Returns:**
- `Skill`: The requested skill object

**Async version:** `aget_skill()`

### `delete_skill()`

Delete a skill.

**Parameters:**
- `skill_id` (str, required): The skill ID to delete
- `custom_llm_provider` (str, optional): Provider name (default: "anthropic")
- `extra_headers` (dict, optional): Additional HTTP headers
- `timeout` (float, optional): Request timeout

**Returns:**
- `DeleteSkillResponse`: Object with `id` and `type` fields

**Async version:** `adelete_skill()`

## Response Types

### `Skill`

Represents a skill from the Anthropic Skills API.

**Fields:**
- `id` (str): Unique identifier
- `created_at` (str): ISO 8601 timestamp
- `display_title` (str, optional): Display title
- `latest_version` (str, optional): Latest version identifier
- `source` (str): Source ("custom" or "anthropic")
- `type` (str): Object type (always "skill")
- `updated_at` (str): ISO 8601 timestamp

### `ListSkillsResponse`

Response from listing skills.

**Fields:**
- `data` (List[Skill]): List of skills
- `next_page` (str, optional): Pagination token for the next page
- `has_more` (bool): Whether more skills are available

### `DeleteSkillResponse`

Response from deleting a skill.

**Fields:**
- `id` (str): The deleted skill ID
- `type` (str): Deleted object type (always "skill_deleted")

## Architecture

The Skills API implementation follows LiteLLM's standard patterns:

1. **Type Definitions** (`litellm/types/llms/anthropic_skills.py`)
   - Pydantic models for request/response types
   - TypedDict definitions for request parameters

2. **Base Configuration** (`litellm/llms/base_llm/skills/transformation.py`)
   - Abstract base class `BaseSkillsAPIConfig`
   - Defines transformation interface for provider-specific implementations

3. **Provider Implementation** (`litellm/llms/anthropic/skills/transformation.py`)
   - `AnthropicSkillsConfig` - Anthropic-specific transformations
   - Handles API authentication, URL construction, and response mapping

4. **Main Handler** (`litellm/skills/main.py`)
   - Public API functions (sync and async)
   - Request validation and routing
   - Error handling

5. **HTTP Handlers** (`litellm/llms/custom_httpx/llm_http_handler.py`)
   - Low-level HTTP request/response handling
   - Connection pooling and retry logic

## Beta API Support

The Skills API is in beta. The beta header (`skills-2025-10-02`) is automatically added by the Anthropic provider configuration. You can customize it if needed:

```python
skill = litellm.create_skill(
    display_title="My Skill",
    extra_headers={
        "anthropic-beta": "skills-2025-10-02"  # Or any other beta version
    },
    custom_llm_provider="anthropic"
)
```

The default beta version is configured in `litellm.constants.ANTHROPIC_SKILLS_API_BETA_VERSION`.

## Error Handling

All Skills API functions follow LiteLLM's standard error handling:

```python
import litellm

try:
    skill = litellm.create_skill(
        display_title="My Skill",
        custom_llm_provider="anthropic"
    )
except litellm.exceptions.AuthenticationError as e:
    print(f"Authentication failed: {e}")
except litellm.exceptions.RateLimitError as e:
    print(f"Rate limit exceeded: {e}")
except litellm.exceptions.APIError as e:
    print(f"API error: {e}")
```

## Contributing

To add support for Skills API to a new provider:

1. Create provider-specific configuration class inheriting from `BaseSkillsAPIConfig`
2. Implement all abstract methods for request/response transformations
3. Register the config in `ProviderConfigManager.get_provider_skills_api_config()`
4. Add appropriate tests

## Related Documentation

- [Anthropic Skills API Documentation](https://platform.claude.com/docs/en/api/beta/skills/create)
- [LiteLLM Responses API](../../../responses/)
- [Provider Configuration System](../../base_llm/)

## Support

For issues or questions:
- GitHub Issues: https://github.com/BerriAI/litellm/issues
- Discord: https://discord.gg/wuPM9dRgDw
