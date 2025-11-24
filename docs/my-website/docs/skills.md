# /skills - Anthropic Skills API

| Feature | Supported | 
|---------|-----------|
| Cost Tracking | ✅ |
| Logging | ✅ |
| Load Balancing | ✅ |
| Supported Providers | `anthropic` |

:::tip

LiteLLM follows the [Anthropic Skills API](https://docs.anthropic.com/en/docs/build-with-claude/skills) for creating, managing, and using reusable AI capabilities.

:::

## **LiteLLM Python SDK Usage**

### Quick Start - Create a Skill

```python showLineNumbers title="create_skill.py"
from litellm import create_skill
import zipfile
import os

# Create a SKILL.md file
skill_content = """---
name: my-skill
description: A custom skill for data analysis
---

# My Skill

This skill helps with data analysis tasks.
"""

# Create skill directory and SKILL.md
os.makedirs("my-skill", exist_ok=True)
with open("my-skill/SKILL.md", "w") as f:
    f.write(skill_content)

# Create a zip file
with zipfile.ZipFile("my-skill.zip", "w") as zipf:
    zipf.write("my-skill/SKILL.md", "my-skill/SKILL.md")

# Create the skill
response = create_skill(
    model="anthropic/claude-3-5-sonnet-20241022",
    display_title="My Custom Skill",
    files=[open("my-skill.zip", "rb")]
)

print(f"Skill created: {response.id}")
```

### List Skills

```python showLineNumbers title="list_skills.py"
from litellm import list_skills

response = list_skills(
    model="anthropic/claude-3-5-sonnet-20241022",
    limit=20
)

for skill in response.data:
    print(f"{skill.display_title}: {skill.id}")
```

### Get Skill Details

```python showLineNumbers title="get_skill.py"
from litellm import get_skill

skill = get_skill(
    model="anthropic/claude-3-5-sonnet-20241022",
    skill_id="skill_01..."
)

print(f"Skill: {skill.display_title}")
print(f"Description: {skill.description}")
```

### Delete a Skill

```python showLineNumbers title="delete_skill.py"
from litellm import delete_skill

response = delete_skill(
    model="anthropic/claude-3-5-sonnet-20241022",
    skill_id="skill_01..."
)

print(f"Deleted: {response.id}")
```

### Async Usage

```python showLineNumbers title="async_skills.py"
from litellm import acreate_skill, alist_skills, aget_skill, adelete_skill
import asyncio

async def manage_skills():
    # Create skill
    with open("my-skill.zip", "rb") as f:
        skill = await acreate_skill(
            model="anthropic/claude-3-5-sonnet-20241022",
            display_title="My Async Skill",
            files=[f]
        )
    
    # List skills
    skills = await alist_skills(
        model="anthropic/claude-3-5-sonnet-20241022"
    )
    
    # Get skill
    skill_detail = await aget_skill(
        model="anthropic/claude-3-5-sonnet-20241022",
        skill_id=skill.id
    )
    
    # Delete skill (if no versions exist)
    # await adelete_skill(
    #     model="anthropic/claude-3-5-sonnet-20241022",
    #     skill_id=skill.id
    # )

asyncio.run(manage_skills())
```

## **LiteLLM Proxy Usage**

LiteLLM provides Anthropic-compatible `/skills` endpoints for managing skills.

**Setup**

Add this to your litellm proxy config.yaml

```yaml showLineNumbers title="config.yaml"
model_list:
  - model_name: claude-sonnet
    litellm_params:
      model: anthropic/claude-3-5-sonnet-20241022
      api_key: os.environ/ANTHROPIC_API_KEY
```

Start litellm

```bash
litellm --config /path/to/config.yaml

# RUNNING on http://0.0.0.0:4000
```

### Create Skill

```bash showLineNumbers title="create_skill.sh"
curl "http://0.0.0.0:4000/v1/skills?beta=true" \
  -X POST \
  -H "Authorization: Bearer sk-1234" \
  -H "anthropic-version: 2023-06-01" \
  -H "anthropic-beta: skills-2025-10-02" \
  -F "model=claude-sonnet" \
  -F "display_title=My Skill" \
  -F "files[]=@my-skill.zip"
```

### List Skills

```bash showLineNumbers title="list_skills.sh"
curl "http://0.0.0.0:4000/v1/skills?beta=true&model=claude-sonnet" \
  -H "Authorization: Bearer sk-1234" \
  -H "anthropic-version: 2023-06-01" \
  -H "anthropic-beta: skills-2025-10-02"
```

### Get Skill

```bash showLineNumbers title="get_skill.sh"
curl "http://0.0.0.0:4000/v1/skills/skill_01abc?beta=true&model=claude-sonnet" \
  -H "Authorization: Bearer sk-1234" \
  -H "anthropic-version: 2023-06-01" \
  -H "anthropic-beta: skills-2025-10-02"
```

### Delete Skill

```bash showLineNumbers title="delete_skill.sh"
curl "http://0.0.0.0:4000/v1/skills/skill_01abc?beta=true&model=claude-sonnet" \
  -X DELETE \
  -H "Authorization: Bearer sk-1234" \
  -H "anthropic-version: 2023-06-01" \
  -H "anthropic-beta: skills-2025-10-02"
```

### Model-Based Routing

Use multiple Anthropic accounts with model routing:

```yaml showLineNumbers title="config.yaml"
model_list:
  - model_name: claude-team-a
    litellm_params:
      model: anthropic/claude-3-5-sonnet-20241022
      api_key: os.environ/ANTHROPIC_API_KEY_TEAM_A
  
  - model_name: claude-team-b
    litellm_params:
      model: anthropic/claude-3-5-sonnet-20241022
      api_key: os.environ/ANTHROPIC_API_KEY_TEAM_B
```

Then route to specific accounts:

```bash showLineNumbers
# Route to Team A
curl "http://0.0.0.0:4000/v1/skills?beta=true" \
  -X POST \
  -H "Authorization: Bearer sk-1234" \
  -F "model=claude-team-a" \
  -F "display_title=Team A Skill" \
  -F "files[]=@skill.zip"

# Route to Team B
curl "http://0.0.0.0:4000/v1/skills?beta=true" \
  -X POST \
  -H "Authorization: Bearer sk-1234" \
  -F "model=claude-team-b" \
  -F "display_title=Team B Skill" \
  -F "files[]=@skill.zip"
```

## **SKILL.md Format**

Skills require a `SKILL.md` file with YAML frontmatter:

```markdown showLineNumbers title="SKILL.md"
---
name: my-skill
description: A brief description of what this skill does
license: MIT
allowed-tools:
  - computer_20250124
  - text_editor_20250124
---

# My Skill

Detailed instructions for Claude on how to use this skill.

## Usage

Examples and best practices...
```

### YAML Frontmatter Requirements

| Field | Required | Description |
|-------|----------|-------------|
| `name` | Yes | Skill identifier (lowercase, numbers, hyphens only) |
| `description` | Yes | Brief description of the skill |
| `license` | No | License type (e.g., MIT, Apache-2.0) |
| `allowed-tools` | No | List of Claude tools this skill can use |
| `metadata` | No | Additional custom metadata |

### File Structure

Skills must be packaged as a ZIP file with this structure:

```
my-skill.zip
└── my-skill/           # Top-level folder (name must match skill name)
    └── SKILL.md        # Required skill definition
```

**Important:** The folder name inside the ZIP must match the `name` in SKILL.md frontmatter.

## **Response Format**

### Skill Object

```json showLineNumbers
{
  "id": "skill_01abc123",
  "type": "skill",
  "name": "my-skill",
  "display_title": "My Custom Skill",
  "description": "A brief description",
  "created_at": "2025-01-15T10:30:00.000Z",
  "updated_at": "2025-01-15T10:30:00.000Z",
  "latest_version_id": "skillver_01xyz789"
}
```

### List Skills Response

```json showLineNumbers
{
  "data": [
    {
      "id": "skill_01abc",
      "type": "skill",
      "name": "skill-one",
      "display_title": "Skill One",
      "description": "First skill"
    },
    {
      "id": "skill_02def",
      "type": "skill",
      "name": "skill-two",
      "display_title": "Skill Two",
      "description": "Second skill"
    }
  ],
  "has_more": false,
  "first_id": "skill_01abc",
  "last_id": "skill_02def"
}
```

## **API Limitations**

:::warning

- **Deletion Restriction:** Skills cannot be deleted if they have existing versions. You must delete all versions first.
- **Unique Titles:** Each skill must have a unique `display_title` within your organization.
- **File Format:** Only ZIP files are supported for skill uploads.
- **Beta Feature:** The Skills API is in beta and requires the `anthropic-beta: skills-2025-10-02` header.

:::

## **Supported Providers**

| Provider | Link to Usage |
|----------|---------------|
| Anthropic | [Usage](#quick-start---create-a-skill) |

