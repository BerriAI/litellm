import Image from '@theme/IdealImage';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Claude Code Plugin Marketplace (Managed Skills)

LiteLLM AI Gateway acts as a central registry for Claude Code plugins. Admins can govern which plugins are available across the organization, and engineers can discover and install approved plugins from a single source.

## Prerequisites

- LiteLLM Proxy running with database connected
- Admin access to LiteLLM UI
- Plugins hosted on GitHub, GitLab, or any git-accessible URL

## Admin Guide: Managing the Marketplace

### Step 1: Navigate to Claude Code Plugins

In the LiteLLM Admin UI, click on **Claude Code Plugins** in the left navigation menu.

<Image img={require('../../img/claude_code_marketplace/step1_navigate_plugins.jpeg')} style={{ width: '800px', height: 'auto' }} />

### Step 2: View the Plugins List

You'll see the list of all registered plugins. From here you can add, enable, disable, or delete plugins.

<Image img={require('../../img/claude_code_marketplace/step3_plugins_list.jpeg')} style={{ width: '800px', height: 'auto' }} />

### Step 3: Add a New Plugin

Click **+ Add New Plugin** to register a plugin in your marketplace.

<Image img={require('../../img/claude_code_marketplace/step4_add_plugin.jpeg')} style={{ width: '800px', height: 'auto' }} />

### Step 4: Fill in Plugin Details

Enter the plugin information:

- **Name**: Plugin identifier (kebab-case, e.g., `my-plugin`)
- **Source Type**: Choose GitHub or URL
- **Repository/URL**: The git source (e.g., `org/repo` for GitHub)
- **Version**: Semantic version (optional)
- **Description**: What the plugin does
- **Category**: Plugin category for organization
- **Keywords**: Search terms

<Image img={require('../../img/claude_code_marketplace/step5_plugin_form.jpeg')} style={{ width: '800px', height: 'auto' }} />

### Step 5: Submit the Plugin

After filling in the details, click **Add Plugin** to register it.

<Image img={require('../../img/claude_code_marketplace/step9_submit.jpeg')} style={{ width: '800px', height: 'auto' }} />

### Step 6: Enable/Disable Plugins

Toggle plugins on or off to control what appears in the public marketplace. Only **enabled** plugins are visible to engineers.

<Image img={require('../../img/claude_code_marketplace/step11_enable_plugin.jpeg')} style={{ width: '800px', height: 'auto' }} />

## Engineer Guide: Installing Plugins

### Step 1: Add the LiteLLM Marketplace

Add your company's LiteLLM marketplace to Claude Code:

```bash
claude plugin marketplace add http://your-litellm-proxy:4000/claude-code/marketplace.json
```

<Image img={require('../../img/claude_code_marketplace/step12_cli_marketplace.jpeg')} style={{ width: '800px', height: 'auto' }} />

### Step 2: Browse Available Plugins

List all available plugins from the marketplace:

```bash
claude plugin search @litellm
```

### Step 3: Install a Plugin

Install any plugin from the marketplace:

```bash
claude plugin install my-plugin@litellm
```

<Image img={require('../../img/claude_code_marketplace/step15_cli_paste.jpeg')} style={{ width: '800px', height: 'auto' }} />

### Step 4: Verify Installation

The plugin is now installed and ready to use:

<Image img={require('../../img/claude_code_marketplace/step16_cli_complete.jpeg')} style={{ width: '800px', height: 'auto' }} />

## API Reference

### Public Endpoint (No Auth Required)

#### GET `/claude-code/marketplace.json`

Returns the marketplace catalog for Claude Code discovery.

```bash
curl http://localhost:4000/claude-code/marketplace.json
```

**Response:**
```json
{
  "name": "litellm",
  "owner": {
    "name": "LiteLLM",
    "email": "support@litellm.ai"
  },
  "plugins": [
    {
      "name": "my-plugin",
      "source": {
        "source": "github",
        "repo": "org/my-plugin"
      },
      "version": "1.0.0",
      "description": "My awesome plugin",
      "category": "productivity",
      "keywords": ["automation", "tools"]
    }
  ]
}
```

### Admin Endpoints (Auth Required)

#### POST `/claude-code/plugins`

Register a new plugin.

```bash
curl -X POST http://localhost:4000/claude-code/plugins \
  -H "Authorization: Bearer sk-..." \
  -H "Content-Type: application/json" \
  -d '{
    "name": "my-plugin",
    "source": {"source": "github", "repo": "org/my-plugin"},
    "version": "1.0.0",
    "description": "My awesome plugin",
    "category": "productivity",
    "keywords": ["automation", "tools"]
  }'
```

#### GET `/claude-code/plugins`

List all registered plugins.

```bash
curl http://localhost:4000/claude-code/plugins \
  -H "Authorization: Bearer sk-..."
```

#### POST `/claude-code/plugins/{name}/enable`

Enable a plugin.

```bash
curl -X POST http://localhost:4000/claude-code/plugins/my-plugin/enable \
  -H "Authorization: Bearer sk-..."
```

#### POST `/claude-code/plugins/{name}/disable`

Disable a plugin.

```bash
curl -X POST http://localhost:4000/claude-code/plugins/my-plugin/disable \
  -H "Authorization: Bearer sk-..."
```

#### DELETE `/claude-code/plugins/{name}`

Delete a plugin.

```bash
curl -X DELETE http://localhost:4000/claude-code/plugins/my-plugin \
  -H "Authorization: Bearer sk-..."
```

## Plugin Source Formats

<Tabs>
<TabItem value="github" label="GitHub">

```json
{
  "name": "my-plugin",
  "source": {
    "source": "github",
    "repo": "organization/repository"
  }
}
```

</TabItem>
<TabItem value="url" label="Git URL">

```json
{
  "name": "my-plugin",
  "source": {
    "source": "url",
    "url": "https://github.com/org/repo.git"
  }
}
```

Use this format for GitLab, Bitbucket, or self-hosted git repositories.

</TabItem>
</Tabs>

## Example: Setting Up an Internal Plugin Marketplace

### 1. Create Internal Plugins

Structure your plugin repository:

```
my-company-plugin/
├── plugin.json          # Plugin manifest
├── SKILL.md            # Main skill file
├── skills/             # Additional skills
│   └── helper.md
└── README.md
```

### 2. Register Plugins via API

```bash
# Register your internal tools plugin
curl -X POST http://localhost:4000/claude-code/plugins \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "internal-tools",
    "source": {"source": "github", "repo": "mycompany/internal-tools"},
    "version": "1.0.0",
    "description": "Internal development tools and utilities",
    "author": {"name": "Platform Team", "email": "platform@mycompany.com"},
    "category": "internal",
    "keywords": ["internal", "tools", "utilities"]
  }'
```

### 3. Use in Claude Code

Send engineers the marketplace URL:

```bash
# One-time setup for each engineer
claude plugin marketplace add http://litellm.internal.company.com/claude-code/marketplace.json

# Install company plugins
claude plugin install internal-tools@litellm
```

## Troubleshooting

**Plugin not appearing in marketplace:**
- Verify the plugin is **enabled** in the admin UI
- Check that the plugin has a valid `source` field

**Installation fails:**
- Ensure the git repository is accessible from the engineer's machine
- For private repos, engineers need appropriate git credentials configured

**Database errors:**
- Verify LiteLLM proxy is connected to the database
- Check proxy logs for detailed error messages
