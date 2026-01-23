import Image from '@theme/IdealImage';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# OpenCode Quickstart

This tutorial shows how to connect OpenCode to your existing LiteLLM instance and switch between models.

:::info 

This integration allows you to use any LiteLLM supported model through OpenCode with centralized authentication, usage tracking, and cost controls.

:::

<br />

### Video Walkthrough

<iframe width="840" height="500" src="https://www.loom.com/embed/00791498f1d84e4ba6d7476bd2e1442f" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen></iframe>

## Prerequisites

- LiteLLM already configured and running (e.g., http://localhost:4000)
- LiteLLM API key

## Installation

### Step 1: Install OpenCode

Choose your preferred installation method:

<Tabs>
<TabItem value="curl" label="One-line install (recommended)">

```bash
curl -fsSL https://opencode.ai/install | bash
```

</TabItem>
<TabItem value="npm" label="NPM">

```bash
npm install -g opencode-ai
```

</TabItem>
<TabItem value="homebrew" label="Homebrew">

```bash
brew install sst/tap/opencode
```

</TabItem>
</Tabs>

Verify installation:

```bash
opencode --version
```

### Step 2: Configure LiteLLM Provider

Create your OpenCode configuration file. You can place this in different locations depending on your needs:

**Configuration locations:**
- **Global**: `~/.config/opencode/opencode.json` (applies to all projects)
- **Project**: `opencode.json` in your project root (project-specific settings)
- **Custom**: Set `OPENCODE_CONFIG` environment variable

Create `~/.config/opencode/opencode.json` (global config):

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "litellm": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "LiteLLM",
      "options": {
        "baseURL": "http://localhost:4000/v1"
      },
      "models": {
        "gpt-4": {
          "name": "GPT-4"
        },
        "claude-3-5-sonnet-20241022": {
          "name": "Claude 3.5 Sonnet"
        },
        "deepseek-chat": {
          "name": "DeepSeek Chat"
        }
      }
    }
  }
}
```

:::tip
The keys in the "models" object (e.g., "gpt-4", "claude-3-5-sonnet-20241022") should match the `model_name` values from your LiteLLM configuration. The "name" field provides a friendly display name that will appear as an alias in OpenCode.
:::

### Step 3: Connect to LiteLLM Provider

Launch OpenCode:

```bash
opencode
```

Add your API key:

```bash
/connect
```

Then:
- **Enter provider name**: `LiteLLM` (must match the "name" field in your config)
- **Enter your LiteLLM API key**: Your LiteLLM master key or virtual key

### Step 4: Switch Between Models

In OpenCode, run:

```bash
/models
```

Select any model from your LiteLLM configuration. OpenCode will route all requests through your LiteLLM instance.

## Advanced Configuration

### Model Parameters

You can customize model parameters like context limits:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "litellm": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "LiteLLM",
      "options": {
        "baseURL": "http://localhost:4000/v1"
      },
      "models": {
        "gpt-4": {
          "name": "GPT-4",
          "limit": {
            "context": 128000,
            "output": 4096
          }
        },
        "claude-3-5-sonnet-20241022": {
          "name": "Claude 3.5 Sonnet",
          "limit": {
            "context": 200000,
            "output": 8192
          }
        }
      }
    }
  }
}
```

### Multi-Provider Setup

You can configure multiple LiteLLM instances or mix with other providers:

<Tabs>
<TabItem value="multi-litellm" label="Multiple LiteLLM Instances">

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "litellm-prod": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "LiteLLM Production",
      "options": {
        "baseURL": "https://your-prod-instance.com/v1"
      },
      "models": {
        "gpt-4": {
          "name": "GPT-4 (Production)"
        }
      }
    },
    "litellm-dev": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "LiteLLM Development",
      "options": {
        "baseURL": "http://localhost:4000/v1"
      },
      "models": {
        "gpt-4": {
          "name": "GPT-4 (Development)"
        }
      }
    }
  }
}
```

</TabItem>
<TabItem value="mixed-providers" label="Mixed Providers">

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "litellm": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "LiteLLM",
      "options": {
        "baseURL": "http://localhost:4000/v1"
      },
      "models": {
        "gpt-4": {
          "name": "GPT-4 via LiteLLM"
        },
        "claude-3-5-sonnet-20241022": {
          "name": "Claude 3.5 Sonnet via LiteLLM"
        }
      }
    },
    "openai": {
      "npm": "@ai-sdk/openai",
      "name": "OpenAI Direct",
      "models": {
        "gpt-4o": {
          "name": "GPT-4o (Direct)"
        }
      }
    }
  }
}
```

</TabItem>
</Tabs>

## Example LiteLLM Configuration

Here's an example LiteLLM `config.yaml` that works well with OpenCode:

```yaml
model_list:
  # OpenAI models
  - model_name: gpt-4
    litellm_params:
      model: openai/gpt-4
      api_key: os.environ/OPENAI_API_KEY
  
  - model_name: gpt-4o
    litellm_params:
      model: openai/gpt-4o
      api_key: os.environ/OPENAI_API_KEY

  # Anthropic models
  - model_name: claude-3-5-sonnet-20241022
    litellm_params:
      model: anthropic/claude-3-5-sonnet-20241022
      api_key: os.environ/ANTHROPIC_API_KEY
  
  # DeepSeek models
  - model_name: deepseek-chat
    litellm_params:
      model: deepseek/deepseek-chat
      api_key: os.environ/DEEPSEEK_API_KEY
```

## Troubleshooting

**OpenCode not connecting:**
- Verify your LiteLLM proxy is running: `curl http://localhost:4000/health`
- Check that the `baseURL` in your OpenCode config matches your LiteLLM instance
- Ensure the provider name in `/connect` matches exactly with your config

**Authentication errors:**
- Verify your LiteLLM API key is correct
- Check that your LiteLLM instance has authentication properly configured
- Ensure your API key has access to the models you're trying to use

**Model not found:**
- Ensure the model names in OpenCode config match your LiteLLM `model_name` values
- Check LiteLLM logs for detailed error messages
- Verify the models are properly configured in your LiteLLM instance

**Configuration not loading:**
- Check the config file path and permissions
- Validate JSON syntax using a JSON validator
- Ensure the `$schema` URL is accessible

## Tips

- Add more models to the config as needed - they'll appear in `/models`
- Use project-specific configs for different codebases with different model requirements
- Monitor your LiteLLM proxy logs to see OpenCode requests in real-time
