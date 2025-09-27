# LiteLLM Release Notes Generation Instructions

This document provides comprehensive instructions for AI agents to generate release notes for LiteLLM following the established format and style.

## Required Inputs

1. **Release Version** (e.g., `v1.77.3-stable`)
2. **PR Diff/Changelog** - List of PRs with titles and contributors
3. **Previous Version Commit Hash** - To compare model pricing changes
4. **Reference Release Notes** - Use recent stable releases (v1.76.3-stable, v1.77.2-stable) as templates for consistent formatting

## Step-by-Step Process

### 1. Initial Setup and Analysis

```bash
# Check git diff for model pricing changes
git diff <previous_commit_hash> HEAD -- model_prices_and_context_window.json
```

**Key Analysis Points:**
- New models added (look for new entries)
- Deprecated models removed (look for deleted entries)
- Pricing updates (look for cost changes)
- Feature support changes (tool calling, reasoning, etc.)

### 2. Release Notes Structure

Follow this exact structure based on recent stable releases (v1.76.3-stable, v1.77.2-stable):

```markdown
---
title: "v1.77.X-stable - [Key Theme]"
slug: "v1-77-X"
date: YYYY-MM-DDTHH:mm:ss
authors: [standard author block]
hide_table_of_contents: false
---

## Deploy this version
[Docker and pip installation tabs]

## Key Highlights
[3-5 bullet points of major features]

## New Models / Updated Models
#### New Model Support
[Model pricing table]

#### Features
[Provider-specific features organized by provider]

### Bug Fixes
[Provider-specific bug fixes organized by provider]

#### New Provider Support
[New provider integrations]

## LLM API Endpoints
#### Features
[API-specific features organized by API type]

#### Bugs
[General bug fixes]

## Management Endpoints / UI
#### Features
[UI and management features]

#### Bugs
[Management-related bug fixes]

## Logging / Guardrail Integrations
#### Features
[Organized by integration provider with proper doc links]

#### Guardrails
[Guardrail-specific features and fixes]

#### New Integration
[Major new integrations]

## Performance / Loadbalancing / Reliability improvements
[Infrastructure improvements]

## General Proxy Improvements
[Other proxy-related changes]

## New Contributors
[List of first-time contributors]

## Full Changelog
[Link to GitHub comparison]
```

### 3. Categorization Rules

**Performance Improvements:**
- RPS improvements
- Memory optimizations
- CPU usage optimizations
- Timeout controls
- Worker configuration

**New Models/Updated Models:**
- Extract from model_prices_and_context_window.json diff
- Create tables with: Provider, Model, Context Window, Input Cost, Output Cost, Features
- **Structure:**
  - `#### New Model Support` - pricing table
  - `#### Features` - organized by provider with documentation links
  - `### Bug Fixes` - provider-specific bug fixes
  - `#### New Provider Support` - major new provider integrations
- Group by provider with proper doc links: `**[Provider Name](../../docs/providers/[provider])**`
- Use bullet points under each provider for multiple features
- Separate features from bug fixes clearly

**LLM API Endpoints:**
- **Structure:**
  - `#### Features` - organized by API type (Responses API, Batch API, etc.)
  - `#### Bugs` - general bug fixes under **General** category
- **API Categories:**
  - Responses API
  - Batch API  
  - CountTokens API
  - Images API
  - Video Generation (if applicable)
  - General (miscellaneous improvements)
- Use proper documentation links for each API type

**UI/Management:**
- Authentication changes
- Dashboard improvements
- Team management
- Key management

**Logging / Guardrail Integrations:**
- **Structure:**
  - `#### Features` - organized by integration provider with proper doc links
  - `#### Guardrails` - guardrail-specific features and fixes
  - `#### New Integration` - major new integrations
- **Integration Categories:**
  - **[DataDog](../../docs/proxy/logging#datadog)** - group all DataDog-related changes
  - **[Langfuse](../../docs/proxy/logging#langfuse)** - Langfuse-specific features
  - **[Prometheus](../../docs/proxy/logging#prometheus)** - monitoring improvements
  - **[PostHog](../../docs/observability/posthog)** - observability integration
  - Other logging providers with proper doc links
- Use bullet points under each provider for multiple features
- Separate logging features from guardrails clearly

### 4. Documentation Linking Strategy

**Link to docs when:**
- New provider support added
- Significant feature additions
- API endpoint changes
- Integration additions

**Link format:** `../../docs/[category]/[specific_doc]`

**Common doc paths:**
- `../../docs/providers/[provider]` - Provider-specific docs
- `../../docs/image_generation` - Image generation
- `../../docs/video_generation` - Video generation (if exists)
- `../../docs/response_api` - Responses API
- `../../docs/proxy/logging` - Logging integrations
- `../../docs/proxy/guardrails` - Guardrails
- `../../docs/pass_through/[provider]` - Passthrough endpoints

### 5. Model Table Generation

From git diff analysis, create tables like:

```markdown
| Provider | Model | Context Window | Input ($/1M tokens) | Output ($/1M tokens) | Features |
| -------- | ----- | -------------- | ------------------- | -------------------- | -------- |
| OpenRouter | `openrouter/openai/gpt-4.1` | 1M | $2.00 | $8.00 | Chat completions with vision |
```

**Extract from JSON:**
- `max_input_tokens` → Context Window
- `input_cost_per_token` × 1,000,000 → Input cost
- `output_cost_per_token` × 1,000,000 → Output cost
- `supports_*` fields → Features
- Special pricing fields (per image, per second) for generation models

### 6. PR Categorization Logic

**By Keywords in PR Title:**
- `[Perf]`, `Performance`, `RPS` → Performance Improvements
- `[Bug]`, `[Bug Fix]`, `Fix` → Bug Fixes section
- `[Feat]`, `[Feature]`, `Add support` → Features section
- `[Docs]` → Documentation (usually exclude from main sections)
- Provider names (Gemini, OpenAI, etc.) → Group under provider

**By PR Content Analysis:**
- New model additions → New Models section
- UI changes → Management Endpoints/UI
- Logging/observability → Logging/Guardrail Integrations
- Rate limiting/budgets → Performance/Reliability
- Authentication → Management Endpoints

### 7. Writing Style Guidelines

**Tone:**
- Professional but accessible
- Focus on user impact
- Highlight breaking changes clearly
- Use active voice

**Formatting:**
- Use consistent markdown formatting
- Include PR links: `[PR #XXXXX](https://github.com/BerriAI/litellm/pull/XXXXX)`
- Use code blocks for configuration examples
- Bold important terms and section headers

**Warnings/Notes:**
- Add warning boxes for breaking changes
- Include migration instructions when needed
- Provide override options for default changes

### 8. Quality Checks

**Before finalizing:**
- Verify all PR links work
- Check documentation links are valid
- Ensure model pricing is accurate
- Confirm provider names are consistent
- Review for typos and formatting issues

### 9. Common Patterns to Follow

**Performance Changes:**
```markdown
- **+400 RPS Performance Boost** - Description - [PR #XXXXX](link)
```

**New Models:**
Always include pricing table and feature highlights

**Breaking Changes:**
```markdown
:::warning
This release has a known issue...
:::
```

**Provider Features (New Models / Updated Models section):**
```markdown
#### Features

- **[Provider Name](../../docs/providers/provider)**
    - Feature description - [PR #XXXXX](link)
    - Another feature description - [PR #YYYYY](link)
```

**API Features (LLM API Endpoints section):**
```markdown
#### Features

- **[API Name](../../docs/api_path)**
    - Feature description - [PR #XXXXX](link)
    - Another feature - [PR #YYYYY](link)
- **General**
    - Miscellaneous improvements - [PR #ZZZZZ](link)
```

**Integration Features (Logging / Guardrail Integrations section):**
```markdown
#### Features

- **[Integration Name](../../docs/proxy/logging#integration)**
    - Feature description - [PR #XXXXX](link)
    - Bug fix description - [PR #YYYYY](link)
```

**Bug Fixes Pattern:**
```markdown
### Bug Fixes

- **[Provider/Component Name](../../docs/providers/provider)**
    - Bug fix description - [PR #XXXXX](link)
```

### 10. Missing Documentation Check

**Review for missing docs:**
- New providers without documentation
- New API endpoints without examples
- Complex features without guides
- Integration setup instructions

**Flag for documentation needs:**
- New provider integrations
- Significant API changes
- Complex configuration options
- Migration requirements

## Example Command Workflow

```bash
# 1. Get model changes
git diff <commit> HEAD -- model_prices_and_context_window.json

# 2. Analyze PR list for categorization
# 3. Create release notes following template
# 4. Link to appropriate documentation
# 5. Review for missing documentation needs
```

## Output Requirements

- Follow exact markdown structure from reference
- Include all PR links and contributors
- Provide accurate model pricing tables
- Link to relevant documentation
- Highlight breaking changes with warnings
- Include deployment instructions
- End with full changelog link

This process ensures consistent, comprehensive release notes that help users understand changes and upgrade smoothly.
