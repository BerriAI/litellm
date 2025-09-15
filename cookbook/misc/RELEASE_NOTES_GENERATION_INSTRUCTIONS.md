# LiteLLM Release Notes Generation Instructions

This document provides comprehensive instructions for AI agents to generate release notes for LiteLLM following the established format and style.

## Required Inputs

1. **Release Version** (e.g., `v1.76.3-stable`)
2. **PR Diff/Changelog** - List of PRs with titles and contributors
3. **Previous Version Commit Hash** - To compare model pricing changes
4. **Reference Release Notes** - Previous release notes to follow style/format

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

Follow this exact structure based on `docs/my-website/release_notes/v1.76.1-stable/index.md`:

```markdown
---
title: "v1.76.X-stable - [Key Theme]"
slug: "v1-76-X"
date: YYYY-MM-DDTHH:mm:ss
authors: [standard author block]
hide_table_of_contents: false
---

## Deploy this version
[Docker and pip installation tabs]

## Key Highlights
[3-5 bullet points of major features]

## Major Changes
[Critical changes users need to know]

## Performance Improvements
[Performance-related changes]

## New Models / Updated Models
[Detailed model tables and provider updates]

## LLM API Endpoints
[API-related features and fixes]

## Management Endpoints / UI
[Admin interface and management changes]

## Logging / Guardrail Integrations
[Observability and security features]

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
- Group by provider
- Note pricing corrections
- Highlight deprecated models

**Provider Features:**
- Group by provider (Gemini, OpenAI, Anthropic, etc.)
- Link to provider docs: `../../docs/providers/[provider_name]`
- Separate features from bug fixes

**API Endpoints:**
- Images API
- Video Generation (if applicable)
- Responses API
- Passthrough endpoints
- General chat completions

**UI/Management:**
- Authentication changes
- Dashboard improvements
- Team management
- Key management

**Integrations:**
- Logging providers (Datadog, Braintrust, etc.)
- Guardrails
- Cost tracking
- Observability

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

**Provider Features:**
```markdown
- **[Provider Name](../../docs/providers/provider)**
    - Feature description - [PR #XXXXX](link)
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
