# Model Context Approval Process

This document outlines the approval requirements for changes to model context JSON files in the LiteLLM repository.

## Overview

Changes to model context files (such as `model_prices_and_context_window.json`) require special approval due to their critical nature in the LiteLLM ecosystem. These files contain pricing information, context window sizes, and other metadata that directly affects how LiteLLM routes requests and calculates costs.

## Files Requiring Approval

The following files require 2 approvals and cited sources for any changes:

- `model_prices_and_context_window.json`
- Any file matching `**/model_context*.json`
- Any file matching `**/context_window*.json`

## GitHub Ruleset Protection

This repository uses **GitHub Rulesets** to enforce approval requirements for model context files. The ruleset configuration is located at:

- `.github/rulesets/model-context-files.yml` - Main ruleset for model context file protection
- `.github/rulesets/model-context-protection.yml` - General model context protection

### Ruleset Features

- **2 approvals required** from different reviewers
- **Source citation validation** via GitHub Actions
- **JSON structure validation** before merge
- **Bypass permissions** for organization admins and maintainers
- **Automatic enforcement** on main, master, and develop branches

## Approval Requirements

### 1. Two Approvals Required

All pull requests that modify model context files must receive **at least 2 approvals** from different reviewers before they can be merged.

### 2. Cited Sources Required

All changes must include proper source citations. This helps maintain data integrity and provides traceability for pricing and context window information.

## How to Submit Changes

### For Manual Changes

1. **Create your PR** with changes to model context files
2. **Include source citations** in your PR description:
   ```
   **Sources:**
   - Provider API: https://provider.com/api/pricing
   - Documentation: https://provider.com/docs/context-windows
   - Release notes: https://provider.com/releases/v1.2.3
   ```
3. **Wait for 2 approvals** from different reviewers
4. **Merge** once approved

### For Automated Updates

The automated weekly updates to `model_prices_and_context_window.json` already include proper source citations and will be subject to the same approval requirements.

## Source Citation Guidelines

When citing sources, include:

- **API endpoints** used to fetch data
- **Official documentation** links
- **Release notes** or changelogs
- **Provider announcements** or blog posts
- **GitHub issues** or discussions (if relevant)

### Good Examples

```
**Sources:**
- OpenAI API Pricing: https://openai.com/pricing
- Anthropic Documentation: https://docs.anthropic.com/claude/docs
- Model Release Notes: https://openai.com/blog/gpt-4-turbo-and-gpt-4-vision
```

### Bad Examples

```
- "Updated pricing" (no source)
- "Based on provider website" (too vague)
- "Internal knowledge" (not verifiable)
```

## Automated Checks

The following automated checks are in place:

### GitHub Rulesets

- **Files**: 
  - `.github/rulesets/model-context-files.yml` - Main protection ruleset
  - `.github/rulesets/model-context-protection.yml` - General protection
- **Features**:
  - Enforces 2 approvals requirement
  - Validates JSON structure via status checks
  - Requires source citations
  - Bypass permissions for admins/maintainers

### GitHub Actions Validation

- **File**: `.github/workflows/model-context-validation.yml`
- **Triggers**: On pull requests that modify model context files
- **Checks**:
  - Validates JSON structure
  - Searches for source citations in PR description
  - Provides helpful error messages

### JSON Structure Validation

- **File**: `.github/workflows/json-structure-validation.yml`
- **Triggers**: On pull requests that modify model context files
- **Checks**:
  - Validates JSON syntax
  - Checks for required fields in model entries
  - Warns about missing cost information
  - Monitors file size

### Pre-commit Hook

- **File**: `.github/scripts/validate_model_context_changes.py`
- **Triggers**: Before commits that modify model context files
- **Checks**:
  - Validates JSON structure
  - Searches for source citations in commit messages
  - Prevents commits without proper citations

## Enforcement

### CI/CD Integration

The approval workflow is integrated into the CI/CD pipeline:

1. **GitHub Rulesets** enforce 2 approvals requirement at the repository level
2. **Pre-commit validation** prevents commits without citations
3. **GitHub Actions** validate JSON structure and source citations
4. **Status checks** must pass before merging

### Bypass Conditions

The approval requirements can be bypassed only in emergency situations with:

1. **Explicit maintainer override** with documented justification
2. **Post-merge review** within 24 hours
3. **Retroactive source citation** if missing

## Troubleshooting

### Common Issues

**"No source citations found"**
- Add source URLs to your PR description
- Include API documentation links
- Reference official provider announcements

**"JSON validation failed"**
- Ensure the JSON file is properly formatted
- Check for syntax errors
- Validate against the expected schema

**"Insufficient approvals"**
- Request reviews from team members
- Ensure at least 2 different people approve
- Wait for the approval workflow to complete

### Getting Help

If you encounter issues with the approval process:

1. **Check the workflow logs** in the GitHub Actions tab
2. **Review the pre-commit hook output** in your terminal
3. **Contact maintainers** via GitHub issues or Discord
4. **Refer to this documentation** for guidance

## Best Practices

### For Contributors

- **Always include sources** when making model context changes
- **Test changes locally** before submitting PRs
- **Request reviews early** to avoid delays
- **Be specific** in your source citations

### For Reviewers

- **Verify source accuracy** before approving
- **Check for completeness** of citations
- **Validate JSON structure** if making changes
- **Provide constructive feedback** for improvements

## History

This approval process was implemented to:

- **Ensure data accuracy** in model pricing and context information
- **Maintain traceability** for all changes
- **Prevent accidental modifications** that could affect production systems
- **Establish clear governance** for critical configuration files

## Related Documentation

- [Contributing Guidelines](CONTRIBUTING.md)
- [GitHub Workflows](.github/workflows/)
- [Model Context Files](model_prices_and_context_window.json)