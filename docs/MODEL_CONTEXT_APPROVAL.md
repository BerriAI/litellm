# Model Context Approval Process

This document outlines the approval requirements for changes to model context JSON files in the LiteLLM repository.

## Overview

Changes to model context files (such as `model_prices_and_context_window.json`) require special approval due to their critical nature in the LiteLLM ecosystem. These files contain pricing information, context window sizes, and other metadata that directly affects how LiteLLM routes requests and calculates costs.

## Files Requiring Approval

The following files require 2 team member approvals for any changes:

- `model_prices_and_context_window.json`
- Any file matching `**/model_context*.json`
- Any file matching `**/context_window*.json`

## GitHub Ruleset Protection

This repository uses a **GitHub Ruleset** to enforce approval requirements for model context files.

**Ruleset Configuration:** `.github/rulesets/model-context-simple.yml`

### Ruleset Features

- **2 team member approvals required** from different reviewers
- **Automatic enforcement** on main, master, and develop branches
- **Bypass permissions** for organization admins and maintainers
- **Simple and focused** - no complex validation, just approval requirements

## How to Submit Changes

### For Manual Changes

1. **Create your PR** with changes to model context files
2. **Request reviews** from at least 2 team members
3. **Wait for 2 approvals** from different reviewers
4. **Merge** once approved

### For Automated Updates

The automated weekly updates to `model_prices_and_context_window.json` will be subject to the same approval requirements.

## Enforcement

### GitHub Ruleset

The approval workflow is enforced by GitHub Rulesets:

1. **Repository-level protection** prevents merging without 2 approvals
2. **Automatic enforcement** on protected branches
3. **Bypass options** for organization admins and maintainers

### Bypass Conditions

The approval requirements can be bypassed only by:

1. **Organization administrators**
2. **Repository maintainers**

## Getting Help

If you encounter issues with the approval process:

1. **Check the ruleset status** in the GitHub repository settings
2. **Request reviews** from team members
3. **Contact maintainers** via GitHub issues or Discord

## Best Practices

### For Contributors

- **Request reviews early** to avoid delays
- **Tag team members** for review requests
- **Be patient** - 2 approvals are required for security

### For Reviewers

- **Review promptly** when requested
- **Verify changes** before approving
- **Provide feedback** if changes are needed

## History

This approval process was implemented to:

- **Ensure data accuracy** in model pricing and context information
- **Prevent accidental modifications** that could affect production systems
- **Establish clear governance** for critical configuration files
- **Require team consensus** for important changes

## Related Documentation

- [Contributing Guidelines](CONTRIBUTING.md)
- [GitHub Rulesets](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-rulesets)