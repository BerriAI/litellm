# Infrastructure & DevOps Expansion Pack

## Overview

This expansion pack extends BMad Method with comprehensive infrastructure and DevOps capabilities. It's designed for teams that need to define, implement, and manage cloud infrastructure alongside their application development.

## Purpose

While the core BMad flow focuses on getting from business requirements to development (Analyst → PM → Architect → SM → Dev), many projects require sophisticated infrastructure planning and implementation. This expansion pack adds:

- Infrastructure architecture design capabilities
- Platform engineering implementation workflows
- DevOps automation and CI/CD pipeline design
- Cloud resource management and optimization
- Security and compliance validation

## When to Use This Pack

Install this expansion pack when your project requires:

- Cloud infrastructure design and implementation
- Kubernetes/container platform setup
- Service mesh and GitOps workflows
- Infrastructure as Code (IaC) development
- Platform engineering and DevOps practices

## What's Included

### Agents

- `devops.yaml` - DevOps and Platform Engineering agent configuration

### Personas

- `devops.md` - DevOps Engineer persona (Alex)

### IDE Agents

- `devops.ide.md` - IDE-specific DevOps agent configuration

### Templates

- `infrastructure-architecture-tmpl.md` - Infrastructure architecture design template
- `infrastructure-platform-from-arch-tmpl.md` - Platform implementation from architecture template

### Tasks

- `infra/validate-infrastructure.md` - Infrastructure validation workflow
- `infra/review-infrastructure.md` - Infrastructure review process

### Checklists

- `infrastructure-checklist.md` - Comprehensive 16-section infrastructure validation checklist

## Integration with Core BMad

This expansion pack integrates with the core BMad flow at these points:

1. **After Architecture Phase**: The Architect can trigger infrastructure architecture design
2. **Parallel to Development**: Infrastructure implementation can proceed alongside application development
3. **Before Deployment**: Infrastructure must be validated before application deployment

## Installation

To install this expansion pack, run:

```bash
npm run install:expansion infrastructure
```

Or manually:

```bash
node tools/install-expansion-pack.js infrastructure
```

This will:

1. Copy all files to their appropriate locations in `.bmad-core/`
2. Update any necessary configurations
3. Make the DevOps agent available in teams

## Usage Examples

### 1. Infrastructure Architecture Design

After the main architecture is complete:

```bash
# Using the Architect agent
*create-infrastructure

# Or directly with DevOps agent
npm run agent devops
```

### 2. Platform Implementation

With an approved infrastructure architecture:

```bash
# DevOps agent implements the platform
*implement-platform
```

### 3. Infrastructure Validation

Before deployment:

```bash
# Validate infrastructure against checklist
*validate-infra
```

## Team Integration

The DevOps agent can be added to team configurations:

- `team-technical.yaml` - For technical implementation teams
- `team-full-org.yaml` - For complete organizational teams

## Dependencies

This expansion pack works best when used with:

- Core BMad agents (especially Architect)
- Technical preferences documentation
- Approved PRD and system architecture

## Customization

You can customize this expansion pack by:

1. Modifying the infrastructure templates for your cloud provider
2. Adjusting the checklist items for your compliance needs
3. Adding custom tasks for your specific workflows

## Notes

- Infrastructure work requires real-world cloud credentials and configurations
- The templates use placeholders ({{variable}}) that need actual values
- Always validate infrastructure changes before production deployment

---

_Version: 1.0_
_Compatible with: BMad Method v4_
