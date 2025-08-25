<!-- Powered by BMAD™ Core -->

# BMad Document Template Specification

## Overview

BMad document templates are defined in YAML format to drive interactive document generation and agent interaction. Templates separate structure definition from content generation, making them both human and LLM-agent-friendly.

## Template Structure

```yaml
template:
  id: template-identifier
  name: Human Readable Template Name
  version: 1.0
  output:
    format: markdown
    filename: default-path/to/{{filename}}.md
    title: '{{variable}} Document Title'

workflow:
  mode: interactive
  elicitation: advanced-elicitation

sections:
  - id: section-id
    title: Section Title
    instruction: |
      Detailed instructions for the LLM on how to handle this section
    # ... additional section properties
```

## Core Fields

### Template Metadata

- **id**: Unique identifier for the template
- **name**: Human-readable name displayed in UI
- **version**: Template version for tracking changes
- **output.format**: Default "markdown" for document templates
- **output.filename**: Default output file path (can include variables)
- **output.title**: Document title (becomes H1 in markdown)

### Workflow Configuration

- **workflow.mode**: Default interaction mode ("interactive" or "yolo")
- **workflow.elicitation**: Elicitation task to use ("advanced-elicitation")

## Section Properties

### Required Fields

- **id**: Unique section identifier
- **title**: Section heading text
- **instruction**: Detailed guidance for LLM on handling this section

### Optional Fields

#### Content Control

- **type**: Content type hint for structured sections
- **template**: Fixed template text for section content
- **item_template**: Template for repeatable items within section
- **prefix**: Prefix for numbered items (e.g., "FR", "NFR")

#### Behavior Flags

- **elicit**: Boolean - Apply elicitation after section rendered
- **repeatable**: Boolean - Section can be repeated multiple times
- **condition**: String - Condition for including section (e.g., "has ui requirements")

#### Agent Permissions

- **owner**: String - Agent role that initially creates/populates this section
- **editors**: Array - List of agent roles allowed to modify this section
- **readonly**: Boolean - Section cannot be modified after initial creation

#### Content Guidance

- **examples**: Array of example content (not included in output)
- **choices**: Object with choice options for common decisions
- **placeholder**: Default placeholder text

#### Structure

- **sections**: Array of nested child sections

## Supported Types

### Content Types

- **bullet-list**: Unordered list items
- **numbered-list**: Ordered list with optional prefix
- **paragraphs**: Free-form paragraph text
- **table**: Structured table data
- **code-block**: Code or configuration blocks
- **template-text**: Fixed template with variable substitution
- **mermaid**: Mermaid diagram with specified type and details

### Special Types

- **repeatable-container**: Container for multiple instances
- **conditional-block**: Content shown based on conditions
- **choice-selector**: Present choices to user

## Advanced Features

### Variable Substitution

Use `{{variable_name}}` in titles, templates, and content:

```yaml
title: 'Epic {{epic_number}} {{epic_title}}'
template: 'As a {{user_type}}, I want {{action}}, so that {{benefit}}.'
```

### Conditional Sections

```yaml
- id: ui-section
  title: User Interface Design
  condition: Project has UX/UI Requirements
  instruction: Only include if project has UI components
```

### Choice Integration

```yaml
choices:
  architecture: [Monolith, Microservices, Serverless]
  testing: [Unit Only, Unit + Integration, Full Pyramid]
```

### Mermaid Diagrams

```yaml
- id: system-architecture
  title: System Architecture Diagram
  type: mermaid
  instruction: Create a system architecture diagram showing key components and data flow
  mermaid_type: flowchart
  details: |
    Show the following components:
    - User interface layer
    - API gateway
    - Core services
    - Database layer
    - External integrations
```

**Supported mermaid_type values:**

**Core Diagram Types:**

- `flowchart` - Flow charts and process diagrams
- `sequenceDiagram` - Sequence diagrams for interactions
- `classDiagram` - Class relationship diagrams (UML)
- `stateDiagram` - State transition diagrams
- `erDiagram` - Entity relationship diagrams
- `gantt` - Gantt charts for timelines
- `pie` - Pie charts for data visualization

**Advanced Diagram Types:**

- `journey` - User journey maps
- `mindmap` - Mindmaps for brainstorming
- `timeline` - Timeline diagrams for chronological events
- `quadrantChart` - Quadrant charts for data categorization
- `xyChart` - XY charts (bar charts, line charts)
- `sankey` - Sankey diagrams for flow visualization

**Specialized Types:**

- `c4Context` - C4 context diagrams (experimental)
- `requirement` - Requirement diagrams
- `packet` - Network packet diagrams
- `block` - Block diagrams
- `kanban` - Kanban boards

### Agent Permissions Example

```yaml
- id: story-details
  title: Story
  owner: scrum-master
  editors: [scrum-master]
  readonly: false
  sections:
    - id: dev-notes
      title: Dev Notes
      owner: dev-agent
      editors: [dev-agent]
      readonly: false
      instruction: Implementation notes and technical details
    - id: qa-results
      title: QA Results
      owner: qa-agent
      editors: [qa-agent]
      readonly: true
      instruction: Quality assurance test results
```

### Repeatable Sections

```yaml
- id: epic-details
  title: Epic {{epic_number}} {{epic_title}}
  repeatable: true
  sections:
    - id: story
      title: Story {{epic_number}}.{{story_number}} {{story_title}}
      repeatable: true
      sections:
        - id: criteria
          title: Acceptance Criteria
          type: numbered-list
          item_template: '{{criterion_number}}: {{criteria}}'
          repeatable: true
```

### Examples with Code Blocks

````yaml
examples:
  - 'FR6: The system must authenticate users within 2 seconds'
  - |
    ```mermaid
    sequenceDiagram
        participant User
        participant API
        participant DB
        User->>API: POST /login
        API->>DB: Validate credentials
        DB-->>API: User data
        API-->>User: JWT token
    ```
  - |
    **Architecture Decision Record**

    **Decision**: Use PostgreSQL for primary database
    **Rationale**: ACID compliance and JSON support needed
    **Consequences**: Requires database management expertise
````

## Section Hierarchy

Templates define the complete document structure starting with the first H2 - each level in is the next H#:

```yaml
sections:
  - id: overview
    title: Project Overview
    sections:
      - id: goals
        title: Goals
      - id: scope
        title: Scope
        sections:
          - id: in-scope
            title: In Scope
          - id: out-scope
            title: Out of Scope
```

## Processing Flow

1. **Parse Template**: Load and validate YAML structure
2. **Initialize Workflow**: Set interaction mode and elicitation
3. **Process Sections**: Handle each section in order:
   - Check conditions
   - Apply instructions
   - Generate content
   - Handle choices and variables
   - Apply elicitation if specified
   - Process nested sections
4. **Generate Output**: Create clean markdown document

## Best Practices

### Template Design

- Keep instructions clear and specific
- Use examples for complex content
- Structure sections logically
- Include all necessary guidance for LLM

### Content Instructions

- Be explicit about expected format
- Include reasoning for decisions
- Specify interaction patterns
- Reference other documents when needed

### Variable Naming

- Use descriptive variable names
- Follow consistent naming conventions
- Document expected variable values

### Examples Usage

- Provide concrete examples for complex sections
- Include both simple and complex cases
- Use realistic project scenarios
- Include code blocks and diagrams when helpful

## Validation

Templates should be validated for:

- Valid YAML syntax
- Required fields present
- Consistent section IDs
- Proper nesting structure
- Valid variable references

## Migration from Legacy

When converting from markdown+frontmatter templates:

1. Extract embedded `[[LLM:]]` instructions to `instruction` fields
2. Convert `<<REPEAT>>` blocks to `repeatable: true` sections
3. Extract `^^CONDITIONS^^` to `condition` fields
4. Move `@{examples}` to `examples` arrays
5. Convert `{{placeholders}}` to proper variable syntax

This specification ensures templates are both human-readable and machine-processable while maintaining the flexibility needed for complex document generation.
