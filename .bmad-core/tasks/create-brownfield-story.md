<!-- Powered by BMAD™ Core -->

# Create Brownfield Story Task

## Purpose

Create detailed, implementation-ready stories for brownfield projects where traditional sharded PRD/architecture documents may not exist. This task bridges the gap between various documentation formats (document-project output, brownfield PRDs, epics, or user documentation) and executable stories for the Dev agent.

## When to Use This Task

**Use this task when:**

- Working on brownfield projects with non-standard documentation
- Stories need to be created from document-project output
- Working from brownfield epics without full PRD/architecture
- Existing project documentation doesn't follow BMad v4+ structure
- Need to gather additional context from user during story creation

**Use create-next-story when:**

- Working with properly sharded PRD and v4 architecture documents
- Following standard greenfield or well-documented brownfield workflow
- All technical context is available in structured format

## Task Execution Instructions

### 0. Documentation Context

Check for available documentation in this order:

1. **Sharded PRD/Architecture** (docs/prd/, docs/architecture/)
   - If found, recommend using create-next-story task instead

2. **Brownfield Architecture Document** (docs/brownfield-architecture.md or similar)
   - Created by document-project task
   - Contains actual system state, technical debt, workarounds

3. **Brownfield PRD** (docs/prd.md)
   - May contain embedded technical details

4. **Epic Files** (docs/epics/ or similar)
   - Created by brownfield-create-epic task

5. **User-Provided Documentation**
   - Ask user to specify location and format

### 1. Story Identification and Context Gathering

#### 1.1 Identify Story Source

Based on available documentation:

- **From Brownfield PRD**: Extract stories from epic sections
- **From Epic Files**: Read epic definition and story list
- **From User Direction**: Ask user which specific enhancement to implement
- **No Clear Source**: Work with user to define the story scope

#### 1.2 Gather Essential Context

CRITICAL: For brownfield stories, you MUST gather enough context for safe implementation. Be prepared to ask the user for missing information.

**Required Information Checklist:**

- [ ] What existing functionality might be affected?
- [ ] What are the integration points with current code?
- [ ] What patterns should be followed (with examples)?
- [ ] What technical constraints exist?
- [ ] Are there any "gotchas" or workarounds to know about?

If any required information is missing, list the missing information and ask the user to provide it.

### 2. Extract Technical Context from Available Sources

#### 2.1 From Document-Project Output

If using brownfield-architecture.md from document-project:

- **Technical Debt Section**: Note any workarounds affecting this story
- **Key Files Section**: Identify files that will need modification
- **Integration Points**: Find existing integration patterns
- **Known Issues**: Check if story touches problematic areas
- **Actual Tech Stack**: Verify versions and constraints

#### 2.2 From Brownfield PRD

If using brownfield PRD:

- **Technical Constraints Section**: Extract all relevant constraints
- **Integration Requirements**: Note compatibility requirements
- **Code Organization**: Follow specified patterns
- **Risk Assessment**: Understand potential impacts

#### 2.3 From User Documentation

Ask the user to help identify:

- Relevant technical specifications
- Existing code examples to follow
- Integration requirements
- Testing approaches used in the project

### 3. Story Creation with Progressive Detail Gathering

#### 3.1 Create Initial Story Structure

Start with the story template, filling in what's known:

```markdown
# Story {{Enhancement Title}}

## Status: Draft

## Story

As a {{user_type}},
I want {{enhancement_capability}},
so that {{value_delivered}}.

## Context Source

- Source Document: {{document name/type}}
- Enhancement Type: {{single feature/bug fix/integration/etc}}
- Existing System Impact: {{brief assessment}}
```

#### 3.2 Develop Acceptance Criteria

Critical: For brownfield, ALWAYS include criteria about maintaining existing functionality

Standard structure:

1. New functionality works as specified
2. Existing {{affected feature}} continues to work unchanged
3. Integration with {{existing system}} maintains current behavior
4. No regression in {{related area}}
5. Performance remains within acceptable bounds

#### 3.3 Gather Technical Guidance

Critical: This is where you'll need to be interactive with the user if information is missing

Create Dev Technical Guidance section with available information:

````markdown
## Dev Technical Guidance

### Existing System Context

[Extract from available documentation]

### Integration Approach

[Based on patterns found or ask user]

### Technical Constraints

[From documentation or user input]

### Missing Information

Critical: List anything you couldn't find that dev will need and ask for the missing information

### 4. Task Generation with Safety Checks

#### 4.1 Generate Implementation Tasks

Based on gathered context, create tasks that:

- Include exploration tasks if system understanding is incomplete
- Add verification tasks for existing functionality
- Include rollback considerations
- Reference specific files/patterns when known

Example task structure for brownfield:

```markdown
## Tasks / Subtasks

- [ ] Task 1: Analyze existing {{component/feature}} implementation
  - [ ] Review {{specific files}} for current patterns
  - [ ] Document integration points
  - [ ] Identify potential impacts

- [ ] Task 2: Implement {{new functionality}}
  - [ ] Follow pattern from {{example file}}
  - [ ] Integrate with {{existing component}}
  - [ ] Maintain compatibility with {{constraint}}

- [ ] Task 3: Verify existing functionality
  - [ ] Test {{existing feature 1}} still works
  - [ ] Verify {{integration point}} behavior unchanged
  - [ ] Check performance impact

- [ ] Task 4: Add tests
  - [ ] Unit tests following {{project test pattern}}
  - [ ] Integration test for {{integration point}}
  - [ ] Update existing tests if needed
```
````

### 5. Risk Assessment and Mitigation

CRITICAL: for brownfield - always include risk assessment

Add section for brownfield-specific risks:

```markdown
## Risk Assessment

### Implementation Risks

- **Primary Risk**: {{main risk to existing system}}
- **Mitigation**: {{how to address}}
- **Verification**: {{how to confirm safety}}

### Rollback Plan

- {{Simple steps to undo changes if needed}}

### Safety Checks

- [ ] Existing {{feature}} tested before changes
- [ ] Changes can be feature-flagged or isolated
- [ ] Rollback procedure documented
```

### 6. Final Story Validation

Before finalizing:

1. **Completeness Check**:
   - [ ] Story has clear scope and acceptance criteria
   - [ ] Technical context is sufficient for implementation
   - [ ] Integration approach is defined
   - [ ] Risks are identified with mitigation

2. **Safety Check**:
   - [ ] Existing functionality protection included
   - [ ] Rollback plan is feasible
   - [ ] Testing covers both new and existing features

3. **Information Gaps**:
   - [ ] All critical missing information gathered from user
   - [ ] Remaining unknowns documented for dev agent
   - [ ] Exploration tasks added where needed

### 7. Story Output Format

Save the story with appropriate naming:

- If from epic: `docs/stories/epic-{n}-story-{m}.md`
- If standalone: `docs/stories/brownfield-{feature-name}.md`
- If sequential: Follow existing story numbering

Include header noting documentation context:

```markdown
# Story: {{Title}}

<!-- Source: {{documentation type used}} -->
<!-- Context: Brownfield enhancement to {{existing system}} -->

## Status: Draft

[Rest of story content...]
```

### 8. Handoff Communication

Provide clear handoff to the user:

```text
Brownfield story created: {{story title}}

Source Documentation: {{what was used}}
Story Location: {{file path}}

Key Integration Points Identified:
- {{integration point 1}}
- {{integration point 2}}

Risks Noted:
- {{primary risk}}

{{If missing info}}:
Note: Some technical details were unclear. The story includes exploration tasks to gather needed information during implementation.

Next Steps:
1. Review story for accuracy
2. Verify integration approach aligns with your system
3. Approve story or request adjustments
4. Dev agent can then implement with safety checks
```

## Success Criteria

The brownfield story creation is successful when:

1. Story can be implemented without requiring dev to search multiple documents
2. Integration approach is clear and safe for existing system
3. All available technical context has been extracted and organized
4. Missing information has been identified and addressed
5. Risks are documented with mitigation strategies
6. Story includes verification of existing functionality
7. Rollback approach is defined

## Important Notes

- This task is specifically for brownfield projects with non-standard documentation
- Always prioritize existing system stability over new features
- When in doubt, add exploration and verification tasks
- It's better to ask the user for clarification than make assumptions
- Each story should be self-contained for the dev agent
- Include references to existing code patterns when available
