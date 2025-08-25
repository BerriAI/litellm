<!-- Powered by BMAD™ Core -->

# Workflow Management

Enables BMad orchestrator to manage and execute team workflows.

## Dynamic Workflow Loading

Read available workflows from current team configuration's `workflows` field. Each team bundle defines its own supported workflows.

**Key Commands**:

- `/workflows` - List workflows in current bundle or workflows folder
- `/agent-list` - Show agents in current bundle

## Workflow Commands

### /workflows

Lists available workflows with titles and descriptions.

### /workflow-start {workflow-id}

Starts workflow and transitions to first agent.

### /workflow-status

Shows current progress, completed artifacts, and next steps.

### /workflow-resume

Resumes workflow from last position. User can provide completed artifacts.

### /workflow-next

Shows next recommended agent and action.

## Execution Flow

1. **Starting**: Load definition → Identify first stage → Transition to agent → Guide artifact creation

2. **Stage Transitions**: Mark complete → Check conditions → Load next agent → Pass artifacts

3. **Artifact Tracking**: Track status, creator, timestamps in workflow_state

4. **Interruption Handling**: Analyze provided artifacts → Determine position → Suggest next step

## Context Passing

When transitioning, pass:

- Previous artifacts
- Current workflow stage
- Expected outputs
- Decisions/constraints

## Multi-Path Workflows

Handle conditional paths by asking clarifying questions when needed.

## Best Practices

1. Show progress
2. Explain transitions
3. Preserve context
4. Allow flexibility
5. Track state

## Agent Integration

Agents should be workflow-aware: know active workflow, their role, access artifacts, understand expected outputs.
