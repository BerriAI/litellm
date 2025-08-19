# /test-design Task

When this command is used, execute the following task:

<!-- Powered by BMAD™ Core -->

# test-design

Create comprehensive test scenarios with appropriate test level recommendations for story implementation.

## Inputs

```yaml
required:
  - story_id: '{epic}.{story}' # e.g., "1.3"
  - story_path: '{devStoryLocation}/{epic}.{story}.*.md' # Path from core-config.yaml
  - story_title: '{title}' # If missing, derive from story file H1
  - story_slug: '{slug}' # If missing, derive from title (lowercase, hyphenated)
```

## Purpose

Design a complete test strategy that identifies what to test, at which level (unit/integration/e2e), and why. This ensures efficient test coverage without redundancy while maintaining appropriate test boundaries.

## Dependencies

```yaml
data:
  - test-levels-framework.md # Unit/Integration/E2E decision criteria
  - test-priorities-matrix.md # P0/P1/P2/P3 classification system
```

## Process

### 1. Analyze Story Requirements

Break down each acceptance criterion into testable scenarios. For each AC:

- Identify the core functionality to test
- Determine data variations needed
- Consider error conditions
- Note edge cases

### 2. Apply Test Level Framework

**Reference:** Load `test-levels-framework.md` for detailed criteria

Quick rules:

- **Unit**: Pure logic, algorithms, calculations
- **Integration**: Component interactions, DB operations
- **E2E**: Critical user journeys, compliance

### 3. Assign Priorities

**Reference:** Load `test-priorities-matrix.md` for classification

Quick priority assignment:

- **P0**: Revenue-critical, security, compliance
- **P1**: Core user journeys, frequently used
- **P2**: Secondary features, admin functions
- **P3**: Nice-to-have, rarely used

### 4. Design Test Scenarios

For each identified test need, create:

```yaml
test_scenario:
  id: '{epic}.{story}-{LEVEL}-{SEQ}'
  requirement: 'AC reference'
  priority: P0|P1|P2|P3
  level: unit|integration|e2e
  description: 'What is being tested'
  justification: 'Why this level was chosen'
  mitigates_risks: ['RISK-001'] # If risk profile exists
```

### 5. Validate Coverage

Ensure:

- Every AC has at least one test
- No duplicate coverage across levels
- Critical paths have multiple levels
- Risk mitigations are addressed

## Outputs

### Output 1: Test Design Document

**Save to:** `qa.qaLocation/assessments/{epic}.{story}-test-design-{YYYYMMDD}.md`

```markdown
# Test Design: Story {epic}.{story}

Date: {date}
Designer: Quinn (Test Architect)

## Test Strategy Overview

- Total test scenarios: X
- Unit tests: Y (A%)
- Integration tests: Z (B%)
- E2E tests: W (C%)
- Priority distribution: P0: X, P1: Y, P2: Z

## Test Scenarios by Acceptance Criteria

### AC1: {description}

#### Scenarios

| ID           | Level       | Priority | Test                      | Justification            |
| ------------ | ----------- | -------- | ------------------------- | ------------------------ |
| 1.3-UNIT-001 | Unit        | P0       | Validate input format     | Pure validation logic    |
| 1.3-INT-001  | Integration | P0       | Service processes request | Multi-component flow     |
| 1.3-E2E-001  | E2E         | P1       | User completes journey    | Critical path validation |

[Continue for all ACs...]

## Risk Coverage

[Map test scenarios to identified risks if risk profile exists]

## Recommended Execution Order

1. P0 Unit tests (fail fast)
2. P0 Integration tests
3. P0 E2E tests
4. P1 tests in order
5. P2+ as time permits
```

### Output 2: Gate YAML Block

Generate for inclusion in quality gate:

```yaml
test_design:
  scenarios_total: X
  by_level:
    unit: Y
    integration: Z
    e2e: W
  by_priority:
    p0: A
    p1: B
    p2: C
  coverage_gaps: [] # List any ACs without tests
```

### Output 3: Trace References

Print for use by trace-requirements task:

```text
Test design matrix: qa.qaLocation/assessments/{epic}.{story}-test-design-{YYYYMMDD}.md
P0 tests identified: {count}
```

## Quality Checklist

Before finalizing, verify:

- [ ] Every AC has test coverage
- [ ] Test levels are appropriate (not over-testing)
- [ ] No duplicate coverage across levels
- [ ] Priorities align with business risk
- [ ] Test IDs follow naming convention
- [ ] Scenarios are atomic and independent

## Key Principles

- **Shift left**: Prefer unit over integration, integration over E2E
- **Risk-based**: Focus on what could go wrong
- **Efficient coverage**: Test once at the right level
- **Maintainability**: Consider long-term test maintenance
- **Fast feedback**: Quick tests run first
