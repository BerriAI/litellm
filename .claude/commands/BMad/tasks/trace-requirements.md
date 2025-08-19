# /trace-requirements Task

When this command is used, execute the following task:

<!-- Powered by BMAD™ Core -->

# trace-requirements

Map story requirements to test cases using Given-When-Then patterns for comprehensive traceability.

## Purpose

Create a requirements traceability matrix that ensures every acceptance criterion has corresponding test coverage. This task helps identify gaps in testing and ensures all requirements are validated.

**IMPORTANT**: Given-When-Then is used here for documenting the mapping between requirements and tests, NOT for writing the actual test code. Tests should follow your project's testing standards (no BDD syntax in test code).

## Prerequisites

- Story file with clear acceptance criteria
- Access to test files or test specifications
- Understanding of the implementation

## Traceability Process

### 1. Extract Requirements

Identify all testable requirements from:

- Acceptance Criteria (primary source)
- User story statement
- Tasks/subtasks with specific behaviors
- Non-functional requirements mentioned
- Edge cases documented

### 2. Map to Test Cases

For each requirement, document which tests validate it. Use Given-When-Then to describe what the test validates (not how it's written):

```yaml
requirement: 'AC1: User can login with valid credentials'
test_mappings:
  - test_file: 'auth/login.test.ts'
    test_case: 'should successfully login with valid email and password'
    # Given-When-Then describes WHAT the test validates, not HOW it's coded
    given: 'A registered user with valid credentials'
    when: 'They submit the login form'
    then: 'They are redirected to dashboard and session is created'
    coverage: full

  - test_file: 'e2e/auth-flow.test.ts'
    test_case: 'complete login flow'
    given: 'User on login page'
    when: 'Entering valid credentials and submitting'
    then: 'Dashboard loads with user data'
    coverage: integration
```

### 3. Coverage Analysis

Evaluate coverage for each requirement:

**Coverage Levels:**

- `full`: Requirement completely tested
- `partial`: Some aspects tested, gaps exist
- `none`: No test coverage found
- `integration`: Covered in integration/e2e tests only
- `unit`: Covered in unit tests only

### 4. Gap Identification

Document any gaps found:

```yaml
coverage_gaps:
  - requirement: 'AC3: Password reset email sent within 60 seconds'
    gap: 'No test for email delivery timing'
    severity: medium
    suggested_test:
      type: integration
      description: 'Test email service SLA compliance'

  - requirement: 'AC5: Support 1000 concurrent users'
    gap: 'No load testing implemented'
    severity: high
    suggested_test:
      type: performance
      description: 'Load test with 1000 concurrent connections'
```

## Outputs

### Output 1: Gate YAML Block

**Generate for pasting into gate file under `trace`:**

```yaml
trace:
  totals:
    requirements: X
    full: Y
    partial: Z
    none: W
  planning_ref: 'qa.qaLocation/assessments/{epic}.{story}-test-design-{YYYYMMDD}.md'
  uncovered:
    - ac: 'AC3'
      reason: 'No test found for password reset timing'
  notes: 'See qa.qaLocation/assessments/{epic}.{story}-trace-{YYYYMMDD}.md'
```

### Output 2: Traceability Report

**Save to:** `qa.qaLocation/assessments/{epic}.{story}-trace-{YYYYMMDD}.md`

Create a traceability report with:

```markdown
# Requirements Traceability Matrix

## Story: {epic}.{story} - {title}

### Coverage Summary

- Total Requirements: X
- Fully Covered: Y (Z%)
- Partially Covered: A (B%)
- Not Covered: C (D%)

### Requirement Mappings

#### AC1: {Acceptance Criterion 1}

**Coverage: FULL**

Given-When-Then Mappings:

- **Unit Test**: `auth.service.test.ts::validateCredentials`
  - Given: Valid user credentials
  - When: Validation method called
  - Then: Returns true with user object

- **Integration Test**: `auth.integration.test.ts::loginFlow`
  - Given: User with valid account
  - When: Login API called
  - Then: JWT token returned and session created

#### AC2: {Acceptance Criterion 2}

**Coverage: PARTIAL**

[Continue for all ACs...]

### Critical Gaps

1. **Performance Requirements**
   - Gap: No load testing for concurrent users
   - Risk: High - Could fail under production load
   - Action: Implement load tests using k6 or similar

2. **Security Requirements**
   - Gap: Rate limiting not tested
   - Risk: Medium - Potential DoS vulnerability
   - Action: Add rate limit tests to integration suite

### Test Design Recommendations

Based on gaps identified, recommend:

1. Additional test scenarios needed
2. Test types to implement (unit/integration/e2e/performance)
3. Test data requirements
4. Mock/stub strategies

### Risk Assessment

- **High Risk**: Requirements with no coverage
- **Medium Risk**: Requirements with only partial coverage
- **Low Risk**: Requirements with full unit + integration coverage
```

## Traceability Best Practices

### Given-When-Then for Mapping (Not Test Code)

Use Given-When-Then to document what each test validates:

**Given**: The initial context the test sets up

- What state/data the test prepares
- User context being simulated
- System preconditions

**When**: The action the test performs

- What the test executes
- API calls or user actions tested
- Events triggered

**Then**: What the test asserts

- Expected outcomes verified
- State changes checked
- Values validated

**Note**: This is for documentation only. Actual test code follows your project's standards (e.g., describe/it blocks, no BDD syntax).

### Coverage Priority

Prioritize coverage based on:

1. Critical business flows
2. Security-related requirements
3. Data integrity requirements
4. User-facing features
5. Performance SLAs

### Test Granularity

Map at appropriate levels:

- Unit tests for business logic
- Integration tests for component interaction
- E2E tests for user journeys
- Performance tests for NFRs

## Quality Indicators

Good traceability shows:

- Every AC has at least one test
- Critical paths have multiple test levels
- Edge cases are explicitly covered
- NFRs have appropriate test types
- Clear Given-When-Then for each test

## Red Flags

Watch for:

- ACs with no test coverage
- Tests that don't map to requirements
- Vague test descriptions
- Missing edge case coverage
- NFRs without specific tests

## Integration with Gates

This traceability feeds into quality gates:

- Critical gaps → FAIL
- Minor gaps → CONCERNS
- Missing P0 tests from test-design → CONCERNS

### Output 3: Story Hook Line

**Print this line for review task to quote:**

```text
Trace matrix: qa.qaLocation/assessments/{epic}.{story}-trace-{YYYYMMDD}.md
```

- Full coverage → PASS contribution

## Key Principles

- Every requirement must be testable
- Use Given-When-Then for clarity
- Identify both presence and absence
- Prioritize based on risk
- Make recommendations actionable
