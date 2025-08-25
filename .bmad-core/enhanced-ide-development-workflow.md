# Enhanced IDE Development Workflow

This is a simple step-by-step guide to help you efficiently manage your development workflow using the BMad Method. The workflow integrates the Test Architect (QA agent) throughout the development lifecycle to ensure quality, prevent regressions, and maintain high standards. Refer to the **[<ins>User Guide</ins>](user-guide.md)** for any scenario that is not covered here.

## Create New Branch

1. **Start new branch**

## Story Creation (Scrum Master)

1. **Start new chat/conversation**
2. **Load SM agent**
3. **Execute**: `*draft` (runs create-next-story task)
4. **Review generated story** in `docs/stories/`
5. **Update status**: Change from "Draft" to "Approved"

## Story Implementation (Developer)

1. **Start new chat/conversation**
2. **Load Dev agent**
3. **Execute**: `*develop-story {selected-story}` (runs execute-checklist task)
4. **Review generated report** in `{selected-story}`

## Test Architect Integration Throughout Workflow

The Test Architect (Quinn) provides comprehensive quality assurance throughout the development lifecycle. Here's how to leverage each capability at the right time.

**Command Aliases:** Documentation uses short forms (`*risk`, `*design`, `*nfr`, `*trace`) for the full commands (`*risk-profile`, `*test-design`, `*nfr-assess`, `*trace-requirements`).

### Quick Command Reference

| **Stage**                | **Command** | **Purpose**                             | **Output**                                                      | **Priority**                |
| ------------------------ | ----------- | --------------------------------------- | --------------------------------------------------------------- | --------------------------- |
| **After Story Approval** | `*risk`     | Identify integration & regression risks | `docs/qa/assessments/{epic}.{story}-risk-{YYYYMMDD}.md`         | High for complex/brownfield |
|                          | `*design`   | Create test strategy for dev            | `docs/qa/assessments/{epic}.{story}-test-design-{YYYYMMDD}.md`  | High for new features       |
| **During Development**   | `*trace`    | Verify test coverage                    | `docs/qa/assessments/{epic}.{story}-trace-{YYYYMMDD}.md`        | Medium                      |
|                          | `*nfr`      | Validate quality attributes             | `docs/qa/assessments/{epic}.{story}-nfr-{YYYYMMDD}.md`          | High for critical features  |
| **After Development**    | `*review`   | Comprehensive assessment                | QA Results in story + `docs/qa/gates/{epic}.{story}-{slug}.yml` | **Required**                |
| **Post-Review**          | `*gate`     | Update quality decision                 | Updated `docs/qa/gates/{epic}.{story}-{slug}.yml`               | As needed                   |

### Stage 1: After Story Creation (Before Dev Starts)

**RECOMMENDED - Set Developer Up for Success:**

```bash
# 1. RISK ASSESSMENT (Run FIRST for complex stories)
@qa *risk {approved-story}
# Identifies:
#   - Technical debt impact
#   - Integration complexity
#   - Regression potential (1-9 scoring)
#   - Mitigation strategies
# Critical for: Brownfield, API changes, data migrations

# 2. TEST DESIGN (Run SECOND to guide implementation)
@qa *design {approved-story}
# Provides:
#   - Test scenarios per acceptance criterion
#   - Test level recommendations (unit/integration/E2E)
#   - Risk-based priorities (P0/P1/P2)
#   - Test data requirements
# Share with Dev: Include in story comments or attach to ticket
```

### Stage 2: During Development (Mid-Implementation Checkpoints)

**Developer Self-Service Quality Checks:**

```bash
# 3. REQUIREMENTS TRACING (Verify coverage mid-development)
@qa *trace {story-in-progress}
# Validates:
#   - All acceptance criteria have tests
#   - No missing test scenarios
#   - Appropriate test levels
#   - Given-When-Then documentation clarity
# Run when: After writing initial tests

# 4. NFR VALIDATION (Check quality attributes)
@qa *nfr {story-in-progress}
# Assesses:
#   - Security: Authentication, authorization, data protection
#   - Performance: Response times, resource usage
#   - Reliability: Error handling, recovery
#   - Maintainability: Code quality, documentation
# Run when: Before marking "Ready for Review"
```

### Stage 3: Story Review (Quality Gate Assessment)

**REQUIRED - Comprehensive Test Architecture Review:**

**Prerequisite:** All tests green locally; lint & type checks pass.

```bash
# 5. FULL REVIEW (Standard review process)
@qa *review {completed-story}
```

**What Happens During Review:**

1. **Deep Code Analysis**
   - Architecture pattern compliance
   - Code quality and maintainability
   - Security vulnerability scanning
   - Performance bottleneck detection

2. **Active Refactoring**
   - Improves code directly when safe
   - Fixes obvious issues immediately
   - Suggests complex refactoring for dev

3. **Test Validation**
   - Coverage at all levels (unit/integration/E2E)
   - Test quality (no flaky tests, proper assertions)
   - Regression test adequacy

4. **Gate Decision**
   - Creates: `docs/qa/gates/{epic}.{story}-{slug}.yml`
   - Adds: QA Results section to story file
   - Status: PASS/CONCERNS/FAIL/WAIVED

### Stage 4: Post-Review (After Addressing Issues)

**Update Gate Status After Fixes:**

```bash
# 6. GATE UPDATE (Document final decision)
@qa *gate {reviewed-story}
# Updates: Quality gate with new status
# Use when: After addressing review feedback
# Documents: What was fixed, what was waived
```

### Understanding Gate Decisions

| **Status**   | **Meaning**                                  | **Action Required**     | **Can Proceed?** |
| ------------ | -------------------------------------------- | ----------------------- | ---------------- |
| **PASS**     | All critical requirements met                | None                    | ✅ Yes           |
| **CONCERNS** | Non-critical issues found                    | Team review recommended | ⚠️ With caution  |
| **FAIL**     | Critical issues (security, missing P0 tests) | Must fix                | ❌ No            |
| **WAIVED**   | Issues acknowledged and accepted             | Document reasoning      | ✅ With approval |

### Risk-Based Testing Strategy

The Test Architect uses risk scoring to prioritize testing:

| **Risk Score** | **Calculation**                | **Testing Priority**      | **Gate Impact**          |
| -------------- | ------------------------------ | ------------------------- | ------------------------ |
| **9**          | High probability × High impact | P0 - Must test thoroughly | FAIL if untested         |
| **6**          | Medium-high combinations       | P1 - Should test well     | CONCERNS if gaps         |
| **4**          | Medium combinations            | P1 - Should test          | CONCERNS if notable gaps |
| **2-3**        | Low-medium combinations        | P2 - Nice to have         | Note in review           |
| **1**          | Minimal risk                   | P2 - Minimal              | Note in review           |

### Special Situations & Best Practices

#### High-Risk or Brownfield Stories

```bash
# ALWAYS run this sequence:
@qa *risk {story}    # First - identify dangers
@qa *design {story}  # Second - plan defense
# Then during dev:
@qa *trace {story}   # Verify regression coverage
@qa *nfr {story}     # Check performance impact
# Finally:
@qa *review {story}  # Deep integration analysis
```

#### Complex Integrations

- Run `*trace` multiple times during development
- Focus on integration test coverage
- Use `*nfr` to validate cross-system performance
- Review with extra attention to API contracts

#### Performance-Critical Features

- Run `*nfr` early and often (not just at review)
- Establish performance baselines before changes
- Document acceptable performance degradation
- Consider load testing requirements in `*design`

### Test Quality Standards Enforced

Quinn ensures all tests meet these standards:

- **No Flaky Tests**: Proper async handling, explicit waits
- **No Hard Waits**: Dynamic strategies only (polling, events)
- **Stateless**: Tests run independently and in parallel
- **Self-Cleaning**: Tests manage their own test data
- **Appropriate Levels**: Unit for logic, integration for interactions, E2E for journeys
- **Clear Assertions**: Keep assertions in tests, not buried in helpers

### Documentation & Audit Trail

All Test Architect activities create permanent records:

- **Assessment Reports**: Timestamped analysis in `docs/qa/assessments/`
- **Gate Files**: Decision records in `docs/qa/gates/`
- **Story Updates**: QA Results sections in story files
- **Traceability**: Requirements to test mapping maintained

## Commit Changes and Push

1. **Commit changes**
2. **Push to remote**

## Complete Development Cycle Flow

### The Full Workflow with Test Architect

1. **SM**: Create next story → Review → Approve
2. **QA (Optional)**: Risk assessment (`*risk`) → Test design (`*design`)
3. **Dev**: Implement story → Write tests → Complete
4. **QA (Optional)**: Mid-dev checks (`*trace`, `*nfr`)
5. **Dev**: Mark Ready for Review
6. **QA (Required)**: Review story (`*review`) → Gate decision
7. **Dev (If needed)**: Address issues
8. **QA (If needed)**: Update gate (`*gate`)
9. **Commit**: All changes
10. **Push**: To remote
11. **Continue**: Until all features implemented

### Quick Decision Guide

**Should I run Test Architect commands?**

| **Scenario**             | **Before Dev**                  | **During Dev**               | **After Dev**                |
| ------------------------ | ------------------------------- | ---------------------------- | ---------------------------- |
| **Simple bug fix**       | Optional                        | Optional                     | Required `*review`           |
| **New feature**          | Recommended `*risk`, `*design`  | Optional `*trace`            | Required `*review`           |
| **Brownfield change**    | **Required** `*risk`, `*design` | Recommended `*trace`, `*nfr` | Required `*review`           |
| **API modification**     | **Required** `*risk`, `*design` | **Required** `*trace`        | Required `*review`           |
| **Performance-critical** | Recommended `*design`           | **Required** `*nfr`          | Required `*review`           |
| **Data migration**       | **Required** `*risk`, `*design` | **Required** `*trace`        | Required `*review` + `*gate` |

### Success Metrics

The Test Architect helps achieve:

- **Zero regression defects** in production
- **100% requirements coverage** with tests
- **Clear quality gates** for go/no-go decisions
- **Documented risk acceptance** for technical debt
- **Consistent test quality** across the team
- **Shift-left testing** with early risk identification
