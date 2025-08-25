# Working in the Brownfield: A Complete Guide

> **HIGHLY RECOMMENDED: Use Gemini Web or Gemini CLI for Brownfield Documentation Generation!**
>
> Gemini Web's 1M+ token context window or Gemini CLI (when it's working) can analyze your ENTIRE codebase, or critical sections of it, all at once (obviously within reason):
>
> - Upload via GitHub URL or use gemini cli in the project folder
> - If working in the web: use `npx bmad-method flatten` to flatten your project into a single file, then upload that file to your web agent.

## What is Brownfield Development?

Brownfield development refers to adding features, fixing bugs, or modernizing existing software projects. Unlike greenfield (new) projects, brownfield work requires understanding existing code, respecting constraints, and ensuring new changes integrate seamlessly without breaking existing functionality.

## When to Use BMad for Brownfield

- Add significant new features to existing applications
- Modernize legacy codebases
- Integrate new technologies or services
- Refactor complex systems
- Fix bugs that require architectural understanding
- Document undocumented systems

## When NOT to use a Brownfield Flow

If you have just completed an MVP with BMad, and you want to continue with post-MVP, its easier to just talk to the PM and ask it to work with you to create a new epic to add into the PRD, shard out the epic, update any architecture documents with the architect, and just go from there.

## The Complete Brownfield Workflow

1. **Follow the [<ins>User Guide - Installation</ins>](user-guide.md#installation) steps to setup your agent in the web.**
2. **Generate a 'flattened' single file of your entire codebase** run: `npx bmad-method flatten`

### Choose Your Approach

#### Approach A: PRD-First (Recommended if adding very large and complex new features, single or multiple epics or massive changes)

**Best for**: Large codebases, monorepos, or when you know exactly what you want to build

1. **Create PRD First** to define requirements
2. **Document only relevant areas** based on PRD needs
3. **More efficient** - avoids documenting unused code

#### Approach B: Document-First (Good for Smaller Projects)

**Best for**: Smaller codebases, unknown systems, or exploratory changes

1. **Document entire system** first
2. **Create PRD** with full context
3. **More thorough** - captures everything

### Approach A: PRD-First Workflow (Recommended)

#### Phase 1: Define Requirements First

**In Gemini Web (with your flattened-codebase.xml uploaded):**

```bash
@pm
*create-brownfield-prd
```

The PM will:

- **Ask about your enhancement** requirements
- **Explore the codebase** to understand current state
- **Identify affected areas** that need documentation
- **Create focused PRD** with clear scope

**Key Advantage**: The PRD identifies which parts of your monorepo/large codebase actually need documentation!

#### Phase 2: Focused Documentation

**Still in Gemini Web, now with PRD context:**

```bash
@architect
*document-project
```

The architect will:

- **Ask about your focus** if no PRD was provided
- **Offer options**: Create PRD, provide requirements, or describe the enhancement
- **Reference the PRD/description** to understand scope
- **Focus on relevant modules** identified in PRD or your description
- **Skip unrelated areas** to keep docs lean
- **Generate ONE architecture document** for all environments

The architect creates:

- **One comprehensive architecture document** following fullstack-architecture template
- **Covers all system aspects** in a single file
- **Easy to copy and save** as `docs/architecture.md`
- **Can be sharded later** in IDE if desired

For example, if you say "Add payment processing to user service":

- Documents only: user service, API endpoints, database schemas, payment integrations
- Creates focused source tree showing only payment-related code paths
- Skips: admin panels, reporting modules, unrelated microservices

### Approach B: Document-First Workflow

#### Phase 1: Document the Existing System

**Best Approach - Gemini Web with 1M+ Context**:

1. **Go to Gemini Web** (gemini.google.com)
2. **Upload your project**:
   - **Option A**: Paste your GitHub repository URL directly
   - **Option B**: Upload your flattened-codebase.xml file
3. **Load the architect agent**: Upload `dist/agents/architect.txt`
4. **Run documentation**: Type `*document-project`

The architect will generate comprehensive documentation of everything.

#### Phase 2: Plan Your Enhancement

##### Option A: Full Brownfield Workflow (Recommended for Major Changes)

**1. Create Brownfield PRD**:

```bash
@pm
*create-brownfield-prd
```

The PM agent will:

- **Analyze existing documentation** from Phase 1
- **Request specific enhancement details** from you
- **Assess complexity** and recommend approach
- **Create epic/story structure** for the enhancement
- **Identify risks and integration points**

**How PM Agent Gets Project Context**:

- In Gemini Web: Already has full project context from Phase 1 documentation
- In IDE: Will ask "Please provide the path to your existing project documentation"

**Key Prompts You'll Encounter**:

- "What specific enhancement or feature do you want to add?"
- "Are there any existing systems or APIs this needs to integrate with?"
- "What are the critical constraints we must respect?"
- "What is your timeline and team size?"

**2. Create Brownfield Architecture**:

```bash
@architect
*create-brownfield-architecture
```

The architect will:

- **Review the brownfield PRD**
- **Design integration strategy**
- **Plan migration approach** if needed
- **Identify technical risks**
- **Define compatibility requirements**

##### Option B: Quick Enhancement (For Focused Changes)

**For Single Epic Without Full PRD**:

```bash
@pm
*create-brownfield-epic
```

Use when:

- Enhancement is well-defined and isolated
- Existing documentation is comprehensive
- Changes don't impact multiple systems
- You need quick turnaround

**For Single Story**:

```bash
@pm
*create-brownfield-story
```

Use when:

- Bug fix or tiny feature
- Very isolated change
- No architectural impact
- Clear implementation path

### Phase 3: Validate Planning Artifacts

```bash
@po
*execute-checklist-po
```

The PO ensures:

- Compatibility with existing system
- No breaking changes planned
- Risk mitigation strategies in place
- Clear integration approach

### Phase 4: Save and Shard Documents

1. Save your PRD and Architecture as:
   docs/prd.md
   docs/architecture.md
   (Note: You can optionally prefix with 'brownfield-' if managing multiple versions)
2. Shard your docs:
   In your IDE

   ```bash
   @po
   shard docs/prd.md
   ```

   ```bash
   @po
   shard docs/architecture.md
   ```

### Phase 5: Transition to Development

**Follow the [<ins>Enhanced IDE Development Workflow</ins>](enhanced-ide-development-workflow.md)**

## Brownfield Best Practices

### 1. Always Document First

Even if you think you know the codebase:

- Run `document-project` to capture current state
- AI agents need this context
- Discovers undocumented patterns

### 2. Respect Existing Patterns

The brownfield templates specifically look for:

- Current coding conventions
- Existing architectural patterns
- Technology constraints
- Team preferences

### 3. Plan for Gradual Rollout

Brownfield changes should:

- Support feature flags
- Plan rollback strategies
- Include migration scripts
- Maintain backwards compatibility

### 4. Test Integration Thoroughly

#### Why the Test Architect is Critical for Brownfield

In brownfield projects, the Test Architect (Quinn) becomes your safety net against breaking existing functionality. Unlike greenfield where you're building fresh, brownfield requires careful validation that new changes don't destabilize what already works.

#### Brownfield-Specific Testing Challenges

The Test Architect addresses unique brownfield complexities:

| **Challenge**               | **How Test Architect Helps**                      | **Command**         |
| --------------------------- | ------------------------------------------------- | ------------------- |
| **Regression Risks**        | Identifies which existing features might break    | `*risk`             |
| **Legacy Dependencies**     | Maps integration points and hidden dependencies   | `*trace`            |
| **Performance Degradation** | Validates no slowdown in existing flows           | `*nfr`              |
| **Coverage Gaps**           | Finds untested legacy code that new changes touch | `*design`           |
| **Breaking Changes**        | Detects API/contract violations                   | `*review`           |
| **Migration Safety**        | Validates data transformations and rollback plans | `*risk` + `*review` |

#### Complete Test Architect Workflow for Brownfield

##### Stage 1: Before Development (Risk & Strategy)

**CRITICAL FOR BROWNFIELD - Run These First:**

```bash
# 1. RISK ASSESSMENT (Run IMMEDIATELY after story creation)
@qa *risk {brownfield-story}
# Identifies: Legacy dependencies, breaking changes, integration points
# Output: docs/qa/assessments/{epic}.{story}-risk-{YYYYMMDD}.md
# Brownfield Focus:
#   - Regression probability scoring
#   - Affected downstream systems
#   - Data migration risks
#   - Rollback complexity

# 2. TEST DESIGN (After risk assessment)
@qa *design {brownfield-story}
# Creates: Regression test strategy + new feature tests
# Output: docs/qa/assessments/{epic}.{story}-test-design-{YYYYMMDD}.md
# Brownfield Focus:
#   - Existing functionality that needs regression tests
#   - Integration test requirements
#   - Performance benchmarks to maintain
#   - Feature flag test scenarios
```

##### Stage 2: During Development (Continuous Validation)

**Monitor Integration Health While Coding:**

```bash
# 3. REQUIREMENTS TRACING (Mid-development checkpoint)
@qa *trace {brownfield-story}
# Maps: New requirements + existing functionality preservation
# Output: docs/qa/assessments/{epic}.{story}-trace-{YYYYMMDD}.md
# Brownfield Focus:
#   - Existing features that must still work
#   - New/old feature interactions
#   - API contract preservation
#   - Missing regression test coverage

# 4. NFR VALIDATION (Before considering "done")
@qa *nfr {brownfield-story}
# Validates: Performance, security, reliability unchanged
# Output: docs/qa/assessments/{epic}.{story}-nfr-{YYYYMMDD}.md
# Brownfield Focus:
#   - Performance regression detection
#   - Security implications of integrations
#   - Backward compatibility validation
#   - Load/stress on legacy components
```

##### Stage 3: Code Review (Deep Integration Analysis)

**Comprehensive Brownfield Review:**

```bash
# 5. FULL REVIEW (When development complete)
@qa *review {brownfield-story}
# Performs: Deep analysis + active refactoring
# Outputs:
#   - QA Results in story file
#   - Gate file: docs/qa/gates/{epic}.{story}-{slug}.yml
```

The review specifically analyzes:

- **API Breaking Changes**: Validates all existing contracts maintained
- **Data Migration Safety**: Checks transformation logic and rollback procedures
- **Performance Regression**: Compares against baseline metrics
- **Integration Points**: Validates all touchpoints with legacy code
- **Feature Flag Logic**: Ensures proper toggle behavior
- **Dependency Impacts**: Maps affected downstream systems

##### Stage 4: Post-Review (Gate Updates)

```bash
# 6. GATE STATUS UPDATE (After addressing issues)
@qa *gate {brownfield-story}
# Updates: Quality gate decision after fixes
# Output: docs/qa/gates/{epic}.{story}-{slug}.yml
# Brownfield Considerations:
#   - May WAIVE certain legacy code issues
#   - Documents technical debt acceptance
#   - Tracks migration progress
```

#### Brownfield-Specific Risk Scoring

The Test Architect uses enhanced risk scoring for brownfield:

| **Risk Category**      | **Brownfield Factors**                     | **Impact on Gate**  |
| ---------------------- | ------------------------------------------ | ------------------- |
| **Regression Risk**    | Number of integration points × Age of code | Score ≥9 = FAIL     |
| **Data Risk**          | Migration complexity × Data volume         | Score ≥6 = CONCERNS |
| **Performance Risk**   | Current load × Added complexity            | Score ≥6 = CONCERNS |
| **Compatibility Risk** | API consumers × Contract changes           | Score ≥9 = FAIL     |

#### Brownfield Testing Standards

Quinn enforces additional standards for brownfield:

- **Regression Test Coverage**: Every touched legacy module needs tests
- **Performance Baselines**: Must maintain or improve current metrics
- **Rollback Procedures**: Every change needs a rollback plan
- **Feature Flags**: All risky changes behind toggles
- **Integration Tests**: Cover all legacy touchpoints
- **Contract Tests**: Validate API compatibility
- **Data Validation**: Migration correctness checks

#### Quick Reference: Brownfield Test Commands

| **Scenario**                      | **Commands to Run**                                  | **Order**  | **Why Critical**              |
| --------------------------------- | ---------------------------------------------------- | ---------- | ----------------------------- |
| **Adding Feature to Legacy Code** | `*risk` → `*design` → `*trace` → `*review`           | Sequential | Map all dependencies first    |
| **API Modification**              | `*risk` → `*design` → `*nfr` → `*review`             | Sequential | Prevent breaking consumers    |
| **Performance-Critical Change**   | `*nfr` early and often → `*review`                   | Continuous | Catch degradation immediately |
| **Data Migration**                | `*risk` → `*design` → `*trace` → `*review` → `*gate` | Full cycle | Ensure data integrity         |
| **Bug Fix in Complex System**     | `*risk` → `*trace` → `*review`                       | Focused    | Prevent side effects          |

#### Integration with Brownfield Scenarios

**Scenario-Specific Guidance:**

1. **Legacy Code Modernization**
   - Start with `*risk` to map all dependencies
   - Use `*design` to plan strangler fig approach
   - Run `*trace` frequently to ensure nothing breaks
   - `*review` with focus on gradual migration

2. **Adding Features to Monolith**
   - `*risk` identifies integration complexity
   - `*design` plans isolation strategies
   - `*nfr` monitors performance impact
   - `*review` validates no monolith degradation

3. **Microservice Extraction**
   - `*risk` maps service boundaries
   - `*trace` ensures functionality preservation
   - `*nfr` validates network overhead acceptable
   - `*gate` documents accepted trade-offs

4. **Database Schema Changes**
   - `*risk` assesses migration complexity
   - `*design` plans backward-compatible approach
   - `*trace` maps all affected queries
   - `*review` validates migration safety

### 5. Communicate Changes

Document:

- What changed and why
- Migration instructions
- New patterns introduced
- Deprecation notices

## Common Brownfield Scenarios

### Scenario 1: Adding a New Feature

1. Document existing system
2. Create brownfield PRD focusing on integration
3. **Test Architect Early Involvement**:
   - Run `@qa *risk` on draft stories to identify integration risks
   - Use `@qa *design` to plan regression test strategy
4. Architecture emphasizes compatibility
5. Stories include integration tasks with test requirements
6. **During Development**:
   - Developer runs `@qa *trace` to verify coverage
   - Use `@qa *nfr` to monitor performance impact
7. **Review Stage**: `@qa *review` validates integration safety

### Scenario 2: Modernizing Legacy Code

1. Extensive documentation phase
2. PRD includes migration strategy
3. **Test Architect Strategy Planning**:
   - `@qa *risk` assesses modernization complexity
   - `@qa *design` plans parallel testing approach
4. Architecture plans gradual transition (strangler fig pattern)
5. Stories follow incremental modernization with:
   - Regression tests for untouched legacy code
   - Integration tests for new/old boundaries
   - Performance benchmarks at each stage
6. **Continuous Validation**: Run `@qa *trace` after each increment
7. **Gate Management**: Use `@qa *gate` to track technical debt acceptance

### Scenario 3: Bug Fix in Complex System

1. Document relevant subsystems
2. Use `create-brownfield-story` for focused fix
3. **Test Architect Risk Assessment**: Run `@qa *risk` to identify side effect potential
4. Include regression test requirements from `@qa *design` output
5. **During Fix**: Use `@qa *trace` to map affected functionality
6. **Before Commit**: Run `@qa *review` for comprehensive validation
7. Test Architect validates no side effects using:
   - Risk profiling for side effect analysis (probability × impact scoring)
   - Trace matrix to ensure fix doesn't break related features
   - NFR assessment to verify performance/security unchanged
   - Gate decision documents fix safety

### Scenario 4: API Integration

1. Document existing API patterns
2. PRD defines integration requirements
3. **Test Architect Contract Analysis**:
   - `@qa *risk` identifies breaking change potential
   - `@qa *design` creates contract test strategy
4. Architecture ensures consistent patterns
5. **API Testing Focus**:
   - Contract tests for backward compatibility
   - Integration tests for new endpoints
   - Performance tests for added load
6. Stories include API documentation updates
7. **Validation Checkpoints**:
   - `@qa *trace` maps all API consumers
   - `@qa *nfr` validates response times
   - `@qa *review` ensures no breaking changes
8. **Gate Decision**: Document any accepted breaking changes with migration path

## Troubleshooting

### "The AI doesn't understand my codebase"

**Solution**: Re-run `document-project` with more specific paths to critical files

### "Generated plans don't fit our patterns"

**Solution**: Update generated documentation with your specific conventions before planning phase

### "Too much boilerplate for small changes"

**Solution**: Use `create-brownfield-story` instead of full workflow

### "Integration points unclear"

**Solution**: Provide more context during PRD creation, specifically highlighting integration systems

## Quick Reference

### Brownfield-Specific Commands

```bash
# Document existing project
@architect *document-project

# Create enhancement PRD
@pm *create-brownfield-prd

# Create architecture with integration focus
@architect *create-brownfield-architecture

# Quick epic creation
@pm *create-brownfield-epic

# Single story creation
@pm *create-brownfield-story
```

### Test Architect Commands for Brownfield

Note: Short forms shown below. Full commands: `*risk-profile`, `*test-design`, `*nfr-assess`, `*trace-requirements`

```bash
# BEFORE DEVELOPMENT (Planning)
@qa *risk {story}     # Assess regression & integration risks
@qa *design {story}   # Plan regression + new feature tests

# DURING DEVELOPMENT (Validation)
@qa *trace {story}    # Verify coverage of old + new
@qa *nfr {story}      # Check performance degradation

# AFTER DEVELOPMENT (Review)
@qa *review {story}   # Deep integration analysis
@qa *gate {story}     # Update quality decision
```

### Decision Tree

```text
Do you have a large codebase or monorepo?
├─ Yes → PRD-First Approach
│   └─ Create PRD → Document only affected areas
└─ No → Is the codebase well-known to you?
    ├─ Yes → PRD-First Approach
    └─ No → Document-First Approach

Is this a major enhancement affecting multiple systems?
├─ Yes → Full Brownfield Workflow
│   └─ ALWAYS run Test Architect *risk + *design first
└─ No → Is this more than a simple bug fix?
    ├─ Yes → *create-brownfield-epic
    │   └─ Run Test Architect *risk for integration points
    └─ No → *create-brownfield-story
        └─ Still run *risk if touching critical paths

Does the change touch legacy code?
├─ Yes → Test Architect is MANDATORY
│   ├─ *risk → Identify regression potential
│   ├─ *design → Plan test coverage
│   └─ *review → Validate no breakage
└─ No → Test Architect is RECOMMENDED
    └─ *review → Ensure quality standards
```

## Conclusion

Brownfield development with BMad Method provides structure and safety when modifying existing systems. The Test Architect becomes your critical safety net, using risk assessment, regression testing, and continuous validation to ensure new changes don't destabilize existing functionality.

**The Brownfield Success Formula:**

1. **Document First** - Understand what exists
2. **Assess Risk Early** - Use Test Architect `*risk` before coding
3. **Plan Test Strategy** - Design regression + new feature tests
4. **Validate Continuously** - Check integration health during development
5. **Review Comprehensively** - Deep analysis before committing
6. **Gate Decisively** - Document quality decisions

Remember: **In brownfield, the Test Architect isn't optional - it's your insurance policy against breaking production.**
