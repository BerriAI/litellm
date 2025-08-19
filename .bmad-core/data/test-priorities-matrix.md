<!-- Powered by BMAD™ Core -->

# Test Priorities Matrix

Guide for prioritizing test scenarios based on risk, criticality, and business impact.

## Priority Levels

### P0 - Critical (Must Test)

**Criteria:**

- Revenue-impacting functionality
- Security-critical paths
- Data integrity operations
- Regulatory compliance requirements
- Previously broken functionality (regression prevention)

**Examples:**

- Payment processing
- Authentication/authorization
- User data creation/deletion
- Financial calculations
- GDPR/privacy compliance

**Testing Requirements:**

- Comprehensive coverage at all levels
- Both happy and unhappy paths
- Edge cases and error scenarios
- Performance under load

### P1 - High (Should Test)

**Criteria:**

- Core user journeys
- Frequently used features
- Features with complex logic
- Integration points between systems
- Features affecting user experience

**Examples:**

- User registration flow
- Search functionality
- Data import/export
- Notification systems
- Dashboard displays

**Testing Requirements:**

- Primary happy paths required
- Key error scenarios
- Critical edge cases
- Basic performance validation

### P2 - Medium (Nice to Test)

**Criteria:**

- Secondary features
- Admin functionality
- Reporting features
- Configuration options
- UI polish and aesthetics

**Examples:**

- Admin settings panels
- Report generation
- Theme customization
- Help documentation
- Analytics tracking

**Testing Requirements:**

- Happy path coverage
- Basic error handling
- Can defer edge cases

### P3 - Low (Test if Time Permits)

**Criteria:**

- Rarely used features
- Nice-to-have functionality
- Cosmetic issues
- Non-critical optimizations

**Examples:**

- Advanced preferences
- Legacy feature support
- Experimental features
- Debug utilities

**Testing Requirements:**

- Smoke tests only
- Can rely on manual testing
- Document known limitations

## Risk-Based Priority Adjustments

### Increase Priority When:

- High user impact (affects >50% of users)
- High financial impact (>$10K potential loss)
- Security vulnerability potential
- Compliance/legal requirements
- Customer-reported issues
- Complex implementation (>500 LOC)
- Multiple system dependencies

### Decrease Priority When:

- Feature flag protected
- Gradual rollout planned
- Strong monitoring in place
- Easy rollback capability
- Low usage metrics
- Simple implementation
- Well-isolated component

## Test Coverage by Priority

| Priority | Unit Coverage | Integration Coverage | E2E Coverage       |
| -------- | ------------- | -------------------- | ------------------ |
| P0       | >90%          | >80%                 | All critical paths |
| P1       | >80%          | >60%                 | Main happy paths   |
| P2       | >60%          | >40%                 | Smoke tests        |
| P3       | Best effort   | Best effort          | Manual only        |

## Priority Assignment Rules

1. **Start with business impact** - What happens if this fails?
2. **Consider probability** - How likely is failure?
3. **Factor in detectability** - Would we know if it failed?
4. **Account for recoverability** - Can we fix it quickly?

## Priority Decision Tree

```
Is it revenue-critical?
├─ YES → P0
└─ NO → Does it affect core user journey?
    ├─ YES → Is it high-risk?
    │   ├─ YES → P0
    │   └─ NO → P1
    └─ NO → Is it frequently used?
        ├─ YES → P1
        └─ NO → Is it customer-facing?
            ├─ YES → P2
            └─ NO → P3
```

## Test Execution Order

1. Execute P0 tests first (fail fast on critical issues)
2. Execute P1 tests second (core functionality)
3. Execute P2 tests if time permits
4. P3 tests only in full regression cycles

## Continuous Adjustment

Review and adjust priorities based on:

- Production incident patterns
- User feedback and complaints
- Usage analytics
- Test failure history
- Business priority changes
