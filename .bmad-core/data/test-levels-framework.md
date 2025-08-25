<!-- Powered by BMAD™ Core -->

# Test Levels Framework

Comprehensive guide for determining appropriate test levels (unit, integration, E2E) for different scenarios.

## Test Level Decision Matrix

### Unit Tests

**When to use:**

- Testing pure functions and business logic
- Algorithm correctness
- Input validation and data transformation
- Error handling in isolated components
- Complex calculations or state machines

**Characteristics:**

- Fast execution (immediate feedback)
- No external dependencies (DB, API, file system)
- Highly maintainable and stable
- Easy to debug failures

**Example scenarios:**

```yaml
unit_test:
  component: 'PriceCalculator'
  scenario: 'Calculate discount with multiple rules'
  justification: 'Complex business logic with multiple branches'
  mock_requirements: 'None - pure function'
```

### Integration Tests

**When to use:**

- Component interaction verification
- Database operations and transactions
- API endpoint contracts
- Service-to-service communication
- Middleware and interceptor behavior

**Characteristics:**

- Moderate execution time
- Tests component boundaries
- May use test databases or containers
- Validates system integration points

**Example scenarios:**

```yaml
integration_test:
  components: ['UserService', 'AuthRepository']
  scenario: 'Create user with role assignment'
  justification: 'Critical data flow between service and persistence'
  test_environment: 'In-memory database'
```

### End-to-End Tests

**When to use:**

- Critical user journeys
- Cross-system workflows
- Visual regression testing
- Compliance and regulatory requirements
- Final validation before release

**Characteristics:**

- Slower execution
- Tests complete workflows
- Requires full environment setup
- Most realistic but most brittle

**Example scenarios:**

```yaml
e2e_test:
  journey: 'Complete checkout process'
  scenario: 'User purchases with saved payment method'
  justification: 'Revenue-critical path requiring full validation'
  environment: 'Staging with test payment gateway'
```

## Test Level Selection Rules

### Favor Unit Tests When:

- Logic can be isolated
- No side effects involved
- Fast feedback needed
- High cyclomatic complexity

### Favor Integration Tests When:

- Testing persistence layer
- Validating service contracts
- Testing middleware/interceptors
- Component boundaries critical

### Favor E2E Tests When:

- User-facing critical paths
- Multi-system interactions
- Regulatory compliance scenarios
- Visual regression important

## Anti-patterns to Avoid

- E2E testing for business logic validation
- Unit testing framework behavior
- Integration testing third-party libraries
- Duplicate coverage across levels

## Duplicate Coverage Guard

**Before adding any test, check:**

1. Is this already tested at a lower level?
2. Can a unit test cover this instead of integration?
3. Can an integration test cover this instead of E2E?

**Coverage overlap is only acceptable when:**

- Testing different aspects (unit: logic, integration: interaction, e2e: user experience)
- Critical paths requiring defense in depth
- Regression prevention for previously broken functionality

## Test Naming Conventions

- Unit: `test_{component}_{scenario}`
- Integration: `test_{flow}_{interaction}`
- E2E: `test_{journey}_{outcome}`

## Test ID Format

`{EPIC}.{STORY}-{LEVEL}-{SEQ}`

Examples:

- `1.3-UNIT-001`
- `1.3-INT-002`
- `1.3-E2E-001`
