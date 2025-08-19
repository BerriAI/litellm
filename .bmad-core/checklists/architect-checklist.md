<!-- Powered by BMAD™ Core -->

# Architect Solution Validation Checklist

This checklist serves as a comprehensive framework for the Architect to validate the technical design and architecture before development execution. The Architect should systematically work through each item, ensuring the architecture is robust, scalable, secure, and aligned with the product requirements.

[[LLM: INITIALIZATION INSTRUCTIONS - REQUIRED ARTIFACTS

Before proceeding with this checklist, ensure you have access to:

1. architecture.md - The primary architecture document (check docs/architecture.md)
2. prd.md - Product Requirements Document for requirements alignment (check docs/prd.md)
3. frontend-architecture.md or fe-architecture.md - If this is a UI project (check docs/frontend-architecture.md)
4. Any system diagrams referenced in the architecture
5. API documentation if available
6. Technology stack details and version specifications

IMPORTANT: If any required documents are missing or inaccessible, immediately ask the user for their location or content before proceeding.

PROJECT TYPE DETECTION:
First, determine the project type by checking:

- Does the architecture include a frontend/UI component?
- Is there a frontend-architecture.md document?
- Does the PRD mention user interfaces or frontend requirements?

If this is a backend-only or service-only project:

- Skip sections marked with [[FRONTEND ONLY]]
- Focus extra attention on API design, service architecture, and integration patterns
- Note in your final report that frontend sections were skipped due to project type

VALIDATION APPROACH:
For each section, you must:

1. Deep Analysis - Don't just check boxes, thoroughly analyze each item against the provided documentation
2. Evidence-Based - Cite specific sections or quotes from the documents when validating
3. Critical Thinking - Question assumptions and identify gaps, not just confirm what's present
4. Risk Assessment - Consider what could go wrong with each architectural decision

EXECUTION MODE:
Ask the user if they want to work through the checklist:

- Section by section (interactive mode) - Review each section, present findings, get confirmation before proceeding
- All at once (comprehensive mode) - Complete full analysis and present comprehensive report at end]]

## 1. REQUIREMENTS ALIGNMENT

[[LLM: Before evaluating this section, take a moment to fully understand the product's purpose and goals from the PRD. What is the core problem being solved? Who are the users? What are the critical success factors? Keep these in mind as you validate alignment. For each item, don't just check if it's mentioned - verify that the architecture provides a concrete technical solution.]]

### 1.1 Functional Requirements Coverage

- [ ] Architecture supports all functional requirements in the PRD
- [ ] Technical approaches for all epics and stories are addressed
- [ ] Edge cases and performance scenarios are considered
- [ ] All required integrations are accounted for
- [ ] User journeys are supported by the technical architecture

### 1.2 Non-Functional Requirements Alignment

- [ ] Performance requirements are addressed with specific solutions
- [ ] Scalability considerations are documented with approach
- [ ] Security requirements have corresponding technical controls
- [ ] Reliability and resilience approaches are defined
- [ ] Compliance requirements have technical implementations

### 1.3 Technical Constraints Adherence

- [ ] All technical constraints from PRD are satisfied
- [ ] Platform/language requirements are followed
- [ ] Infrastructure constraints are accommodated
- [ ] Third-party service constraints are addressed
- [ ] Organizational technical standards are followed

## 2. ARCHITECTURE FUNDAMENTALS

[[LLM: Architecture clarity is crucial for successful implementation. As you review this section, visualize the system as if you were explaining it to a new developer. Are there any ambiguities that could lead to misinterpretation? Would an AI agent be able to implement this architecture without confusion? Look for specific diagrams, component definitions, and clear interaction patterns.]]

### 2.1 Architecture Clarity

- [ ] Architecture is documented with clear diagrams
- [ ] Major components and their responsibilities are defined
- [ ] Component interactions and dependencies are mapped
- [ ] Data flows are clearly illustrated
- [ ] Technology choices for each component are specified

### 2.2 Separation of Concerns

- [ ] Clear boundaries between UI, business logic, and data layers
- [ ] Responsibilities are cleanly divided between components
- [ ] Interfaces between components are well-defined
- [ ] Components adhere to single responsibility principle
- [ ] Cross-cutting concerns (logging, auth, etc.) are properly addressed

### 2.3 Design Patterns & Best Practices

- [ ] Appropriate design patterns are employed
- [ ] Industry best practices are followed
- [ ] Anti-patterns are avoided
- [ ] Consistent architectural style throughout
- [ ] Pattern usage is documented and explained

### 2.4 Modularity & Maintainability

- [ ] System is divided into cohesive, loosely-coupled modules
- [ ] Components can be developed and tested independently
- [ ] Changes can be localized to specific components
- [ ] Code organization promotes discoverability
- [ ] Architecture specifically designed for AI agent implementation

## 3. TECHNICAL STACK & DECISIONS

[[LLM: Technology choices have long-term implications. For each technology decision, consider: Is this the simplest solution that could work? Are we over-engineering? Will this scale? What are the maintenance implications? Are there security vulnerabilities in the chosen versions? Verify that specific versions are defined, not ranges.]]

### 3.1 Technology Selection

- [ ] Selected technologies meet all requirements
- [ ] Technology versions are specifically defined (not ranges)
- [ ] Technology choices are justified with clear rationale
- [ ] Alternatives considered are documented with pros/cons
- [ ] Selected stack components work well together

### 3.2 Frontend Architecture [[FRONTEND ONLY]]

[[LLM: Skip this entire section if this is a backend-only or service-only project. Only evaluate if the project includes a user interface.]]

- [ ] UI framework and libraries are specifically selected
- [ ] State management approach is defined
- [ ] Component structure and organization is specified
- [ ] Responsive/adaptive design approach is outlined
- [ ] Build and bundling strategy is determined

### 3.3 Backend Architecture

- [ ] API design and standards are defined
- [ ] Service organization and boundaries are clear
- [ ] Authentication and authorization approach is specified
- [ ] Error handling strategy is outlined
- [ ] Backend scaling approach is defined

### 3.4 Data Architecture

- [ ] Data models are fully defined
- [ ] Database technologies are selected with justification
- [ ] Data access patterns are documented
- [ ] Data migration/seeding approach is specified
- [ ] Data backup and recovery strategies are outlined

## 4. FRONTEND DESIGN & IMPLEMENTATION [[FRONTEND ONLY]]

[[LLM: This entire section should be skipped for backend-only projects. Only evaluate if the project includes a user interface. When evaluating, ensure alignment between the main architecture document and the frontend-specific architecture document.]]

### 4.1 Frontend Philosophy & Patterns

- [ ] Framework & Core Libraries align with main architecture document
- [ ] Component Architecture (e.g., Atomic Design) is clearly described
- [ ] State Management Strategy is appropriate for application complexity
- [ ] Data Flow patterns are consistent and clear
- [ ] Styling Approach is defined and tooling specified

### 4.2 Frontend Structure & Organization

- [ ] Directory structure is clearly documented with ASCII diagram
- [ ] Component organization follows stated patterns
- [ ] File naming conventions are explicit
- [ ] Structure supports chosen framework's best practices
- [ ] Clear guidance on where new components should be placed

### 4.3 Component Design

- [ ] Component template/specification format is defined
- [ ] Component props, state, and events are well-documented
- [ ] Shared/foundational components are identified
- [ ] Component reusability patterns are established
- [ ] Accessibility requirements are built into component design

### 4.4 Frontend-Backend Integration

- [ ] API interaction layer is clearly defined
- [ ] HTTP client setup and configuration documented
- [ ] Error handling for API calls is comprehensive
- [ ] Service definitions follow consistent patterns
- [ ] Authentication integration with backend is clear

### 4.5 Routing & Navigation

- [ ] Routing strategy and library are specified
- [ ] Route definitions table is comprehensive
- [ ] Route protection mechanisms are defined
- [ ] Deep linking considerations addressed
- [ ] Navigation patterns are consistent

### 4.6 Frontend Performance

- [ ] Image optimization strategies defined
- [ ] Code splitting approach documented
- [ ] Lazy loading patterns established
- [ ] Re-render optimization techniques specified
- [ ] Performance monitoring approach defined

## 5. RESILIENCE & OPERATIONAL READINESS

[[LLM: Production systems fail in unexpected ways. As you review this section, think about Murphy's Law - what could go wrong? Consider real-world scenarios: What happens during peak load? How does the system behave when a critical service is down? Can the operations team diagnose issues at 3 AM? Look for specific resilience patterns, not just mentions of "error handling".]]

### 5.1 Error Handling & Resilience

- [ ] Error handling strategy is comprehensive
- [ ] Retry policies are defined where appropriate
- [ ] Circuit breakers or fallbacks are specified for critical services
- [ ] Graceful degradation approaches are defined
- [ ] System can recover from partial failures

### 5.2 Monitoring & Observability

- [ ] Logging strategy is defined
- [ ] Monitoring approach is specified
- [ ] Key metrics for system health are identified
- [ ] Alerting thresholds and strategies are outlined
- [ ] Debugging and troubleshooting capabilities are built in

### 5.3 Performance & Scaling

- [ ] Performance bottlenecks are identified and addressed
- [ ] Caching strategy is defined where appropriate
- [ ] Load balancing approach is specified
- [ ] Horizontal and vertical scaling strategies are outlined
- [ ] Resource sizing recommendations are provided

### 5.4 Deployment & DevOps

- [ ] Deployment strategy is defined
- [ ] CI/CD pipeline approach is outlined
- [ ] Environment strategy (dev, staging, prod) is specified
- [ ] Infrastructure as Code approach is defined
- [ ] Rollback and recovery procedures are outlined

## 6. SECURITY & COMPLIANCE

[[LLM: Security is not optional. Review this section with a hacker's mindset - how could someone exploit this system? Also consider compliance: Are there industry-specific regulations that apply? GDPR? HIPAA? PCI? Ensure the architecture addresses these proactively. Look for specific security controls, not just general statements.]]

### 6.1 Authentication & Authorization

- [ ] Authentication mechanism is clearly defined
- [ ] Authorization model is specified
- [ ] Role-based access control is outlined if required
- [ ] Session management approach is defined
- [ ] Credential management is addressed

### 6.2 Data Security

- [ ] Data encryption approach (at rest and in transit) is specified
- [ ] Sensitive data handling procedures are defined
- [ ] Data retention and purging policies are outlined
- [ ] Backup encryption is addressed if required
- [ ] Data access audit trails are specified if required

### 6.3 API & Service Security

- [ ] API security controls are defined
- [ ] Rate limiting and throttling approaches are specified
- [ ] Input validation strategy is outlined
- [ ] CSRF/XSS prevention measures are addressed
- [ ] Secure communication protocols are specified

### 6.4 Infrastructure Security

- [ ] Network security design is outlined
- [ ] Firewall and security group configurations are specified
- [ ] Service isolation approach is defined
- [ ] Least privilege principle is applied
- [ ] Security monitoring strategy is outlined

## 7. IMPLEMENTATION GUIDANCE

[[LLM: Clear implementation guidance prevents costly mistakes. As you review this section, imagine you're a developer starting on day one. Do they have everything they need to be productive? Are coding standards clear enough to maintain consistency across the team? Look for specific examples and patterns.]]

### 7.1 Coding Standards & Practices

- [ ] Coding standards are defined
- [ ] Documentation requirements are specified
- [ ] Testing expectations are outlined
- [ ] Code organization principles are defined
- [ ] Naming conventions are specified

### 7.2 Testing Strategy

- [ ] Unit testing approach is defined
- [ ] Integration testing strategy is outlined
- [ ] E2E testing approach is specified
- [ ] Performance testing requirements are outlined
- [ ] Security testing approach is defined

### 7.3 Frontend Testing [[FRONTEND ONLY]]

[[LLM: Skip this subsection for backend-only projects.]]

- [ ] Component testing scope and tools defined
- [ ] UI integration testing approach specified
- [ ] Visual regression testing considered
- [ ] Accessibility testing tools identified
- [ ] Frontend-specific test data management addressed

### 7.4 Development Environment

- [ ] Local development environment setup is documented
- [ ] Required tools and configurations are specified
- [ ] Development workflows are outlined
- [ ] Source control practices are defined
- [ ] Dependency management approach is specified

### 7.5 Technical Documentation

- [ ] API documentation standards are defined
- [ ] Architecture documentation requirements are specified
- [ ] Code documentation expectations are outlined
- [ ] System diagrams and visualizations are included
- [ ] Decision records for key choices are included

## 8. DEPENDENCY & INTEGRATION MANAGEMENT

[[LLM: Dependencies are often the source of production issues. For each dependency, consider: What happens if it's unavailable? Is there a newer version with security patches? Are we locked into a vendor? What's our contingency plan? Verify specific versions and fallback strategies.]]

### 8.1 External Dependencies

- [ ] All external dependencies are identified
- [ ] Versioning strategy for dependencies is defined
- [ ] Fallback approaches for critical dependencies are specified
- [ ] Licensing implications are addressed
- [ ] Update and patching strategy is outlined

### 8.2 Internal Dependencies

- [ ] Component dependencies are clearly mapped
- [ ] Build order dependencies are addressed
- [ ] Shared services and utilities are identified
- [ ] Circular dependencies are eliminated
- [ ] Versioning strategy for internal components is defined

### 8.3 Third-Party Integrations

- [ ] All third-party integrations are identified
- [ ] Integration approaches are defined
- [ ] Authentication with third parties is addressed
- [ ] Error handling for integration failures is specified
- [ ] Rate limits and quotas are considered

## 9. AI AGENT IMPLEMENTATION SUITABILITY

[[LLM: This architecture may be implemented by AI agents. Review with extreme clarity in mind. Are patterns consistent? Is complexity minimized? Would an AI agent make incorrect assumptions? Remember: explicit is better than implicit. Look for clear file structures, naming conventions, and implementation patterns.]]

### 9.1 Modularity for AI Agents

- [ ] Components are sized appropriately for AI agent implementation
- [ ] Dependencies between components are minimized
- [ ] Clear interfaces between components are defined
- [ ] Components have singular, well-defined responsibilities
- [ ] File and code organization optimized for AI agent understanding

### 9.2 Clarity & Predictability

- [ ] Patterns are consistent and predictable
- [ ] Complex logic is broken down into simpler steps
- [ ] Architecture avoids overly clever or obscure approaches
- [ ] Examples are provided for unfamiliar patterns
- [ ] Component responsibilities are explicit and clear

### 9.3 Implementation Guidance

- [ ] Detailed implementation guidance is provided
- [ ] Code structure templates are defined
- [ ] Specific implementation patterns are documented
- [ ] Common pitfalls are identified with solutions
- [ ] References to similar implementations are provided when helpful

### 9.4 Error Prevention & Handling

- [ ] Design reduces opportunities for implementation errors
- [ ] Validation and error checking approaches are defined
- [ ] Self-healing mechanisms are incorporated where possible
- [ ] Testing patterns are clearly defined
- [ ] Debugging guidance is provided

## 10. ACCESSIBILITY IMPLEMENTATION [[FRONTEND ONLY]]

[[LLM: Skip this section for backend-only projects. Accessibility is a core requirement for any user interface.]]

### 10.1 Accessibility Standards

- [ ] Semantic HTML usage is emphasized
- [ ] ARIA implementation guidelines provided
- [ ] Keyboard navigation requirements defined
- [ ] Focus management approach specified
- [ ] Screen reader compatibility addressed

### 10.2 Accessibility Testing

- [ ] Accessibility testing tools identified
- [ ] Testing process integrated into workflow
- [ ] Compliance targets (WCAG level) specified
- [ ] Manual testing procedures defined
- [ ] Automated testing approach outlined

[[LLM: FINAL VALIDATION REPORT GENERATION

Now that you've completed the checklist, generate a comprehensive validation report that includes:

1. Executive Summary
   - Overall architecture readiness (High/Medium/Low)
   - Critical risks identified
   - Key strengths of the architecture
   - Project type (Full-stack/Frontend/Backend) and sections evaluated

2. Section Analysis
   - Pass rate for each major section (percentage of items passed)
   - Most concerning failures or gaps
   - Sections requiring immediate attention
   - Note any sections skipped due to project type

3. Risk Assessment
   - Top 5 risks by severity
   - Mitigation recommendations for each
   - Timeline impact of addressing issues

4. Recommendations
   - Must-fix items before development
   - Should-fix items for better quality
   - Nice-to-have improvements

5. AI Implementation Readiness
   - Specific concerns for AI agent implementation
   - Areas needing additional clarification
   - Complexity hotspots to address

6. Frontend-Specific Assessment (if applicable)
   - Frontend architecture completeness
   - Alignment between main and frontend architecture docs
   - UI/UX specification coverage
   - Component design clarity

After presenting the report, ask the user if they would like detailed analysis of any specific section, especially those with warnings or failures.]]
