<!-- Powered by BMAD™ Core -->

# Product Owner (PO) Master Validation Checklist

This checklist serves as a comprehensive framework for the Product Owner to validate project plans before development execution. It adapts intelligently based on project type (greenfield vs brownfield) and includes UI/UX considerations when applicable.

[[LLM: INITIALIZATION INSTRUCTIONS - PO MASTER CHECKLIST

PROJECT TYPE DETECTION:
First, determine the project type by checking:

1. Is this a GREENFIELD project (new from scratch)?
   - Look for: New project initialization, no existing codebase references
   - Check for: prd.md, architecture.md, new project setup stories

2. Is this a BROWNFIELD project (enhancing existing system)?
   - Look for: References to existing codebase, enhancement/modification language
   - Check for: brownfield-prd.md, brownfield-architecture.md, existing system analysis

3. Does the project include UI/UX components?
   - Check for: frontend-architecture.md, UI/UX specifications, design files
   - Look for: Frontend stories, component specifications, user interface mentions

DOCUMENT REQUIREMENTS:
Based on project type, ensure you have access to:

For GREENFIELD projects:

- prd.md - The Product Requirements Document
- architecture.md - The system architecture
- frontend-architecture.md - If UI/UX is involved
- All epic and story definitions

For BROWNFIELD projects:

- brownfield-prd.md - The brownfield enhancement requirements
- brownfield-architecture.md - The enhancement architecture
- Existing project codebase access (CRITICAL - cannot proceed without this)
- Current deployment configuration and infrastructure details
- Database schemas, API documentation, monitoring setup

SKIP INSTRUCTIONS:

- Skip sections marked [[BROWNFIELD ONLY]] for greenfield projects
- Skip sections marked [[GREENFIELD ONLY]] for brownfield projects
- Skip sections marked [[UI/UX ONLY]] for backend-only projects
- Note all skipped sections in your final report

VALIDATION APPROACH:

1. Deep Analysis - Thoroughly analyze each item against documentation
2. Evidence-Based - Cite specific sections or code when validating
3. Critical Thinking - Question assumptions and identify gaps
4. Risk Assessment - Consider what could go wrong with each decision

EXECUTION MODE:
Ask the user if they want to work through the checklist:

- Section by section (interactive mode) - Review each section, get confirmation before proceeding
- All at once (comprehensive mode) - Complete full analysis and present report at end]]

## 1. PROJECT SETUP & INITIALIZATION

[[LLM: Project setup is the foundation. For greenfield, ensure clean start. For brownfield, ensure safe integration with existing system. Verify setup matches project type.]]

### 1.1 Project Scaffolding [[GREENFIELD ONLY]]

- [ ] Epic 1 includes explicit steps for project creation/initialization
- [ ] If using a starter template, steps for cloning/setup are included
- [ ] If building from scratch, all necessary scaffolding steps are defined
- [ ] Initial README or documentation setup is included
- [ ] Repository setup and initial commit processes are defined

### 1.2 Existing System Integration [[BROWNFIELD ONLY]]

- [ ] Existing project analysis has been completed and documented
- [ ] Integration points with current system are identified
- [ ] Development environment preserves existing functionality
- [ ] Local testing approach validated for existing features
- [ ] Rollback procedures defined for each integration point

### 1.3 Development Environment

- [ ] Local development environment setup is clearly defined
- [ ] Required tools and versions are specified
- [ ] Steps for installing dependencies are included
- [ ] Configuration files are addressed appropriately
- [ ] Development server setup is included

### 1.4 Core Dependencies

- [ ] All critical packages/libraries are installed early
- [ ] Package management is properly addressed
- [ ] Version specifications are appropriately defined
- [ ] Dependency conflicts or special requirements are noted
- [ ] [[BROWNFIELD ONLY]] Version compatibility with existing stack verified

## 2. INFRASTRUCTURE & DEPLOYMENT

[[LLM: Infrastructure must exist before use. For brownfield, must integrate with existing infrastructure without breaking it.]]

### 2.1 Database & Data Store Setup

- [ ] Database selection/setup occurs before any operations
- [ ] Schema definitions are created before data operations
- [ ] Migration strategies are defined if applicable
- [ ] Seed data or initial data setup is included if needed
- [ ] [[BROWNFIELD ONLY]] Database migration risks identified and mitigated
- [ ] [[BROWNFIELD ONLY]] Backward compatibility ensured

### 2.2 API & Service Configuration

- [ ] API frameworks are set up before implementing endpoints
- [ ] Service architecture is established before implementing services
- [ ] Authentication framework is set up before protected routes
- [ ] Middleware and common utilities are created before use
- [ ] [[BROWNFIELD ONLY]] API compatibility with existing system maintained
- [ ] [[BROWNFIELD ONLY]] Integration with existing authentication preserved

### 2.3 Deployment Pipeline

- [ ] CI/CD pipeline is established before deployment actions
- [ ] Infrastructure as Code (IaC) is set up before use
- [ ] Environment configurations are defined early
- [ ] Deployment strategies are defined before implementation
- [ ] [[BROWNFIELD ONLY]] Deployment minimizes downtime
- [ ] [[BROWNFIELD ONLY]] Blue-green or canary deployment implemented

### 2.4 Testing Infrastructure

- [ ] Testing frameworks are installed before writing tests
- [ ] Test environment setup precedes test implementation
- [ ] Mock services or data are defined before testing
- [ ] [[BROWNFIELD ONLY]] Regression testing covers existing functionality
- [ ] [[BROWNFIELD ONLY]] Integration testing validates new-to-existing connections

## 3. EXTERNAL DEPENDENCIES & INTEGRATIONS

[[LLM: External dependencies often block progress. For brownfield, ensure new dependencies don't conflict with existing ones.]]

### 3.1 Third-Party Services

- [ ] Account creation steps are identified for required services
- [ ] API key acquisition processes are defined
- [ ] Steps for securely storing credentials are included
- [ ] Fallback or offline development options are considered
- [ ] [[BROWNFIELD ONLY]] Compatibility with existing services verified
- [ ] [[BROWNFIELD ONLY]] Impact on existing integrations assessed

### 3.2 External APIs

- [ ] Integration points with external APIs are clearly identified
- [ ] Authentication with external services is properly sequenced
- [ ] API limits or constraints are acknowledged
- [ ] Backup strategies for API failures are considered
- [ ] [[BROWNFIELD ONLY]] Existing API dependencies maintained

### 3.3 Infrastructure Services

- [ ] Cloud resource provisioning is properly sequenced
- [ ] DNS or domain registration needs are identified
- [ ] Email or messaging service setup is included if needed
- [ ] CDN or static asset hosting setup precedes their use
- [ ] [[BROWNFIELD ONLY]] Existing infrastructure services preserved

## 4. UI/UX CONSIDERATIONS [[UI/UX ONLY]]

[[LLM: Only evaluate this section if the project includes user interface components. Skip entirely for backend-only projects.]]

### 4.1 Design System Setup

- [ ] UI framework and libraries are selected and installed early
- [ ] Design system or component library is established
- [ ] Styling approach (CSS modules, styled-components, etc.) is defined
- [ ] Responsive design strategy is established
- [ ] Accessibility requirements are defined upfront

### 4.2 Frontend Infrastructure

- [ ] Frontend build pipeline is configured before development
- [ ] Asset optimization strategy is defined
- [ ] Frontend testing framework is set up
- [ ] Component development workflow is established
- [ ] [[BROWNFIELD ONLY]] UI consistency with existing system maintained

### 4.3 User Experience Flow

- [ ] User journeys are mapped before implementation
- [ ] Navigation patterns are defined early
- [ ] Error states and loading states are planned
- [ ] Form validation patterns are established
- [ ] [[BROWNFIELD ONLY]] Existing user workflows preserved or migrated

## 5. USER/AGENT RESPONSIBILITY

[[LLM: Clear ownership prevents confusion. Ensure tasks are assigned appropriately based on what only humans can do.]]

### 5.1 User Actions

- [ ] User responsibilities limited to human-only tasks
- [ ] Account creation on external services assigned to users
- [ ] Purchasing or payment actions assigned to users
- [ ] Credential provision appropriately assigned to users

### 5.2 Developer Agent Actions

- [ ] All code-related tasks assigned to developer agents
- [ ] Automated processes identified as agent responsibilities
- [ ] Configuration management properly assigned
- [ ] Testing and validation assigned to appropriate agents

## 6. FEATURE SEQUENCING & DEPENDENCIES

[[LLM: Dependencies create the critical path. For brownfield, ensure new features don't break existing ones.]]

### 6.1 Functional Dependencies

- [ ] Features depending on others are sequenced correctly
- [ ] Shared components are built before their use
- [ ] User flows follow logical progression
- [ ] Authentication features precede protected features
- [ ] [[BROWNFIELD ONLY]] Existing functionality preserved throughout

### 6.2 Technical Dependencies

- [ ] Lower-level services built before higher-level ones
- [ ] Libraries and utilities created before their use
- [ ] Data models defined before operations on them
- [ ] API endpoints defined before client consumption
- [ ] [[BROWNFIELD ONLY]] Integration points tested at each step

### 6.3 Cross-Epic Dependencies

- [ ] Later epics build upon earlier epic functionality
- [ ] No epic requires functionality from later epics
- [ ] Infrastructure from early epics utilized consistently
- [ ] Incremental value delivery maintained
- [ ] [[BROWNFIELD ONLY]] Each epic maintains system integrity

## 7. RISK MANAGEMENT [[BROWNFIELD ONLY]]

[[LLM: This section is CRITICAL for brownfield projects. Think pessimistically about what could break.]]

### 7.1 Breaking Change Risks

- [ ] Risk of breaking existing functionality assessed
- [ ] Database migration risks identified and mitigated
- [ ] API breaking change risks evaluated
- [ ] Performance degradation risks identified
- [ ] Security vulnerability risks evaluated

### 7.2 Rollback Strategy

- [ ] Rollback procedures clearly defined per story
- [ ] Feature flag strategy implemented
- [ ] Backup and recovery procedures updated
- [ ] Monitoring enhanced for new components
- [ ] Rollback triggers and thresholds defined

### 7.3 User Impact Mitigation

- [ ] Existing user workflows analyzed for impact
- [ ] User communication plan developed
- [ ] Training materials updated
- [ ] Support documentation comprehensive
- [ ] Migration path for user data validated

## 8. MVP SCOPE ALIGNMENT

[[LLM: MVP means MINIMUM viable product. For brownfield, ensure enhancements are truly necessary.]]

### 8.1 Core Goals Alignment

- [ ] All core goals from PRD are addressed
- [ ] Features directly support MVP goals
- [ ] No extraneous features beyond MVP scope
- [ ] Critical features prioritized appropriately
- [ ] [[BROWNFIELD ONLY]] Enhancement complexity justified

### 8.2 User Journey Completeness

- [ ] All critical user journeys fully implemented
- [ ] Edge cases and error scenarios addressed
- [ ] User experience considerations included
- [ ] [[UI/UX ONLY]] Accessibility requirements incorporated
- [ ] [[BROWNFIELD ONLY]] Existing workflows preserved or improved

### 8.3 Technical Requirements

- [ ] All technical constraints from PRD addressed
- [ ] Non-functional requirements incorporated
- [ ] Architecture decisions align with constraints
- [ ] Performance considerations addressed
- [ ] [[BROWNFIELD ONLY]] Compatibility requirements met

## 9. DOCUMENTATION & HANDOFF

[[LLM: Good documentation enables smooth development. For brownfield, documentation of integration points is critical.]]

### 9.1 Developer Documentation

- [ ] API documentation created alongside implementation
- [ ] Setup instructions are comprehensive
- [ ] Architecture decisions documented
- [ ] Patterns and conventions documented
- [ ] [[BROWNFIELD ONLY]] Integration points documented in detail

### 9.2 User Documentation

- [ ] User guides or help documentation included if required
- [ ] Error messages and user feedback considered
- [ ] Onboarding flows fully specified
- [ ] [[BROWNFIELD ONLY]] Changes to existing features documented

### 9.3 Knowledge Transfer

- [ ] [[BROWNFIELD ONLY]] Existing system knowledge captured
- [ ] [[BROWNFIELD ONLY]] Integration knowledge documented
- [ ] Code review knowledge sharing planned
- [ ] Deployment knowledge transferred to operations
- [ ] Historical context preserved

## 10. POST-MVP CONSIDERATIONS

[[LLM: Planning for success prevents technical debt. For brownfield, ensure enhancements don't limit future growth.]]

### 10.1 Future Enhancements

- [ ] Clear separation between MVP and future features
- [ ] Architecture supports planned enhancements
- [ ] Technical debt considerations documented
- [ ] Extensibility points identified
- [ ] [[BROWNFIELD ONLY]] Integration patterns reusable

### 10.2 Monitoring & Feedback

- [ ] Analytics or usage tracking included if required
- [ ] User feedback collection considered
- [ ] Monitoring and alerting addressed
- [ ] Performance measurement incorporated
- [ ] [[BROWNFIELD ONLY]] Existing monitoring preserved/enhanced

## VALIDATION SUMMARY

[[LLM: FINAL PO VALIDATION REPORT GENERATION

Generate a comprehensive validation report that adapts to project type:

1. Executive Summary
   - Project type: [Greenfield/Brownfield] with [UI/No UI]
   - Overall readiness (percentage)
   - Go/No-Go recommendation
   - Critical blocking issues count
   - Sections skipped due to project type

2. Project-Specific Analysis

   FOR GREENFIELD:
   - Setup completeness
   - Dependency sequencing
   - MVP scope appropriateness
   - Development timeline feasibility

   FOR BROWNFIELD:
   - Integration risk level (High/Medium/Low)
   - Existing system impact assessment
   - Rollback readiness
   - User disruption potential

3. Risk Assessment
   - Top 5 risks by severity
   - Mitigation recommendations
   - Timeline impact of addressing issues
   - [BROWNFIELD] Specific integration risks

4. MVP Completeness
   - Core features coverage
   - Missing essential functionality
   - Scope creep identified
   - True MVP vs over-engineering

5. Implementation Readiness
   - Developer clarity score (1-10)
   - Ambiguous requirements count
   - Missing technical details
   - [BROWNFIELD] Integration point clarity

6. Recommendations
   - Must-fix before development
   - Should-fix for quality
   - Consider for improvement
   - Post-MVP deferrals

7. [BROWNFIELD ONLY] Integration Confidence
   - Confidence in preserving existing functionality
   - Rollback procedure completeness
   - Monitoring coverage for integration points
   - Support team readiness

After presenting the report, ask if the user wants:

- Detailed analysis of any failed sections
- Specific story reordering suggestions
- Risk mitigation strategies
- [BROWNFIELD] Integration risk deep-dive]]

### Category Statuses

| Category                                | Status | Critical Issues |
| --------------------------------------- | ------ | --------------- |
| 1. Project Setup & Initialization       | _TBD_  |                 |
| 2. Infrastructure & Deployment          | _TBD_  |                 |
| 3. External Dependencies & Integrations | _TBD_  |                 |
| 4. UI/UX Considerations                 | _TBD_  |                 |
| 5. User/Agent Responsibility            | _TBD_  |                 |
| 6. Feature Sequencing & Dependencies    | _TBD_  |                 |
| 7. Risk Management (Brownfield)         | _TBD_  |                 |
| 8. MVP Scope Alignment                  | _TBD_  |                 |
| 9. Documentation & Handoff              | _TBD_  |                 |
| 10. Post-MVP Considerations             | _TBD_  |                 |

### Critical Deficiencies

(To be populated during validation)

### Recommendations

(To be populated during validation)

### Final Decision

- **APPROVED**: The plan is comprehensive, properly sequenced, and ready for implementation.
- **CONDITIONAL**: The plan requires specific adjustments before proceeding.
- **REJECTED**: The plan requires significant revision to address critical deficiencies.
