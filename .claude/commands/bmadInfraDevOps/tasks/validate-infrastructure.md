# /validate-infrastructure Task

When this command is used, execute the following task:

<!-- Powered by BMAD™ Core -->

# Infrastructure Validation Task

## Purpose

To comprehensively validate platform infrastructure changes against security, reliability, operational, and compliance requirements before deployment. This task ensures all platform infrastructure meets organizational standards, follows best practices, and properly integrates with the broader BMad ecosystem.

## Inputs

- Infrastructure Change Request (`docs/infrastructure/{ticketNumber}.change.md`)
- **Infrastructure Architecture Document** (`docs/infrastructure-architecture.md` - from Architect Agent)
- Infrastructure Guidelines (`docs/infrastructure/guidelines.md`)
- Technology Stack Document (`docs/tech-stack.md`)
- `infrastructure-checklist.md` (primary validation framework - 16 comprehensive sections)

## Key Activities & Instructions

### 1. Confirm Interaction Mode

- Ask the user: "How would you like to proceed with platform infrastructure validation? We can work:
  A. **Incrementally (Default & Recommended):** We'll work through each section of the checklist step-by-step, documenting compliance or gaps for each item before moving to the next section. This is best for thorough validation and detailed documentation of the complete platform stack.
  B. **"YOLO" Mode:** I can perform a rapid assessment of all checklist items and present a comprehensive validation report for review. This is faster but may miss nuanced details that would be caught in the incremental approach."
- Request the user to select their preferred mode (e.g., "Please let me know if you'd prefer A or B.").
- Once the user chooses, confirm the selected mode and proceed accordingly.

### 2. Initialize Platform Validation

- Review the infrastructure change documentation to understand platform implementation scope and purpose
- Analyze the infrastructure architecture document for platform design patterns and compliance requirements
- Examine infrastructure guidelines for organizational standards across all platform components
- Prepare the validation environment and tools for comprehensive platform testing
- <critical_rule>Verify the infrastructure change request is approved for validation. If not, HALT and inform the user.</critical_rule>

### 3. Architecture Design Review Gate

- **DevOps/Platform → Architect Design Review:**
  - Conduct systematic review of infrastructure architecture document for implementability
  - Evaluate architectural decisions against operational constraints and capabilities:
    - **Implementation Complexity:** Assess if proposed architecture can be implemented with available tools and expertise
    - **Operational Feasibility:** Validate that operational patterns are achievable within current organizational maturity
    - **Resource Availability:** Confirm required infrastructure resources are available and within budget constraints
    - **Technology Compatibility:** Verify selected technologies integrate properly with existing infrastructure
    - **Security Implementation:** Validate that security patterns can be implemented with current security toolchain
    - **Maintenance Overhead:** Assess ongoing operational burden and maintenance requirements
  - Document design review findings and recommendations:
    - **Approved Aspects:** Document architectural decisions that are implementable as designed
    - **Implementation Concerns:** Identify architectural decisions that may face implementation challenges
    - **Required Modifications:** Recommend specific changes needed to make architecture implementable
    - **Alternative Approaches:** Suggest alternative implementation patterns where needed
  - **Collaboration Decision Point:**
    - If **critical implementation blockers** identified: HALT validation and escalate to Architect Agent for architectural revision
    - If **minor concerns** identified: Document concerns and proceed with validation, noting required implementation adjustments
    - If **architecture approved**: Proceed with comprehensive platform validation
  - <critical_rule>All critical design review issues must be resolved before proceeding to detailed validation</critical_rule>

### 4. Execute Comprehensive Platform Validation Process

- **If "Incremental Mode" was selected:**
  - For each section of the infrastructure checklist (Sections 1-16):
    - **a. Present Section Purpose:** Explain what this section validates and why it's important for platform operations
    - **b. Work Through Items:** Present each checklist item, guide the user through validation, and document compliance or gaps
    - **c. Evidence Collection:** For each compliant item, document how compliance was verified
    - **d. Gap Documentation:** For each non-compliant item, document specific issues and proposed remediation
    - **e. Platform Integration Testing:** For platform engineering sections (13-16), validate integration between platform components
    - **f. [Offer Advanced Self-Refinement & Elicitation Options](#offer-advanced-self-refinement--elicitation-options)**
    - **g. Section Summary:** Provide a compliance percentage and highlight critical findings before moving to the next section

- **If "YOLO Mode" was selected:**
  - Work through all checklist sections rapidly (foundation infrastructure sections 1-12 + platform engineering sections 13-16)
  - Document compliance status for each item across all platform components
  - Identify and document critical non-compliance issues affecting platform operations
  - Present a comprehensive validation report for all sections
  - <important_note>After presenting the full validation report in YOLO mode, you MAY still offer the 'Advanced Reflective & Elicitation Options' menu for deeper investigation of specific sections with issues.</important_note>

### 5. Generate Comprehensive Platform Validation Report

- Summarize validation findings by section across all 16 checklist areas
- Calculate and present overall compliance percentage for complete platform stack
- Clearly document all non-compliant items with remediation plans prioritized by platform impact
- Highlight critical security or operational risks affecting platform reliability
- Include design review findings and architectural implementation recommendations
- Provide validation signoff recommendation based on complete platform assessment
- Document platform component integration validation results

### 6. BMad Integration Assessment

- Review how platform infrastructure changes support other BMad agents:
  - **Development Agent Alignment:** Verify platform infrastructure supports Frontend Dev, Backend Dev, and Full Stack Dev requirements including:
    - Container platform development environment provisioning
    - GitOps workflows for application deployment
    - Service mesh integration for development testing
    - Developer experience platform self-service capabilities
  - **Product Alignment:** Ensure platform infrastructure implements PRD requirements from Product Owner including:
    - Scalability and performance requirements through container platform
    - Deployment automation through GitOps workflows
    - Service reliability through service mesh implementation
  - **Architecture Alignment:** Validate that platform implementation aligns with architecture decisions including:
    - Technology selections implemented correctly across all platform components
    - Security architecture implemented in container platform, service mesh, and GitOps
    - Integration patterns properly implemented between platform components
  - Document all integration points and potential impacts on other agents' workflows

### 7. Next Steps Recommendation

- If validation successful:
  - Prepare platform deployment recommendation with component dependencies
  - Outline monitoring requirements for complete platform stack
  - Suggest knowledge transfer activities for platform operations
  - Document platform readiness certification
- If validation failed:
  - Prioritize remediation actions by platform component and integration impact
  - Recommend blockers vs. non-blockers for platform deployment
  - Schedule follow-up validation with focus on failed platform components
  - Document platform risks and mitigation strategies
- If design review identified architectural issues:
  - **Escalate to Architect Agent** for architectural revision and re-design
  - Document specific architectural changes required for implementability
  - Schedule follow-up design review after architectural modifications
- Update documentation with validation results across all platform components
- <important_note>Always ensure the Infrastructure Change Request status is updated to reflect the platform validation outcome.</important_note>

## Output

A comprehensive platform validation report documenting:

1. **Architecture Design Review Results** - Implementability assessment and architectural recommendations
2. **Compliance percentage by checklist section** (all 16 sections including platform engineering)
3. **Detailed findings for each non-compliant item** across foundation and platform components
4. **Platform integration validation results** documenting component interoperability
5. **Remediation recommendations with priority levels** based on platform impact
6. **BMad integration assessment results** for complete platform stack
7. **Clear signoff recommendation** for platform deployment readiness or architectural revision requirements
8. **Next steps for implementation or remediation** prioritized by platform dependencies

## Offer Advanced Self-Refinement & Elicitation Options

Present the user with the following list of 'Advanced Reflective, Elicitation & Brainstorming Actions'. Explain that these are optional steps to help ensure quality, explore alternatives, and deepen the understanding of the current section before finalizing it and moving on. The user can select an action by number, or choose to skip this and proceed to finalize the section.

"To ensure the quality of the current section: **[Specific Section Name]** and to ensure its robustness, explore alternatives, and consider all angles, I can perform any of the following actions. Please choose a number (8 to finalize and proceed):

**Advanced Reflective, Elicitation & Brainstorming Actions I Can Take:**

1. **Critical Security Assessment & Risk Analysis**
2. **Platform Integration & Component Compatibility Evaluation**
3. **Cross-Environment Consistency Review**
4. **Technical Debt & Maintainability Analysis**
5. **Compliance & Regulatory Alignment Deep Dive**
6. **Cost Optimization & Resource Efficiency Analysis**
7. **Operational Resilience & Platform Failure Mode Testing (Theoretical)**
8. **Finalize this Section and Proceed.**

After I perform the selected action, we can discuss the outcome and decide on any further revisions for this section."

REPEAT by Asking the user if they would like to perform another Reflective, Elicitation & Brainstorming Action UNTIL the user indicates it is time to proceed to the next section (or selects #8)
