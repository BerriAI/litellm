# /review-infrastructure Task

When this command is used, execute the following task:

<!-- Powered by BMAD™ Core -->

# Infrastructure Review Task

## Purpose

To conduct a thorough review of existing infrastructure to identify improvement opportunities, security concerns, and alignment with best practices. This task helps maintain infrastructure health, optimize costs, and ensure continued alignment with organizational requirements.

## Inputs

- Current infrastructure documentation
- Monitoring and logging data
- Recent incident reports
- Cost and performance metrics
- `infrastructure-checklist.md` (primary review framework)

## Key Activities & Instructions

### 1. Confirm Interaction Mode

- Ask the user: "How would you like to proceed with the infrastructure review? We can work:
  A. **Incrementally (Default & Recommended):** We'll work through each section of the checklist methodically, documenting findings for each item before moving to the next section. This provides a thorough review.
  B. **"YOLO" Mode:** I can perform a rapid assessment of all infrastructure components and present a comprehensive findings report. This is faster but may miss nuanced details."
- Request the user to select their preferred mode and proceed accordingly.

### 2. Prepare for Review

- Gather and organize current infrastructure documentation
- Access monitoring and logging systems for operational data
- Review recent incident reports for recurring issues
- Collect cost and performance metrics
- <critical_rule>Establish review scope and boundaries with the user before proceeding</critical_rule>

### 3. Conduct Systematic Review

- **If "Incremental Mode" was selected:**
  - For each section of the infrastructure checklist:
    - **a. Present Section Focus:** Explain what aspects of infrastructure this section reviews
    - **b. Work Through Items:** Examine each checklist item against current infrastructure
    - **c. Document Current State:** Record how current implementation addresses or fails to address each item
    - **d. Identify Gaps:** Document improvement opportunities with specific recommendations
    - **e. [Offer Advanced Self-Refinement & Elicitation Options](#offer-advanced-self-refinement--elicitation-options)**
    - **f. Section Summary:** Provide an assessment summary before moving to the next section

- **If "YOLO Mode" was selected:**
  - Rapidly assess all infrastructure components
  - Document key findings and improvement opportunities
  - Present a comprehensive review report
  - <important_note>After presenting the full review in YOLO mode, you MAY still offer the 'Advanced Reflective & Elicitation Options' menu for deeper investigation of specific areas with issues.</important_note>

### 4. Generate Findings Report

- Summarize review findings by category (Security, Performance, Cost, Reliability, etc.)
- Prioritize identified issues (Critical, High, Medium, Low)
- Document recommendations with estimated effort and impact
- Create an improvement roadmap with suggested timelines
- Highlight cost optimization opportunities

### 5. BMad Integration Assessment

- Evaluate how current infrastructure supports other BMad agents:
  - **Development Support:** Assess how infrastructure enables Frontend Dev (Mira), Backend Dev (Enrique), and Full Stack Dev workflows
  - **Product Alignment:** Verify infrastructure supports PRD requirements from Product Owner (Oli)
  - **Architecture Compliance:** Check if implementation follows Architect (Alphonse) decisions
  - Document any gaps in BMad integration

### 6. Architectural Escalation Assessment

- **DevOps/Platform → Architect Escalation Review:**
  - Evaluate review findings for issues requiring architectural intervention:
    - **Technical Debt Escalation:**
      - Identify infrastructure technical debt that impacts system architecture
      - Document technical debt items that require architectural redesign vs. operational fixes
      - Assess cumulative technical debt impact on system maintainability and scalability
    - **Performance/Security Issue Escalation:**
      - Identify performance bottlenecks that require architectural solutions (not just operational tuning)
      - Document security vulnerabilities that need architectural security pattern changes
      - Assess capacity and scalability issues requiring architectural scaling strategy revision
    - **Technology Evolution Escalation:**
      - Identify outdated technologies that need architectural migration planning
      - Document new technology opportunities that could improve system architecture
      - Assess technology compatibility issues requiring architectural integration strategy changes
  - **Escalation Decision Matrix:**
    - **Critical Architectural Issues:** Require immediate Architect Agent involvement for system redesign
    - **Significant Architectural Concerns:** Recommend Architect Agent review for potential architecture evolution
    - **Operational Issues:** Can be addressed through operational improvements without architectural changes
    - **Unclear/Ambiguous Issues:** When escalation level is uncertain, consult with user for guidance and decision
  - Document escalation recommendations with clear justification and impact assessment
  - <critical_rule>If escalation classification is unclear or ambiguous, HALT and ask user for guidance on appropriate escalation level and approach</critical_rule>

### 7. Present and Plan

- Prepare an executive summary of key findings
- Create detailed technical documentation for implementation teams
- Develop an action plan for critical and high-priority items
- **Prepare Architectural Escalation Report** (if applicable):
  - Document all findings requiring Architect Agent attention
  - Provide specific recommendations for architectural changes or reviews
  - Include impact assessment and priority levels for architectural work
  - Prepare escalation summary for Architect Agent collaboration
- Schedule follow-up reviews for specific areas
- <important_note>Present findings in a way that enables clear decision-making on next steps and escalation needs.</important_note>

### 8. Execute Escalation Protocol

- **If Critical Architectural Issues Identified:**
  - **Immediate Escalation to Architect Agent:**
    - Present architectural escalation report with critical findings
    - Request architectural review and potential redesign for identified issues
    - Collaborate with Architect Agent on priority and timeline for architectural changes
    - Document escalation outcomes and planned architectural work
- **If Significant Architectural Concerns Identified:**
  - **Scheduled Architectural Review:**
    - Prepare detailed technical findings for Architect Agent review
    - Request architectural assessment of identified concerns
    - Schedule collaborative planning session for potential architectural evolution
    - Document architectural recommendations and planned follow-up
- **If Only Operational Issues Identified:**
  - Proceed with operational improvement planning without architectural escalation
  - Monitor for future architectural implications of operational changes
- **If Unclear/Ambiguous Escalation Needed:**
  - **User Consultation Required:**
    - Present unclear findings and escalation options to user
    - Request user guidance on appropriate escalation level and approach
    - Document user decision and rationale for escalation approach
    - Proceed with user-directed escalation path
- <critical_rule>All critical architectural escalations must be documented and acknowledged by Architect Agent before proceeding with implementation</critical_rule>

## Output

A comprehensive infrastructure review report that includes:

1. **Current state assessment** for each infrastructure component
2. **Prioritized findings** with severity ratings
3. **Detailed recommendations** with effort/impact estimates
4. **Cost optimization opportunities**
5. **BMad integration assessment**
6. **Architectural escalation assessment** with clear escalation recommendations
7. **Action plan** for critical improvements and architectural work
8. **Escalation documentation** for Architect Agent collaboration (if applicable)

## Offer Advanced Self-Refinement & Elicitation Options

Present the user with the following list of 'Advanced Reflective, Elicitation & Brainstorming Actions'. Explain that these are optional steps to help ensure quality, explore alternatives, and deepen the understanding of the current section before finalizing it and moving on. The user can select an action by number, or choose to skip this and proceed to finalize the section.

"To ensure the quality of the current section: **[Specific Section Name]** and to ensure its robustness, explore alternatives, and consider all angles, I can perform any of the following actions. Please choose a number (8 to finalize and proceed):

**Advanced Reflective, Elicitation & Brainstorming Actions I Can Take:**

1. **Root Cause Analysis & Pattern Recognition**
2. **Industry Best Practice Comparison**
3. **Future Scalability & Growth Impact Assessment**
4. **Security Vulnerability & Threat Model Analysis**
5. **Operational Efficiency & Automation Opportunities**
6. **Cost Structure Analysis & Optimization Strategy**
7. **Compliance & Governance Gap Assessment**
8. **Finalize this Section and Proceed.**

After I perform the selected action, we can discuss the outcome and decide on any further revisions for this section."

REPEAT by Asking the user if they would like to perform another Reflective, Elicitation & Brainstorming Action UNTIL the user indicates it is time to proceed to the next section (or selects #8)
