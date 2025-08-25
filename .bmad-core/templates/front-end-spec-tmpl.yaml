# <!-- Powered by BMAD™ Core -->
template:
  id: frontend-spec-template-v2
  name: UI/UX Specification
  version: 2.0
  output:
    format: markdown
    filename: docs/front-end-spec.md
    title: "{{project_name}} UI/UX Specification"

workflow:
  mode: interactive
  elicitation: advanced-elicitation

sections:
  - id: introduction
    title: Introduction
    instruction: |
      Review provided documents including Project Brief, PRD, and any user research to gather context. Focus on understanding user needs, pain points, and desired outcomes before beginning the specification.

      Establish the document's purpose and scope. Keep the content below but ensure project name is properly substituted.
    content: |
      This document defines the user experience goals, information architecture, user flows, and visual design specifications for {{project_name}}'s user interface. It serves as the foundation for visual design and frontend development, ensuring a cohesive and user-centered experience.
    sections:
      - id: ux-goals-principles
        title: Overall UX Goals & Principles
        instruction: |
          Work with the user to establish and document the following. If not already defined, facilitate a discussion to determine:

          1. Target User Personas - elicit details or confirm existing ones from PRD
          2. Key Usability Goals - understand what success looks like for users
          3. Core Design Principles - establish 3-5 guiding principles
        elicit: true
        sections:
          - id: user-personas
            title: Target User Personas
            template: "{{persona_descriptions}}"
            examples:
              - "**Power User:** Technical professionals who need advanced features and efficiency"
              - "**Casual User:** Occasional users who prioritize ease of use and clear guidance"
              - "**Administrator:** System managers who need control and oversight capabilities"
          - id: usability-goals
            title: Usability Goals
            template: "{{usability_goals}}"
            examples:
              - "Ease of learning: New users can complete core tasks within 5 minutes"
              - "Efficiency of use: Power users can complete frequent tasks with minimal clicks"
              - "Error prevention: Clear validation and confirmation for destructive actions"
              - "Memorability: Infrequent users can return without relearning"
          - id: design-principles
            title: Design Principles
            template: "{{design_principles}}"
            type: numbered-list
            examples:
              - "**Clarity over cleverness** - Prioritize clear communication over aesthetic innovation"
              - "**Progressive disclosure** - Show only what's needed, when it's needed"
              - "**Consistent patterns** - Use familiar UI patterns throughout the application"
              - "**Immediate feedback** - Every action should have a clear, immediate response"
              - "**Accessible by default** - Design for all users from the start"
      - id: changelog
        title: Change Log
        type: table
        columns: [Date, Version, Description, Author]
        instruction: Track document versions and changes

  - id: information-architecture
    title: Information Architecture (IA)
    instruction: |
      Collaborate with the user to create a comprehensive information architecture:

      1. Build a Site Map or Screen Inventory showing all major areas
      2. Define the Navigation Structure (primary, secondary, breadcrumbs)
      3. Use Mermaid diagrams for visual representation
      4. Consider user mental models and expected groupings
    elicit: true
    sections:
      - id: sitemap
        title: Site Map / Screen Inventory
        type: mermaid
        mermaid_type: graph
        template: "{{sitemap_diagram}}"
        examples:
          - |
            graph TD
                A[Homepage] --> B[Dashboard]
                A --> C[Products]
                A --> D[Account]
                B --> B1[Analytics]
                B --> B2[Recent Activity]
                C --> C1[Browse]
                C --> C2[Search]
                C --> C3[Product Details]
                D --> D1[Profile]
                D --> D2[Settings]
                D --> D3[Billing]
      - id: navigation-structure
        title: Navigation Structure
        template: |
          **Primary Navigation:** {{primary_nav_description}}

          **Secondary Navigation:** {{secondary_nav_description}}

          **Breadcrumb Strategy:** {{breadcrumb_strategy}}

  - id: user-flows
    title: User Flows
    instruction: |
      For each critical user task identified in the PRD:

      1. Define the user's goal clearly
      2. Map out all steps including decision points
      3. Consider edge cases and error states
      4. Use Mermaid flow diagrams for clarity
      5. Link to external tools (Figma/Miro) if detailed flows exist there

      Create subsections for each major flow.
    elicit: true
    repeatable: true
    sections:
      - id: flow
        title: "{{flow_name}}"
        template: |
          **User Goal:** {{flow_goal}}

          **Entry Points:** {{entry_points}}

          **Success Criteria:** {{success_criteria}}
        sections:
          - id: flow-diagram
            title: Flow Diagram
            type: mermaid
            mermaid_type: graph
            template: "{{flow_diagram}}"
          - id: edge-cases
            title: "Edge Cases & Error Handling:"
            type: bullet-list
            template: "- {{edge_case}}"
          - id: notes
            template: "**Notes:** {{flow_notes}}"

  - id: wireframes-mockups
    title: Wireframes & Mockups
    instruction: |
      Clarify where detailed visual designs will be created (Figma, Sketch, etc.) and how to reference them. If low-fidelity wireframes are needed, offer to help conceptualize layouts for key screens.
    elicit: true
    sections:
      - id: design-files
        template: "**Primary Design Files:** {{design_tool_link}}"
      - id: key-screen-layouts
        title: Key Screen Layouts
        repeatable: true
        sections:
          - id: screen
            title: "{{screen_name}}"
            template: |
              **Purpose:** {{screen_purpose}}

              **Key Elements:**
              - {{element_1}}
              - {{element_2}}
              - {{element_3}}

              **Interaction Notes:** {{interaction_notes}}

              **Design File Reference:** {{specific_frame_link}}

  - id: component-library
    title: Component Library / Design System
    instruction: |
      Discuss whether to use an existing design system or create a new one. If creating new, identify foundational components and their key states. Note that detailed technical specs belong in front-end-architecture.
    elicit: true
    sections:
      - id: design-system-approach
        template: "**Design System Approach:** {{design_system_approach}}"
      - id: core-components
        title: Core Components
        repeatable: true
        sections:
          - id: component
            title: "{{component_name}}"
            template: |
              **Purpose:** {{component_purpose}}

              **Variants:** {{component_variants}}

              **States:** {{component_states}}

              **Usage Guidelines:** {{usage_guidelines}}

  - id: branding-style
    title: Branding & Style Guide
    instruction: Link to existing style guide or define key brand elements. Ensure consistency with company brand guidelines if they exist.
    elicit: true
    sections:
      - id: visual-identity
        title: Visual Identity
        template: "**Brand Guidelines:** {{brand_guidelines_link}}"
      - id: color-palette
        title: Color Palette
        type: table
        columns: ["Color Type", "Hex Code", "Usage"]
        rows:
          - ["Primary", "{{primary_color}}", "{{primary_usage}}"]
          - ["Secondary", "{{secondary_color}}", "{{secondary_usage}}"]
          - ["Accent", "{{accent_color}}", "{{accent_usage}}"]
          - ["Success", "{{success_color}}", "Positive feedback, confirmations"]
          - ["Warning", "{{warning_color}}", "Cautions, important notices"]
          - ["Error", "{{error_color}}", "Errors, destructive actions"]
          - ["Neutral", "{{neutral_colors}}", "Text, borders, backgrounds"]
      - id: typography
        title: Typography
        sections:
          - id: font-families
            title: Font Families
            template: |
              - **Primary:** {{primary_font}}
              - **Secondary:** {{secondary_font}}
              - **Monospace:** {{mono_font}}
          - id: type-scale
            title: Type Scale
            type: table
            columns: ["Element", "Size", "Weight", "Line Height"]
            rows:
              - ["H1", "{{h1_size}}", "{{h1_weight}}", "{{h1_line}}"]
              - ["H2", "{{h2_size}}", "{{h2_weight}}", "{{h2_line}}"]
              - ["H3", "{{h3_size}}", "{{h3_weight}}", "{{h3_line}}"]
              - ["Body", "{{body_size}}", "{{body_weight}}", "{{body_line}}"]
              - ["Small", "{{small_size}}", "{{small_weight}}", "{{small_line}}"]
      - id: iconography
        title: Iconography
        template: |
          **Icon Library:** {{icon_library}}

          **Usage Guidelines:** {{icon_guidelines}}
      - id: spacing-layout
        title: Spacing & Layout
        template: |
          **Grid System:** {{grid_system}}

          **Spacing Scale:** {{spacing_scale}}

  - id: accessibility
    title: Accessibility Requirements
    instruction: Define specific accessibility requirements based on target compliance level and user needs. Be comprehensive but practical.
    elicit: true
    sections:
      - id: compliance-target
        title: Compliance Target
        template: "**Standard:** {{compliance_standard}}"
      - id: key-requirements
        title: Key Requirements
        template: |
          **Visual:**
          - Color contrast ratios: {{contrast_requirements}}
          - Focus indicators: {{focus_requirements}}
          - Text sizing: {{text_requirements}}

          **Interaction:**
          - Keyboard navigation: {{keyboard_requirements}}
          - Screen reader support: {{screen_reader_requirements}}
          - Touch targets: {{touch_requirements}}

          **Content:**
          - Alternative text: {{alt_text_requirements}}
          - Heading structure: {{heading_requirements}}
          - Form labels: {{form_requirements}}
      - id: testing-strategy
        title: Testing Strategy
        template: "{{accessibility_testing}}"

  - id: responsiveness
    title: Responsiveness Strategy
    instruction: Define breakpoints and adaptation strategies for different device sizes. Consider both technical constraints and user contexts.
    elicit: true
    sections:
      - id: breakpoints
        title: Breakpoints
        type: table
        columns: ["Breakpoint", "Min Width", "Max Width", "Target Devices"]
        rows:
          - ["Mobile", "{{mobile_min}}", "{{mobile_max}}", "{{mobile_devices}}"]
          - ["Tablet", "{{tablet_min}}", "{{tablet_max}}", "{{tablet_devices}}"]
          - ["Desktop", "{{desktop_min}}", "{{desktop_max}}", "{{desktop_devices}}"]
          - ["Wide", "{{wide_min}}", "-", "{{wide_devices}}"]
      - id: adaptation-patterns
        title: Adaptation Patterns
        template: |
          **Layout Changes:** {{layout_adaptations}}

          **Navigation Changes:** {{nav_adaptations}}

          **Content Priority:** {{content_adaptations}}

          **Interaction Changes:** {{interaction_adaptations}}

  - id: animation
    title: Animation & Micro-interactions
    instruction: Define motion design principles and key interactions. Keep performance and accessibility in mind.
    elicit: true
    sections:
      - id: motion-principles
        title: Motion Principles
        template: "{{motion_principles}}"
      - id: key-animations
        title: Key Animations
        repeatable: true
        template: "- **{{animation_name}}:** {{animation_description}} (Duration: {{duration}}, Easing: {{easing}})"

  - id: performance
    title: Performance Considerations
    instruction: Define performance goals and strategies that impact UX design decisions.
    sections:
      - id: performance-goals
        title: Performance Goals
        template: |
          - **Page Load:** {{load_time_goal}}
          - **Interaction Response:** {{interaction_goal}}
          - **Animation FPS:** {{animation_goal}}
      - id: design-strategies
        title: Design Strategies
        template: "{{performance_strategies}}"

  - id: next-steps
    title: Next Steps
    instruction: |
      After completing the UI/UX specification:

      1. Recommend review with stakeholders
      2. Suggest creating/updating visual designs in design tool
      3. Prepare for handoff to Design Architect for frontend architecture
      4. Note any open questions or decisions needed
    sections:
      - id: immediate-actions
        title: Immediate Actions
        type: numbered-list
        template: "{{action}}"
      - id: design-handoff-checklist
        title: Design Handoff Checklist
        type: checklist
        items:
          - "All user flows documented"
          - "Component inventory complete"
          - "Accessibility requirements defined"
          - "Responsive strategy clear"
          - "Brand guidelines incorporated"
          - "Performance goals established"

  - id: checklist-results
    title: Checklist Results
    instruction: If a UI/UX checklist exists, run it against this document and report results here.
