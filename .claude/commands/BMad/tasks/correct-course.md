# /correct-course Task

When this command is used, execute the following task:

<!-- Powered by BMAD™ Core -->

# Correct Course Task

## Purpose

- Guide a structured response to a change trigger using the `.bmad-core/checklists/change-checklist`.
- Analyze the impacts of the change on epics, project artifacts, and the MVP, guided by the checklist's structure.
- Explore potential solutions (e.g., adjust scope, rollback elements, re-scope features) as prompted by the checklist.
- Draft specific, actionable proposed updates to any affected project artifacts (e.g., epics, user stories, PRD sections, architecture document sections) based on the analysis.
- Produce a consolidated "Sprint Change Proposal" document that contains the impact analysis and the clearly drafted proposed edits for user review and approval.
- Ensure a clear handoff path if the nature of the changes necessitates fundamental replanning by other core agents (like PM or Architect).

## Instructions

### 1. Initial Setup & Mode Selection

- **Acknowledge Task & Inputs:**
  - Confirm with the user that the "Correct Course Task" (Change Navigation & Integration) is being initiated.
  - Verify the change trigger and ensure you have the user's initial explanation of the issue and its perceived impact.
  - Confirm access to all relevant project artifacts (e.g., PRD, Epics/Stories, Architecture Documents, UI/UX Specifications) and, critically, the `.bmad-core/checklists/change-checklist`.
- **Establish Interaction Mode:**
  - Ask the user their preferred interaction mode for this task:
    - **"Incrementally (Default & Recommended):** Shall we work through the change-checklist section by section, discussing findings and collaboratively drafting proposed changes for each relevant part before moving to the next? This allows for detailed, step-by-step refinement."
    - **"YOLO Mode (Batch Processing):** Or, would you prefer I conduct a more batched analysis based on the checklist and then present a consolidated set of findings and proposed changes for a broader review? This can be quicker for initial assessment but might require more extensive review of the combined proposals."
  - Once the user chooses, confirm the selected mode and then inform the user: "We will now use the change-checklist to analyze the change and draft proposed updates. I will guide you through the checklist items based on our chosen interaction mode."

### 2. Execute Checklist Analysis (Iteratively or Batched, per Interaction Mode)

- Systematically work through Sections 1-4 of the change-checklist (typically covering Change Context, Epic/Story Impact Analysis, Artifact Conflict Resolution, and Path Evaluation/Recommendation).
- For each checklist item or logical group of items (depending on interaction mode):
  - Present the relevant prompt(s) or considerations from the checklist to the user.
  - Request necessary information and actively analyze the relevant project artifacts (PRD, epics, architecture documents, story history, etc.) to assess the impact.
  - Discuss your findings for each item with the user.
  - Record the status of each checklist item (e.g., `[x] Addressed`, `[N/A]`, `[!] Further Action Needed`) and any pertinent notes or decisions.
  - Collaboratively agree on the "Recommended Path Forward" as prompted by Section 4 of the checklist.

### 3. Draft Proposed Changes (Iteratively or Batched)

- Based on the completed checklist analysis (Sections 1-4) and the agreed "Recommended Path Forward" (excluding scenarios requiring fundamental replans that would necessitate immediate handoff to PM/Architect):
  - Identify the specific project artifacts that require updates (e.g., specific epics, user stories, PRD sections, architecture document components, diagrams).
  - **Draft the proposed changes directly and explicitly for each identified artifact.** Examples include:
    - Revising user story text, acceptance criteria, or priority.
    - Adding, removing, reordering, or splitting user stories within epics.
    - Proposing modified architecture diagram snippets (e.g., providing an updated Mermaid diagram block or a clear textual description of the change to an existing diagram).
    - Updating technology lists, configuration details, or specific sections within the PRD or architecture documents.
    - Drafting new, small supporting artifacts if necessary (e.g., a brief addendum for a specific decision).
  - If in "Incremental Mode," discuss and refine these proposed edits for each artifact or small group of related artifacts with the user as they are drafted.
  - If in "YOLO Mode," compile all drafted edits for presentation in the next step.

### 4. Generate "Sprint Change Proposal" with Edits

- Synthesize the complete change-checklist analysis (covering findings from Sections 1-4) and all the agreed-upon proposed edits (from Instruction 3) into a single document titled "Sprint Change Proposal." This proposal should align with the structure suggested by Section 5 of the change-checklist.
- The proposal must clearly present:
  - **Analysis Summary:** A concise overview of the original issue, its analyzed impact (on epics, artifacts, MVP scope), and the rationale for the chosen path forward.
  - **Specific Proposed Edits:** For each affected artifact, clearly show or describe the exact changes (e.g., "Change Story X.Y from: [old text] To: [new text]", "Add new Acceptance Criterion to Story A.B: [new AC]", "Update Section 3.2 of Architecture Document as follows: [new/modified text or diagram description]").
- Present the complete draft of the "Sprint Change Proposal" to the user for final review and feedback. Incorporate any final adjustments requested by the user.

### 5. Finalize & Determine Next Steps

- Obtain explicit user approval for the "Sprint Change Proposal," including all the specific edits documented within it.
- Provide the finalized "Sprint Change Proposal" document to the user.
- **Based on the nature of the approved changes:**
  - **If the approved edits sufficiently address the change and can be implemented directly or organized by a PO/SM:** State that the "Correct Course Task" is complete regarding analysis and change proposal, and the user can now proceed with implementing or logging these changes (e.g., updating actual project documents, backlog items). Suggest handoff to a PO/SM agent for backlog organization if appropriate.
  - **If the analysis and proposed path (as per checklist Section 4 and potentially Section 6) indicate that the change requires a more fundamental replan (e.g., significant scope change, major architectural rework):** Clearly state this conclusion. Advise the user that the next step involves engaging the primary PM or Architect agents, using the "Sprint Change Proposal" as critical input and context for that deeper replanning effort.

## Output Deliverables

- **Primary:** A "Sprint Change Proposal" document (in markdown format). This document will contain:
  - A summary of the change-checklist analysis (issue, impact, rationale for the chosen path).
  - Specific, clearly drafted proposed edits for all affected project artifacts.
- **Implicit:** An annotated change-checklist (or the record of its completion) reflecting the discussions, findings, and decisions made during the process.
