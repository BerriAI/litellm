# Playground Chat Shadcn Migration

## Scope

Migrate the Playground Chat tab and its rendered component tree from Ant Design, Ant Design icons, and Tremor to the dashboard's shadcn/Base UI components and Lucide icons. Preserve behavior while fixing accessibility, responsiveness, state, validation, and test gaps discovered during migration.

## Completed

- [x] Audit the Playground Chat component tree and legacy UI dependencies.
- [x] Rank migration work from lowest to highest implementation risk.
- [x] Migrate `EndpointSelector.tsx` from Ant Design `Select` to the shared searchable `SearchSelect`.
- [x] Migrate `SessionManagement.tsx` to shadcn `Switch`, `Button`, and `Tooltip` components.
- [x] Add accessible labels and clipboard failure handling to session controls.
- [x] Migrate `CodeInterpreterTool.tsx` to shadcn controls and Lucide icons.
- [x] Migrate `SearchResultsDisplay.tsx` to keyboard-accessible shadcn collapsibles.
- [x] Migrate `A2AMetrics.tsx` buttons, tooltips, collapsibles, and icons.
- [x] Add `components/shared/MultiSelect.tsx` for searchable multi-value selection with removable chips.
- [x] Migrate `TagSelector.tsx` to `MultiSelect`, including custom tag creation.
- [x] Migrate `VectorStoreSelector.tsx` to `MultiSelect`.
- [x] Migrate `GuardrailSelector.tsx` to `MultiSelect`.
- [x] Migrate `PolicySelector.tsx` to `MultiSelect`.
- [x] Migrate the Virtual Key Source selector in `ChatUI.tsx` to shadcn `Select`.
- [x] Migrate the voice selector in `ChatUI.tsx` to shadcn `Select`.
- [x] Migrate the model selector in `ChatUI.tsx` to searchable `SearchSelect`.
- [x] Migrate the agent selector in `ChatUI.tsx` to searchable `SearchSelect`.
- [x] Standardize migrated dropdown controls to a compact 32px height.
- [x] Keep searchable dropdown menus below their fields instead of flipping above them.
- [x] Remove the fixed `80vh` Chat layout and percentage-width columns.
- [x] Add shrink, overflow, wrapping, and responsive behavior to the Playground shell, configuration panel, chat panel, composer, and message bubbles.
- [x] Parse model-list entries using `model_group`, `id`, or `model_name` so database-backed and key-scoped models can be displayed.
- [x] Deduplicate and alphabetically sort model results.
- [x] Fetch models when switching between the current UI session and a custom Virtual Key.
- [x] Add model loading, empty, missing-key, and request-failure states.
- [x] Show all models returned for the active key rather than filtering the list down to the current endpoint.
- [x] Automatically update the endpoint when a selected model has a known mode.
- [x] Migrate `ChatImageUpload.tsx` and `ResponsesImageUpload.tsx` from Ant Design uploads to semantic file inputs and shadcn buttons.
- [x] Enforce upload MIME type, extension, size, and count restrictions in application logic (`uploadValidation.ts`).
- [x] Migrate `FilePreviewCard.tsx` icons and remove button to Lucide and shadcn.
- [x] Migrate `CodeInterpreterOutput.tsx` collapsible, loading, and download controls.
- [x] Migrate `ReasoningContent.tsx` collapsible and icons.
- [x] Migrate `MCPEventsDisplay.tsx` collapsibles (remove Ant Design Collapse and styled-jsx).
- [x] Migrate `ResponseMetrics.tsx` tooltips and icons.
- [x] Migrate `AdditionalModelSettings.tsx` checkbox, numeric inputs, range sliders, popover, tooltip, and typography.

## Remaining Migration

- [ ] Migrate `MCPToolArgumentsForm.tsx` form controls.
- [ ] Migrate `ByokCredentialModal.tsx` to shadcn `Dialog`, inputs, and switch.
- [ ] Migrate `RealtimePlayground.tsx` buttons, inputs, selects, typography, and icons.
- [ ] Migrate remaining `ChatUI.tsx` Ant Design and Tremor buttons, inputs, dialogs, popovers, tool selectors, loading indicators, uploads, typography, cards, and icons.
- [ ] Migrate `ChatMessageBubble.tsx`, `ChatImageRenderer.tsx`, and `ResponsesImageRenderer.tsx` icons.
- [ ] Migrate `AgentBuilderView.tsx` if it remains in the Chat tab tree.
- [ ] Migrate the Playground page tabs from Tremor to shadcn tabs.
- [ ] Remove every `antd`, `@ant-design/icons`, and `@tremor/react` import from the Chat tab render tree.

## Behavior And Quality Gaps

- [ ] Validate the selected model and endpoint requirements before adding a message or clearing the draft.
- [ ] Prevent rapid duplicate submissions and stale chat-history updates.
- [ ] Keep attachments available after request failure so the user can retry.
- [ ] Add runtime validation and error handling for persisted Playground state.
- [ ] Review storage of chat content, session identifiers, selections, and API keys.
- [ ] Replace reversible Base64 API-key storage with an appropriate authentication/storage design.
- [ ] Audit remaining icon-only controls for accessible names and state attributes.
- [ ] Audit object URL lifecycle and revoke previews when removed or unmounted.
- [ ] Reduce `ChatUI.tsx` complexity by extracting request/state logic into testable modules.

## Tests And Verification

- [x] Add unit tests for upload MIME, extension, size, and count validation.
- [ ] Update tests that still target Ant Design selectors and DOM classes.
- [ ] Add unit tests for model response normalization, deduplication, and fallback fields.
- [ ] Add integration coverage for UI-session and Virtual Key model loading.
- [ ] Add integration coverage for searchable model selection and endpoint synchronization.
- [ ] Add integration coverage for upload validation and retry behavior.
- [ ] Add Playwright coverage for the core Playground Chat flow and responsive layout.
- [ ] Run the focused Vitest suites under the repository-required Node and npm versions.
- [ ] Run the dashboard lint, formatting, type-check, and build commands.
- [ ] Run ESLint with `--prune-suppressions` after resolving grandfathered warnings.

## Current Verification Notes

- Branch: `litellm_playground_chat_shadcn` from `litellm_internal_staging`.
- Focused Vitest suites pass under Node `24.14.1` / npm `11.11.0`:
  - `uploadValidation.test.ts`
  - `FilePreviewCard.test.tsx`
  - `CodeInterpreterOutput.test.tsx`
  - `AdditionalModelSettings.test.tsx`
  - `ReasoningContent.test.tsx`
- Remaining Ant Design / Tremor usage is concentrated in `ChatUI.tsx`, `RealtimePlayground.tsx`, image renderers, `ChatMessageBubble.tsx`, and `AgentBuilderView.tsx`.
