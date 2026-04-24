# Blockers — phase 1 shadcn migration

One-line entries per blocker. Format:
`<path>: <reason>`

## Deferred until owning call sites migrate

- `src/components/common_components/check_openapi_schema.tsx`: tightly coupled to antd `Form.Item` API (uses `rules`, `validator`, `help`, `initialValue` props that have no direct RHF analog in one file). Only caller is `organisms/create_key_button.tsx` which still uses antd Form. Defer migration until create_key_button migrates.
- `src/components/common_components/PassThroughGuardrailsSection.tsx`: uses `antd.Select` in `mode="tags"` (free-form tag input) and antd `Form.Item` labels. Callers (`add_pass_through.tsx`, `CreatePassThroughEndpoint.tsx`) still on antd Form. Defer.

## Very large / deeply-coupled antd forms — out of scope for this run

Files where the cost to faithfully migrate exceeds the two-attempt budget
and a visual/behavioral regression would break core workflows. These
stay on antd for phase 1 and will be addressed in a targeted follow-up.

- `src/components/organisms/create_key_button.tsx` (1672 LoC): end-to-end
  antd `Form` with deeply nested validation, collapse panels, dynamic
  field visibility based on Team/Org selection. Migration would
  realistically require extracting 8+ sub-forms; the existing behavior
  is heavily test-covered.
- `src/components/templates/key_info_view.tsx` (863 LoC): mirrors
  create_key_button's form structure for the edit flow. Same concerns.
- `src/components/templates/key_edit_view.tsx` (774 LoC): shares
  common_components (RateLimitTypeFormItem, etc.) with the above;
  defer with them.
- `src/components/public_model_hub.tsx` (2033 LoC): multi-tab page with
  its own filter/search/form state; deep antd Select + Form coupling.
- `src/components/pass_through_info.tsx` (427 LoC): uses
  `PassThroughGuardrailsSection` (itself blocked), plus an antd Form.Item
  with Switch/InputNumber/Select controls. Defer until shared section
  unblocks.
- `src/components/add_pass_through.tsx`: same dependency chain.
