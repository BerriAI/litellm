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
- `src/components/team/TeamInfo.tsx` (1724 LoC): team detail/edit page with a
  deeply coupled antd `Form` — `Form.List` + inline Promise validators for
  per-model rate limits, `Form.useWatch` hooks driving dependent rendering,
  `shouldUpdate` patterns, hidden form fields registered via `Form.Item`,
  antd `Select.OptGroup` with a custom `tagRender` for guardrails, and the
  Tremor `Accordion` still required by the settings panel. Migration
  exceeds the two-attempt budget for this batch. Defer until Tremor
  `Accordion` has a shadcn replacement adopted and the antd Form surface
  can be broken into sub-forms.
- `src/components/guardrails/add_guardrail_form.tsx` (1202 LoC): multi-step
  wizard driven by antd `Form.useForm` with provider-specific dynamic
  fields (Presidio PII, PromptGuard, content filter, custom code, tool
  permission, Azure Text Moderation). Uses `form.validateFields` per step,
  `form.setFieldsValue` to apply presets, `form.resetFields`, antd
  `Select` with children `Option` rendering provider logos, `Tag`,
  `Typography.Title/Text/Link`. The step navigation and provider-param
  validation are tightly coupled to antd's validation API.
- `src/components/guardrails/edit_guardrail_form.tsx` (451 LoC): mirrors
  add_guardrail_form's provider-switching logic for edit flow. Defer with
  it.
- `src/components/guardrails/guardrail_info.tsx` (888 LoC): guardrail
  detail/edit page with Tremor `TabGroup`/`TabList`/`Card`/`Grid` (still
  required by overview), a large antd `Form` settings panel
  (`Form.useForm` + `Form.Item` rules, `Input.TextArea`, antd `Select`,
  `Divider orientation="left"` section headers) that reuses
  `GuardrailProviderFields` and `GuardrailOptionalParams`. Settings-panel
  migration requires migrating both child components in tandem.
- `src/components/guardrails/guardrail_provider_fields.tsx` (240 LoC):
  renders antd `Form.Item` elements keyed to the parent's antd Form
  context (including nested `optional_params` + antd `Slider` with marks
  for percentage fields). Cannot be migrated independently of its two
  callers (`add_guardrail_form.tsx` + `guardrail_info.tsx`) without
  breaking the form wiring. Defer with them.
- `src/components/guardrails/guardrail_optional_params.tsx` (256 LoC):
  renders antd `Form.Item` for provider-specific optional params including
  a dict-field builder with dynamic antd `Select.Option` lists. Same
  dependency on the parent antd Form context as the above. Defer with the
  parent forms.
