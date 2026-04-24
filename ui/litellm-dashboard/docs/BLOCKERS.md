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
- `src/components/settings.tsx` (840 LoC): logging-callbacks settings page.
  Two separate antd `Form`s (`addForm`, `editForm`) with `Form.useForm`,
  `validateFields`, `setFieldsValue`, `resetFields`, plus Tremor
  `TabGroup`/`TabList`/`Tab`/`TabPanels`/`TabPanel` and Tremor `Table`
  containers, dynamic provider-specific field rendering via `DynamicParamsFields`
  (which imports `antd/es/form/FormItem` directly and registers under the
  parent antd Form). A faithful migration needs both forms rewritten to RHF,
  the tabs flipped to shadcn, and `DynamicParamsFields` ported in lockstep
  with its two parents — exceeds the 2-attempt budget.
- `src/components/OldTeams.tsx` (1577 LoC): legacy teams page that mirrors
  `team/TeamInfo.tsx` in structure — deeply coupled antd `Form` with model /
  guardrail multi-selects, `Form.List`-style panels, antd `Pagination` + `Table`
  driven by server-side sort, antd `Tabs` container, and Tremor `Accordion`
  sections for router / logging / advanced settings. Same migration surface
  and blockers as TeamInfo (still-required Tremor `Accordion`, wide antd Form
  context consumed by several common_components). Defer until the TeamInfo
  blocker clears — these two pages must be migrated together or the shared
  form-surface common_components (ModelAliasManager, PremiumLoggingSettings,
  RouterSettingsAccordion) will be in an inconsistent state.
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
- `src/components/mcp_tools/create_mcp_server.tsx` (1060 LoC): end-to-end
  antd `Form` with deeply nested field paths (`["credentials", "client_id"]`,
  `static_headers` list with inline validators), `shouldUpdate` driven
  conditional rendering, `Form.useWatch`-style derived state, antd `Modal`,
  antd `Collapse` auth panel, antd `Select mode="tags"` for access groups.
  Existing `create_mcp_server.test.tsx` has ~15 tests keyed to antd class
  selectors (`.ant-select`, `.ant-select-selector`, `.ant-select-item-option`,
  `.ant-collapse-item`, `.ant-form-item`) and a full dropdown-click helper
  (`selectAntOption`) that has no stable shadcn analog in jsdom (Radix
  `Select` requires hasPointerCapture + scrollIntoView polyfills). Migration
  requires rewriting every test + migrating all 4 child form sections
  (`OAuthFormFields`, `StdioConfiguration`, `OpenAPIFormSection`,
  `MCPPermissionManagement`) in the same commit. Exceeds 2-attempt budget.
- `src/components/mcp_tools/mcp_server_edit.tsx` (1151 LoC): mirrors
  create_mcp_server's antd Form structure for the edit flow, plus Tremor
  `TabGroup`/`TabList`/`TabPanels`. Uses `Form.useWatch` for derived state
  (authType, transportType, oauth_flow_type) that drives conditional
  rendering. Shares the same child form components as create; cannot be
  migrated independently. Existing `mcp_server_edit.test.tsx` has 7 tests
  keyed to antd form id lookups (`document.getElementById("token_validation_json")`)
  and `Form.Item`-generated labels. Defer with create_mcp_server.
- `src/components/mcp_tools/OAuthFormFields.tsx` (327 LoC): renders antd
  `Form.Item` elements (including `name={["credentials", "client_id"]}`,
  `Select mode="tags"`, inline `validator` rules for JSON parse, `InputNumber`)
  keyed to the parent's antd Form context. Cannot be migrated independently
  of its two callers (`create_mcp_server.tsx`, `mcp_server_edit.tsx`).
  `OAuthFormFields.test.tsx` has 13 tests directly importing `Form` from
  antd and asserting on antd-validation error messages. Defer with parents.
- `src/components/mcp_tools/StdioConfiguration.tsx`: renders a single antd
  `Form.Item` with `name="stdio_config"` and inline JSON-parse `validator`,
  registering into the parent antd Form context. Cannot be migrated
  independently. Defer with parents.
- `src/components/mcp_tools/OpenAPIFormSection.tsx`: renders an antd
  `Form.Item` with `name="spec_path"` and takes an antd `FormInstance`
  prop for `setFieldsValue`/`resetFields`. Tightly coupled to the parent
  antd Form. Defer with parents.
- `src/components/mcp_tools/MCPPermissionManagement.tsx`: uses
 `Form.useFormInstance()` + `Form.List` for dynamic static-headers plus
 several `Form.Item` fields (`allow_all_keys`, `available_on_public_internet`,
 `mcp_access_groups`, `extra_headers`). Directly writes to the parent's
 antd Form via `form.setFieldValue`. `MCPPermissionManagement.test.tsx`
 wraps it in an antd `<Form form={form}>` harness.  Cannot migrate
 independently of the parent forms. Defer with parents.
- `src/components/usage.tsx` (949 LoC): legacy admin usage page with a very
 wide tremor surface — BarList, DonutChart, AreaChart, DateRangePickerValue,
 Tremor MultiSelect / MultiSelectItem (no direct shadcn primitive — requires
 a custom searchable-multi-select combobox), Select/SelectItem used with
 the Tremor API, plus TabGroup / TabList / Tab / TabPanels / TabPanel driving
 four distinct admin reports, and Tremor Table inside Card wrappers for tag /
 provider / customer views. Accurate migration requires building the same
 searchable MultiSelect primitive blocking ChatUI, migrating the tab content
 into value-keyed shadcn Tabs, and splitting the many chart imports from the
 non-chart primitives in a way that preserves the categorical color palettes.
 Exceeds the two-attempt budget for this run — defer.
- `src/components/playground/chat_ui/ChatUI.tsx` (2239 LoC): chrome-only
 migration blocked by deeply-coupled antd `Select.OptGroup` + `optionLabelProp`
 + `maxTagCount="responsive"` + custom `filterOption` on the MCP servers
 multi-select (with "__all__" sentinel, toolset/server option groups, and per-
 option custom JSX rendering). The direct-tool / arg selectors inside the MCP
 tool picker use the same pattern. shadcn `<Select>` has no `OptGroup` and no
 searchable multi-select primitive; a faithful rewrite needs a custom
 `Command`+`Popover` combobox plus a chip display, in tandem with the "__all__"
 sentinel handling. Migrating the surrounding chrome (Modal, Popover, Tooltip,
 Spin, Button, TextArea, Dragger) alone would leave the file in an
 inconsistent half-migrated state and exceeds the 2-attempt budget. Defer until
 a reusable searchable-multi-select-with-groups primitive exists.
