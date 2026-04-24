# Blockers — phase 1 shadcn migration

One-line entries per blocker. Format:
`<path>: <reason>`

## Deferred until owning call sites migrate

- `src/components/common_components/check_openapi_schema.tsx`: tightly coupled to antd `Form.Item` API (uses `rules`, `validator`, `help`, `initialValue` props that have no direct RHF analog in one file). Only caller is `organisms/create_key_button.tsx` which still uses antd Form. Defer migration until create_key_button migrates.
- `src/components/common_components/PassThroughGuardrailsSection.tsx`: uses `antd.Select` in `mode="tags"` (free-form tag input) and antd `Form.Item` labels. Callers (`add_pass_through.tsx`, `CreatePassThroughEndpoint.tsx`) still on antd Form. Defer.
