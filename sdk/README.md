# XCT LiteLLM SDKs

In-tree SDK packages for the XCT capability-provider surface. Both packages
are published independently from this monorepo:

| Package | Path | Registry |
|---|---|---|
| `@xct/litellm-sdk` (TypeScript) | `sdk/typescript/` | npm (private scope) |
| `xct-litellm` (Python) | `sdk/python/` | PyPI |

## Why in the main repo

Type definitions for the API surface live in `litellm/types/capabilities.py`,
`litellm/types/xct_apps.py`, `litellm/types/xct_skills.py`,
`litellm/types/webhooks.py`. Keeping the SDKs in the same repo means:

- one PR to add a field; both server and SDK move together
- no cross-repo OpenAPI sync; `/openapi-public.json` is generated from the
  same proxy process the SDK targets
- shared CI

## Release flow

- A push to `litellm_internal_staging` that touches `sdk/typescript/` runs
  the `publish-typescript-sdk.yml` workflow → `npm publish` from the
  `sdk/typescript/` working directory
- Same for `sdk/python/` → PyPI via `uv publish`
- Versions in each package's `package.json` / `pyproject.toml` are the
  source of truth; CI verifies the version was bumped before publishing

## Type generation

`scripts/gen_sdk_types.py` reads `/openapi-public.json` (the filtered
public OpenAPI from S5-02) and emits:

- `sdk/typescript/src/types/generated.ts` (via `openapi-typescript`)
- `sdk/python/xct_litellm/_generated.py` (via `datamodel-codegen`)

Run after any server-side type change:

```bash
make gen-sdk-types
```
