---
sidebar_position: 2
title: "openapi-public.json (auto-generated)"
---

# `/openapi-public.json` — auto-generated reference

This page is the rendered output of the proxy's filtered OpenAPI schema.
**Do not edit by hand.** Run `npm run gen:api` (or the nightly CI job) to
regenerate.

The generator script is at `xct-docs/scripts/generate-api-reference.mjs`.
It:

1. Fetches `${PROXY_URL}/openapi-public.json`
2. Parses every path + method into a section
3. Renders Markdown with request/response schemas linked to
   `components.schemas`
4. Writes this file in place

## Sections (to be filled by the generator)

When the generator runs, sections appear here as:

```
## POST /v1/chat/completions
### Request
…
### Response 200
…
```

For now, **see the live proxy's `/openapi-public.json` directly** or use
the curl one-liner in [API Reference Overview](./overview.md).

:::note Why a stub page?

The Docusaurus build needs every sidebar entry to point at a file that
exists. This stub keeps the sidebar valid during the docs-bootstrap
period. The first nightly run of `gen:api` replaces it with the real
rendered reference.

:::
