# API reference preview

Open `/reference` on the proxy to use the branded Scalar reference. Swagger stays
at its configured URL. The preview uses the same OpenAPI document, including
enterprise filtering, custom titles and descriptions, and deployment root paths.
Disabling docs or OpenAPI also disables the preview.

The top picker scopes navigation and search to AI Routes, Management Routes, or
MCP Routes. Sidebar order can use common workflows first or alphabetical order.
The category is included in the URL so category-specific links can be shared.
Unclassified operations stay visible under Management Routes. Category changes
reuse the same renderer to keep credentials in memory without browser storage.

Run the category behavior checks with
`node --test tests/unit/proxy/api_reference/test_route_categories.mjs`.

The renderer, logo, and fonts are served locally. The browser sends test requests
directly to the selected API server. Scalar's hosted proxy, telemetry, AI agent,
and persistent credential storage are disabled.

## Bundled assets

- `vendor/scalar-1.76.0.js` is `dist/browser/standalone.js` from
  [`@scalar/api-reference@1.76.0`](https://registry.npmjs.org/@scalar/api-reference/-/api-reference-1.76.0.tgz).
  The npm archive's SHA-512 integrity was verified before extraction.
  Bundle SHA-256: `dcb027152bf20bf2f6dcd66e8f9fb073c1a9e274cf0b94b939a7d99ab8947060`.
  The MIT license is in `vendor/scalar-LICENSE`.
- Barlow Condensed (700), DM Sans (400/600), and IBM Plex Mono (400)
  are distributed under the SIL Open Font License. Licenses are in `fonts/`.
  These match the fonts on the LiteLLM Handoff site.
- The two logo SVGs are from the official LiteLLM logo export.

When updating Scalar, pin the release, verify the archive integrity, retain its
license, update the filename in `index.html`, and test search, authentication,
request execution, mobile layout, and deployment prefixes against a live proxy.
Keep the mount element distinct from Scalar's reserved `api-reference` ID.

If a deployment lacks the lazy-route schema snapshot, the preview links readers
to Swagger to load the additional endpoint groups. It does not eagerly warm
services when the reference page opens.
