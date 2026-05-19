# xct-litellm-docs

Documentation site for the xct-litellm capability provider.

## Stack

- [Docusaurus 3](https://docusaurus.io/) static site
- Markdown + MDX
- Lives in the main repo at `xct-docs/`

## Local dev

```bash
cd xct-docs
npm install      # or pnpm install
npm start        # dev server at http://localhost:3010
```

## Build

```bash
npm run build    # produces ./build
npm run serve    # preview ./build
```

## Regenerate API reference

```bash
PROXY_URL=https://api.xct.test npm run gen:api
```

Reads `${PROXY_URL}/openapi-public.json` and writes
`docs/api-reference/openapi-public.md`. CI runs this nightly.

## Content layout

```
docs/
├── intro.md
├── concepts/         5 markdown files — entity classes + tenancy
├── quickstart/       3 per-app guides (xct-chat / xct-home / xct-agent-desktop)
├── recipes/          10 short how-tos
└── api-reference/    overview + auto-generated openapi-public.md
```

## Deployment

`build/` is a static site. Drop into any static host:

- Cloudflare Pages (recommended — github webhook deploy)
- Netlify
- S3 + CloudFront

Domain target: `docs.xct.ai`.

## Authoring conventions

- One sentence per concept; tables for enumeration
- Code blocks always show the language (`python`, `typescript`, `bash`, `http`)
- Cross-link with `[label](./other.md)` — Docusaurus rewrites
- Avoid jargon without first paragraph definition
- No screenshots unless a UX flow genuinely depends on them
