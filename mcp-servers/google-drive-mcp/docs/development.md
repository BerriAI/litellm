# Development

## Project structure

```text
google-drive-mcp/
├── src/
│   ├── index.ts                 # MCP server, transports, routing, and CLI
│   ├── auth.ts                  # Authentication-system assembly
│   ├── auth/                    # OAuth, account storage, external auth, and team mode
│   ├── tools/                   # Drive, Docs, Sheets, Slides, and Calendar tools
│   ├── utils/                   # CLI, retry, and image-upload helpers
│   ├── download-file.ts         # Download helper
│   └── types.ts                 # Shared types
├── test/                        # Unit, integration, schema, and auth tests
├── scripts/
│   ├── build.js                 # esbuild bundle
│   └── docker-mcp.sh            # Reusable Docker stdio wrapper
├── Dockerfile
├── gcp-oauth.keys.example.json
├── package.json
├── tsconfig.json
└── README.md
```

`dist/` and `.tmp-test/` are generated and are not source directories.

## Install dependencies

```bash
npm ci
```

Use `npm install` when intentionally changing dependencies and the lockfile.

## Scripts

| Command | Purpose |
|---|---|
| `npm run typecheck` | Run TypeScript without emitting files |
| `npm run build` | Type-check and bundle `dist/index.js` |
| `npm run watch` | Rebuild when source files change |
| `npm run lint` | Run TypeScript checks and ESLint |
| `npm run lint:types` | Run only TypeScript checks |
| `npm run lint:eslint` | Run only ESLint |
| `npm test` | Build the test tree and run all tests |
| `npm run test:unit` | Run top-level unit tests |
| `npm run test:integration` | Run integration and schema tests |
| `npm start` | Start the compiled server |
| `npm run auth` | Run authentication through the compiled server |
| `npm run registry:validate` | Validate `server.json` with the official MCP Registry API |
| `npm run registry:verify` | Verify the published MCP Registry entry |

`npm run prepare` runs the build automatically during package preparation.

Maintainers should follow the [MCP Registry publishing contract](https://github.com/piotr-agier/google-drive-mcp/blob/master/docs/registry-publishing.md) when preparing releases. That guide is maintainer-only and is not part of the published npm package, so the link is absolute.

## Local MCP configuration

Build the project, then point a client at the absolute `dist/index.js` path. See [Local build](clients.md#local-build).

## Tests

Tests use Node's built-in test runner. `npm test` compiles tests into `.tmp-test/`, creates the production bundle, and runs with `MCP_TESTING=1` so importing the server does not start a live transport.

New or changed tools should update schema/registry coverage as well as focused behavior tests. Documentation changes should keep the internal-link and tool-reference checks passing.

## Contributing

1. Fork the repository and create a focused branch.
2. Make the change with tests and documentation.
3. Run `npm run lint`, `npm run build`, and `npm test`.
4. Push the branch and open a pull request.

Report defects and feature requests in [GitHub Issues](https://github.com/piotr-agier/google-drive-mcp/issues).

