# MCP Registry publishing

The official MCP Registry identity for this project is
`io.github.piotr-agier/google-drive-mcp`. The npm package advertises that name
through `package.json#mcpName`, and `server.json` describes the supported
Registry installation path: the public npm package running over stdio.

## Release contract

Registry versions are immutable. Every release must use the same exact semantic
version in all of these places:

- `package.json` and the root package entries in `package-lock.json`
- `server.json` and its npm package entry
- the published npm artifact
- the Git tag and GitHub release
- the MCP Registry record

The npm artifact must be published before its Registry record because the
Registry verifies `mcpName` from the public package. Run
`npm run registry:validate` before publishing; the release workflow repeats that
validation, publishes npm, then publishes Registry metadata using GitHub OIDC.
It finally retrieves the exact name and version and checks the public record
against `server.json`.

When the Registry schema or publisher changes, update the schema URI in
`server.json`, the pinned publisher version and checksum in the release
workflow, and this document together.

DIST-002 and DIST-003 must reuse the Registry name and exact release-version
contract above when adding MCPB metadata and client installation recipes.
