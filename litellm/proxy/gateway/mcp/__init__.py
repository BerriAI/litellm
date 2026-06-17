"""MCP Gateway v2 — clean-room subdomain (Phase 0 mini-chassis).

Written v2-native: nothing here imports from v1 (the rest of
`litellm.proxy._experimental.mcp_server.*`). v1 only ever reaches v2 through a thin
adapter built in Phase 1, never the other way round.

Phase 0 scope = the typed OAuth-credential seam only (`oauth/types.py`), the vendored
`Result` (`result.py`), and the basedpyright match-exhaustiveness spike
(`_spike_exhaustiveness.py`). No transport, registry, or CI/semgrep/composition-root
infrastructure — those land in Phase 2 (S0).

House style is adopted from the sibling v2 effort in `litellm/translation/` (see its
`CLAUDE.md`): product types are frozen dataclasses / frozen pydantic models, sum types are
Expression `@tagged_union` discriminated on a `Literal` `tag` (match on `self.tag`,
`assert_never` the tail), failures are values (`Result`, nothing raises in-tree).
"""
