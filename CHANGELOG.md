# Changelog (Fork)

## v0.1.1-exp (2025-09-19)
- Experimental mini-agent (in-code loop) with local tools and optional HTTP tools.
- Router: extracted streaming iterator seam (opt-in via `LITELLM_ROUTER_CORE=extracted`).
- Import-time robustness: fastuuid→uuid fallbacks; MCP soft-dep import guards.
- HTTP diagnostics: short body tail in errors for triage.
- Offline smokes + CI workflow.
