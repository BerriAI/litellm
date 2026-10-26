# Auto Router setup QA

Real dashboard and proxy with an isolated local database and synthetic deployment records. No upstream inference requests were made.

Before is main `b7a2226f438`. After is `6585496b7c5`. Comparison captures use the same URL, viewport, and solver fixtures. The after capture also has the newly saved router, so the generated name advances to `auto-router-2`. Red outlines mark the form for comparison only.

| State | Evidence |
| --- | --- |
| Original setup | [Before](before-default.png) |
| Goal-first setup | [After](after-default.png) |
| Unannotated default | [Default](after-review.png) |
| Only openai/* enabled | [Concrete Luna + Astra recommendations](wildcard-openai.png) |
| UI save and list refresh | [Created wildcard-backed router](wildcard-saved.png) |

[Saved wildcard configuration](wildcard-saved-config.json) was read back from the real API after UI creation: SIMPLE/MEDIUM use openai/gpt-6-luna; COMPLEX/REASONING use openai/gpt-6-astra; classifier_type is heuristic (v1).

[Deployment fixtures](fixtures.json) contain model identities only. For the wildcard case, all 12 explicit fixtures were temporarily blocked, leaving only openai/* enabled. Wildcard expansion may use the LiteLLM catalog; it does not verify upstream account entitlements.

## Validation

241 focused tests passed (133 resolver, 108 form integration), plus make check and the production build including TypeScript. Following a lint-only predicate extraction, the affected team-selection regression was rerun and passed. New alias and team-selection regressions failed before their fixes. Earlier review fixes cover Terra, equal-rank cross-family fallback, cached inventory after refetch errors, and Gemini/Grok recognition.

Catalog-driven coverage recognizes all 108 standard chat Opus entries, including all 19 Opus 5.5 variants. Latest selection compares numeric releases and prefers undated aliases within a release. Saved configurations remain explicit; there is no background model upgrade.

[Earlier edge-case gallery at 41cad716471](https://github.com/BerriAI/litellm/blob/46d1641c167a5a73e8a6d83f0c1b270299887bb8/README.md) covers Bedrock provider detection, cross-family fallback, one-model recovery, exhausted Heuristic v2 allowance, and narrow layouts. Those captures are historical, not recaptures of the final review fixes.
