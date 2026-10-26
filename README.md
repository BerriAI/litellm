# Auto Router setup QA

Real dashboard and proxy with an isolated local database and synthetic deployment records. No upstream inference requests were made

Before is main `f96200bcea3`. After is `41cad716471`. Both comparison screenshots use the same URL, viewport, and deployed models. Red outlines only mark the form for comparison and are not part of the change

## Screenshots

| State | Evidence |
| --- | --- |
| Original setup | [Before](before-default.png) |
| Goal-first setup | [After](after-default.png) |
| Polished layout | [Default](polished-default.png) |
| Newer Sol on Bedrock preferred over older native Sol | [OpenAI Best quality](openai-bedrock.png) |
| One model from each of two families | [Cross-family fallback](mixed-family-fallback.png) |
| Only one supported model available | [Creation blocked with Advanced recovery](one-model.png) |
| Existing router consumes the Heuristic v2 allowance | [Exhausted option disabled, standard available](license-exhausted.png) |
| Narrow viewport | [Stacked models](mobile.png) |
| Real save and list refresh | [Created routers](saved-router.png) |

## Verification

Catalog-driven regression coverage recognizes all 108 standard chat Opus entries, including all 19 Opus 5.5 provider variants. Special fast and thinking variants remain in Advanced

129 resolver tests and 104 form integration tests passed, plus `make check` and `npm run build`

The focused tests cover all three goals, four model families, provider aliases and base-model metadata, numeric releases and dated snapshots, unavailable and blocked models, non-chat models, mixed-version groups, recursive routers, same-family substitutions, cross-family fallback, no valid pair, manual overrides, error/retry, team scope, and advanced configuration

Standalone typecheck with development-generated routes reports the same existing `AgentControlPlaneView` export error on base and branch. The production build and its TypeScript phase pass

[Deployment fixtures](fixtures.json) contain model identities only. [Saved configurations](saved-configs.json) were read back from the real API after UI creation

The newest recognized version is resolved when creating a router. Saved configurations remain explicit. Unknown naming conventions remain available in Advanced. Goals select model pairs within families, not measured global cost or quality optima
