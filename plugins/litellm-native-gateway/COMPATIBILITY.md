# Compatibility qualification

As checked on 2026-11-28, this package is experimental. A successful row covers only the named workflow, not every feature lifecycle or provider. Live private-account evidence is retained outside the public package.

| Check | Evidence and limit |
| --- | --- |
| Managed routing | CLI 2.1.332: primary requests, continued tool turns and another mod's known inference routed to a scripted gateway. Credentials isolated; missing-key and redirect cases denied. |
| Missing module | CLI 2.1.332 still sent directly. Release blocker. |
| Native token refresh | Official macOS CLI 2.1.332 refreshed after simulated local access-token expiry. Access and refresh tokens changed and expiry advanced in Keychain. Native usage command used no model. No refresh token exported. This was not a long-running plugin session crossing natural expiry. |
| Post-refresh native service | Interactive Linux CLI 2.1.332 with the routing module loaded read an existing private Artifact successfully; real LiteLLM recorded auxiliary Haiku inference. |
| Remote Control | Live browser/local success and gateway-key-revocation denial passed on CLI 2.1.332 after native renewal. LiteLLM recorded Opus and auxiliary Haiku inference. Native connection remained connected when the revoked key stopped the next request. The temporary organization setting was restored to Off. |
| Cross-provider routing | Real LiteLLM with scripted Anthropic/OpenAI endpoints exercised fallback. This is not live GitHub Copilot or native feature equivalence. |

## Subscription feature inventory

| Feature | End-to-end status for this plugin |
| --- | --- |
| Cloud sessions and mobile | Local plugin does not govern hosted workers; earlier transport's cloud success is not plugin qualification. Mobile untested. |
| Desktop | Not tested on the required application/runtime. |
| Cloud scheduled tasks | Native command discovered; earlier transport completed a task. Hosted inference through this plugin not established. |
| Ultrareview | Command discovered; paid execution excluded. |
| Code Review | Setup inspected; paid execution excluded. |
| PR Steward | Paid execution excluded. |
| Claude Approvals | Activation and paid checks excluded. |
| Remote Control | Bounded browser-to-local workflow and revoked-key denial passed; see version-scoped results above. |
| Chrome extension | Command/tools discovered; no connected-browser action verified. |
| CLI computer use | Supplied Team account is ineligible for the current CLI preview. |
| Artifacts | Existing private artifact read passed. Creation/update with generated model content not certified. |
| Voice dictation | Command discovered; microphone-to-transcript-to-gateway workflow not tested with this plugin. |

Other provider-dependent capabilities, IDEs, compaction, reconnect, cancellation, cloud workers and long-running refresh remain separate qualification work. Do not present discovery as execution or a subscription feature as being inside gateway budgets merely because it is visible.

The live checks used an existing LiteLLM test checkout and the identical routing module. They do not certify this PR's gateway base commit, a multiworker production deployment, or automatic subscription delegation. The test gateway used a temporary configured native access token. Native refresh tokens stayed in the native client store.

[Anthropic feature requirements](https://code.claude.com/docs/en/feature-availability) · [native OAuth lifecycle](https://code.claude.com/docs/en/authentication)
