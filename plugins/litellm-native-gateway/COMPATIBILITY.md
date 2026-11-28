# Compatibility qualification

As checked on 2026-11-28, this package is experimental. A successful row covers only the named workflow, not every feature lifecycle or provider. Account-specific evidence is outside the public contribution.

| Check | Evidence and limit |
| --- | --- |
| Managed routing | CLI 2.1.332: primary requests, continued tool turns and another mod's known inference routed to a scripted gateway. Credentials isolated; missing-key and redirect cases denied. |
| Missing module | CLI 2.1.332 still sent directly. Release blocker. |
| Native token refresh | The native client owns this lifecycle. Natural expiry, revocation and maximum login lifetime require deployment qualification. No automatic subscription delegation is provided. |
| Native services | Pass-through is covered by handler tests. Individual service workflows require account and runtime qualification. |
| Remote Control | Requires native login and organization eligibility. Production lifecycle, reconnection and denial behavior require deployment qualification. |
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
| Remote Control | Requires browser-to-local success, gateway denial, reconnection and cleanup qualification. |
| Chrome extension | Command/tools discovered; no connected-browser action verified. |
| CLI computer use | Supplied Team account is ineligible for the current CLI preview. |
| Artifacts | Read, creation/update and generated-content workflows require separate qualification. |
| Voice dictation | Command discovered; microphone-to-transcript-to-gateway workflow not tested with this plugin. |

Other provider-dependent capabilities, IDEs, compaction, reconnect, cancellation, cloud workers and long-running refresh remain separate qualification work. Do not present discovery as execution or a subscription feature as being inside gateway budgets merely because it is visible.

The contribution is not certified against a multiworker production gateway or every account entitlement. Native subscription delegation and provider API credentials are separate deployment choices.

[Anthropic feature requirements](https://code.claude.com/docs/en/feature-availability) · [native OAuth lifecycle](https://code.claude.com/docs/en/authentication)
