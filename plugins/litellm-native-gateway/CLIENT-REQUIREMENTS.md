# Required Claude Code support

These are proposed contracts, not settings implemented by this plugin or by Claude Code today.

## Minimum supported-client change

Add a core inference gateway destination separate from the native feature-service origin. An organization-required destination must apply in the final inference sender even if a plugin is absent, fails to install, is rejected, times out, or is unloaded. Authentication must use the gateway identity for inference and native OAuth for native services. The optional plugin can provide installation and diagnostics; it must not be the enforcement boundary.

A minimal proposed policy would contain one gateway URL and a required/optional mode. Existing native login and organization identity remain separate. The client must fetch or use valid managed policy before its first inference; missing required policy or credentials must stop inference. Safe mode, bare mode, overrides, resume, child processes and alternate provider selection must not remove this requirement in a managed session. Default behavior for organizations that do not opt in stays unchanged.

The client must classify inference by purpose, not a growing list of native feature URLs. Apply the destination to main turns, auxiliary requests, subagents, compaction, counting, retries and fallbacks. Reject unknown inference transports until supported. Native feature API and WebSocket connections keep their native destinations and authentication. Gateway errors must not trigger native sign-in recovery. Redirect handling must retain the required destination constraint.

## Stronger organization-wide boundary

Core client enforcement does not stop a user from running another client with the same native inference-capable credential. If all account inference must be gateway-only, Anthropic must also support organization policy or scoped credentials that refuse direct inference while permitting native services. A supported, audience-bound delegation contract is needed if a gateway must use each user's subscription. This plugin cannot mint such scopes or change server authorization.

Anthropic-hosted jobs require their own execution-side integration. A local plugin cannot reroute a hosted review worker. Keeping a hosted feature available is different from putting its inference inside LiteLLM budgets.

## Why the tested alternatives are insufficient

- Mandatory installation and the built-in mod guard do not stop missing-module bypass on CLI 2.1.332. Installed mods can also be unloaded after worker crashes.
- `forceRemoteSettingsRefresh` governs policy fetching, not required module execution.
- A core custom-endpoint pin routes without a module, but the resulting base-URL mode disables native Remote Control.
- Gateway-only model aliases can cause a missing-plugin main request to fail upstream. They still send the request outside LiteLLM and do not guard raw inference from other mods. They are not an all-inference security boundary.
- Another optional plugin, a heartbeat or a user-agent header cannot prove that every request used the gateway.

## LiteLLM changes

The minimum server change for configured-provider inference is none: use the existing Messages/counting endpoints, virtual-key authorization, router and spend hooks. This contribution supplies the opt-in client package, managed configuration and tests. Do not put client feature routing into the Anthropic provider or duplicate gateway OAuth/session storage.

If Anthropic introduces a delegation contract, integrate it through existing request authentication and provider credential isolation. Never forward an Anthropic credential to a cross-provider fallback. Retain attribution separately for native subscription usage and configured API credentials. Treat callback/plugin-presence headers as observability, not attestation.

## Release acceptance

Require zero direct inference sends during module deletion, malformed module, safe/bare mode, worker termination, stale policy, gateway timeout, 401/403, redirects and concurrent retry tests. Test every supported CLI/IDE/Desktop runtime and hosted execution boundary. Verify native OAuth refresh and login-expiry recovery separately from gateway session renewal. Compare both the client network path and the gateway ledger; a successful native UI alone is insufficient.

References: [managed mods](https://code.claude.com/docs/en/plugins/mods/admin), [mod HTTP contract](https://code.claude.com/docs/en/plugins/mods/reference), [native authentication](https://code.claude.com/docs/en/authentication), [gateway login](https://code.claude.com/docs/en/claude-apps-gateway), [feature availability](https://code.claude.com/docs/en/feature-availability).
