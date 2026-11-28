# Native Claude gateway plugin (experimental)

This managed Claude Code plugin sends Messages and token-count requests to an existing LiteLLM gateway while retaining native Claude login for native services. It runs inside Claude Code, with no local listener, forward proxy, or custom certificate trust. It reuses LiteLLM's existing Messages endpoint, router, authentication and accounting.

**Not production-ready enforcement.** Claude Code 2.1.332 can continue without a missing routing module. A plugin cannot stop requests when it is absent. Do not use this package to promise mandatory gateway budgets, universal native-feature support, or control of Anthropic-hosted inference. See [client requirements](CLIENT-REQUIREMENTS.md) and [compatibility](COMPATIBILITY.md).

## Installation

An organization administrator supplies the reviewed plugin source, gateway URL and managed mod policy once. Developers retain their normal Claude `/login` and enter their own LiteLLM credential in the plugin's sensitive `gatewayKey` setting. `gatewayUrl` is the only endpoint setting. The plugin does not accept a Claude OAuth token as a gateway key.

Once this marketplace is published at the reviewed repository/ref, installation uses Claude's normal plugin flow:

```text
/plugin install litellm-native-gateway --marketplace YOUR_ORG/YOUR_REVIEWED_REPOSITORY
```

This is a template, not an already released marketplace address. For automatic delivery, merge [managed-settings.example.json](managed-settings.example.json) into Claude Code's organization-managed policy and replace the URL and reviewed commit placeholder. Preserve the built-in guard and review the example's restrictions on other mods. A user-scoped install alone does not receive core inference events under that guard. Organization Plugins & skills installation is not equivalent to managed Claude Code policy.

The gateway remains an ordinary LiteLLM deployment. Authorize the required primary and auxiliary models on each user's key; [gateway-config.example.yaml](gateway-config.example.yaml) shows observed aliases, not an exhaustive list. Keep `drop_params: false` when preserving native request capabilities. Do not set `ANTHROPIC_BASE_URL`, `ANTHROPIC_AUTH_TOKEN`, or `ANTHROPIC_API_KEY` on the client for this mode: those can change native feature eligibility or the active authentication method.

For an existing LiteLLM plugin registry, register a pinned Git subdirectory source through the existing admin-only `POST /claude-code/plugins` endpoint. Point it at `plugins/litellm-native-gateway` in your reviewed repository. Registration is discovery; it does not make the module mandatory in Claude Code. The existing Desktop `installation_preference: required` setting is not a CLI inference-enforcement contract. No database migration, new server route or provider change is needed.

## Behavior

The handler preserves the model, body, beta headers, compression and streaming response. It strips native authorization, cookies and credential handles before adding the separate LiteLLM identity. Requests to native services pass through unchanged. A loaded handler denies missing configuration, invalid credentials, known unsupported inference routes, redirects and handler failures. Gateway 401/403 errors identify LiteLLM rather than asking the user to renew native Claude login.

LiteLLM owns provider credentials, model routing, fallbacks and request accounting. Native OAuth stays owned by Claude Code. The supported plugin authorization handle is first-party-only; this package does not extract native credentials or implement subscription-token delegation. Production gateway credential enrollment/storage/renewal and remote marketplace rollout remain unqualified. The environment fallback `LITELLM_API_KEY` is read on each inference request for automated environments; do not distribute shared user keys in organization policy.

Unknown native-service routes pass through. That helps services evolve, but cannot guarantee that a future inference endpoint is intercepted. Other mods calling unrelated services are outside this handler's inference classification. Native feature availability does not establish that a fallback provider supports every requested model capability.

## Validation

With Node.js 22 or later, from this directory:

```sh
node --test policy.test.mjs
```

The tests verify routing, credential isolation, native-service pass-through and rejection behavior without an account or network. Runtime checks on the vendor CLI, native account eligibility, and cloud-worker behavior are compatibility qualification, not invariants owned by LiteLLM. No customer credentials or private test artifacts are included.
