# Gateway authentication

Inbound authentication uses a verifier, identity resolver, and authorizer injected into `Auth`. `Auth::from_config` composes the current master-key implementation. `Auth::new` accepts alternate implementations and an injected clock

`CredentialExtractor` selects a token or transport credential from HTTP request parts. The default `Bearer` extractor is strict; a host can supply another extractor with `Auth::with_extractor` for a separate route profile. A transport credential is only a selection marker, never proof of identity

`Authenticator` receives that selected credential and HTTP request parts. Transport verifiers must obtain proof from trusted server extensions, not client-supplied identity headers. It returns `VerifiedIdentity` only after verification. Request parts provide headers and trusted transport extensions for integrations; the verifier must validate any client-controlled claims before treating them as identity. There is no fallback chain after verification failure

`IdentityResolver` maps verified identity into an internal principal and permissions. The caller retains the original authentication evidence and credential restrictions independently of that mapping. Principals include an authority, subject, and kind. External subjects must remain authority-scoped unless the resolver explicitly maps them to a shared internal identity

`AuthenticatedRequest::authorize` checks expiry and intersects resolved permissions with credential restrictions before calling the injected authorizer. The authorizer may impose additional policy but cannot expand either permission set. The returned operation grant binds the exact authenticated caller instance and access request. Consuming it for another caller or target fails

`Permissions::All`, `None`, and `Only` provide the initial policy vocabulary. `Only` matches exact typed access requests. Model requests contain both the public name and resolved deployment model. MCP requests contain the resolved server ID and upstream operation, including bare tool/prompt names and resource URIs. More expressive policy adapters can be added without changing credential verification

The Axum `authenticate` middleware currently accepts one Authorization header using the existing Bearer format. Missing, duplicate, empty, or malformed credentials fail. Authentication replaces any preexisting caller extension and checks the method and matched route before dispatch. Handlers extract `AuthenticatedRequest` and authorize their parsed operation before calling a provider. Missing authenticated context fails closed

Authentication evidence and principals contain no raw token, password, request body, or mutable accounting state. Session ownership is scoped by principal authority, subject, verifier, and credential ID, so separate credentials do not silently share an MCP session. Credential rotation may retain ownership when the verifier preserves a stable credential ID. Scope and expiry checks still run for each operation

Failures distinguish invalid or expired credentials, forbidden operations, unavailable authentication services, and missing server configuration/context. HTTP adapters map those outcomes to status codes; inference keeps its API-specific error envelopes

Virtual-key storage, JWT verification, OAuth2 introspection, trusted-proxy validation, SSO, and custom Python hook adapters are not implemented here yet. They should supply these contracts rather than bypassing the shared authorization boundary. Request-body-dependent custom hooks will need a bounded endpoint adapter after parsing

Budget reservations, rate limits, usage settlement, retries, and cancellation accounting belong after authorization in the execution lifecycle. This foundation adds no accounting backend or cached authorization decisions. Future reservations must bind the same caller and resolved operation and settle idempotently across streaming completion and cancellation
