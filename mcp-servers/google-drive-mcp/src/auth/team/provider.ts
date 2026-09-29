// ---------------------------------------------------------------------------
// TeamOAuthProvider — the OAuthServerProvider behind the SDK's mcpAuthRouter.
//
// Division of labor with the SDK (v1.26): the SDK validates redirect_uri
// against the registered client at /authorize, authenticates the client at
// /token, and verifies PKCE S256 itself via challengeForAuthorizationCode
// (skipLocalPkceValidation stays unset). Everything the SDK deliberately does
// NOT do is enforced here: binding codes to the client that exchanges them,
// re-validating redirect_uri and resource at exchange time, single-use
// consumption, refresh-token rotation with reuse detection, and mapping the
// bearer token to a team user.
// ---------------------------------------------------------------------------

import type { Response } from 'express';
import type { OAuthRegisteredClientsStore } from '@modelcontextprotocol/sdk/server/auth/clients.js';
import {
  InvalidClientMetadataError,
  InvalidGrantError,
  InvalidScopeError,
  InvalidTargetError,
  InvalidTokenError,
} from '@modelcontextprotocol/sdk/server/auth/errors.js';
import type {
  AuthorizationParams,
  OAuthServerProvider,
} from '@modelcontextprotocol/sdk/server/auth/provider.js';
import type { AuthInfo } from '@modelcontextprotocol/sdk/server/auth/types.js';
import type {
  OAuthClientInformationFull,
  OAuthTokenRevocationRequest,
  OAuthTokens,
} from '@modelcontextprotocol/sdk/shared/auth.js';
import type { TeamConfig } from './config.js';
import type { GoogleIdp } from './googleIdp.js';
import {
  mintAccessToken,
  mintRefreshToken,
  mintState,
  newFamilyId,
  sha256Hex,
} from './tokens.js';
import {
  PENDING_AUTH_TTL_MS,
  REFRESH_GRACE_MS,
  REFRESH_TOKEN_TTL_MS,
  TOMBSTONE_TTL_MS,
  TeamStore,
  TeamStoreCapacityError,
} from './types.js';

export interface TeamProviderDeps {
  store: TeamStore;
  idp: GoogleIdp;
  config: TeamConfig;
}

export class TeamOAuthProvider implements OAuthServerProvider {
  constructor(private readonly deps: TeamProviderDeps) {}

  get clientsStore(): OAuthRegisteredClientsStore {
    const { store, config } = this.deps;
    return {
      getClient: (clientId: string) => store.getClient(clientId),
      // Presence of registerClient makes the SDK mount /register (DCR). The
      // SDK generates client_id/client_secret before calling; we only gate and
      // persist.
      registerClient: async (client: OAuthClientInformationFull) => {
        if (config.allowedRedirectUris.length > 0) {
          for (const uri of client.redirect_uris) {
            if (!config.allowedRedirectUris.includes(uri)) {
              throw new InvalidClientMetadataError(
                `redirect_uri is not in this server's allowlist: ${uri}`,
              );
            }
          }
        }
        try {
          await store.saveClient(client);
        } catch (err) {
          if (err instanceof TeamStoreCapacityError) {
            throw new InvalidClientMetadataError(err.message);
          }
          throw err;
        }
        return client;
      },
    };
  }

  /**
   * First hop inbound: stash everything the final redirect will need, keyed by
   * a fresh state for the Google hop, and send the user to Google's consent
   * screen. The MCP client's own `state` rides along in the pending record.
   */
  async authorize(
    client: OAuthClientInformationFull,
    params: AuthorizationParams,
    res: Response,
  ): Promise<void> {
    const scopes = this.resolveRequestedScopes(params.scopes);
    const googleState = mintState();
    const now = Date.now();
    await this.deps.store.savePendingAuthorization({
      googleState,
      clientId: client.client_id,
      clientState: params.state,
      codeChallenge: params.codeChallenge,
      redirectUri: params.redirectUri,
      scopes,
      resource: params.resource?.href,
      createdAt: now,
      expiresAt: now + PENDING_AUTH_TTL_MS,
    });
    res.redirect(this.deps.idp.buildConsentUrl(googleState));
  }

  /**
   * Read (never consume) the stored PKCE challenge for the SDK's local S256
   * verification. The store burns the code after too many lookups, bounding
   * brute-force of a stolen code's verifier.
   */
  async challengeForAuthorizationCode(
    client: OAuthClientInformationFull,
    authorizationCode: string,
  ): Promise<string> {
    const code = await this.deps.store.peekAuthorizationCode(sha256Hex(authorizationCode));
    if (!code || code.clientId !== client.client_id) {
      throw new InvalidGrantError('Invalid or expired authorization code.');
    }
    return code.codeChallenge;
  }

  async exchangeAuthorizationCode(
    client: OAuthClientInformationFull,
    authorizationCode: string,
    _codeVerifier?: string,
    redirectUri?: string,
    resource?: URL,
  ): Promise<OAuthTokens> {
    const code = await this.deps.store.consumeAuthorizationCode(sha256Hex(authorizationCode));
    if (!code) {
      throw new InvalidGrantError('Invalid or expired authorization code.');
    }
    // The SDK authenticates the client but does not check code ownership, and
    // passes redirect_uri through unvalidated at the token step — both are
    // this provider's job. PKCE alone does not bind client identity.
    if (code.clientId !== client.client_id) {
      throw new InvalidGrantError('Invalid or expired authorization code.');
    }
    // redirect_uri and resource are optional in the token request (OAuth 2.1
    // drops redirect_uri; RFC 8707 treats resource as per-request). Enforce
    // equality only when the client actually sends the parameter — the code is
    // already bound to its client and PKCE, and the token is always minted with
    // the stored values. Rejecting a legal omission would dead-end a
    // spec-compliant client (the single-use code is already consumed).
    if (redirectUri !== undefined && redirectUri !== code.redirectUri) {
      throw new InvalidGrantError('redirect_uri does not match the authorization request.');
    }
    if (resource !== undefined && resource.href !== code.resource) {
      throw new InvalidTargetError('resource does not match the authorization request.');
    }
    return this.mintTokenPair({
      clientId: client.client_id,
      sub: code.sub,
      accessScopes: code.scopes,
      refreshScopes: code.scopes,
      resource: code.resource,
      familyId: newFamilyId(),
    });
  }

  async exchangeRefreshToken(
    client: OAuthClientInformationFull,
    refreshToken: string,
    scopes?: string[],
    resource?: URL,
  ): Promise<OAuthTokens> {
    const record = await this.deps.store.getRefreshToken(sha256Hex(refreshToken));
    const now = Date.now();
    if (!record || record.clientId !== client.client_id || record.expiresAt <= now) {
      throw new InvalidGrantError('Invalid or expired refresh token.');
    }
    if (record.supersededByHash !== undefined) {
      const inGrace = record.graceUntil !== undefined && now <= record.graceUntil;
      if (!inGrace) {
        // Reuse of a rotated token past its grace window is a theft signal
        // (OAuth 2.1): revoke the entire family and force a full re-auth.
        await this.deps.store.revokeTokenFamily(record.familyId);
        throw new InvalidGrantError('Invalid or expired refresh token.');
      }
    }
    // Enforce resource equality only when the client sends it (RFC 8707,
    // per-request); otherwise inherit the grant's bound resource.
    if (resource !== undefined && resource.href !== record.resource) {
      throw new InvalidTargetError('resource does not match the token grant.');
    }
    // Scope narrowing only (RFC 6749 §6): a request may shrink the scope of the
    // ISSUED access token but never the underlying grant. The narrowing applies
    // to the access token only — the rotated refresh token keeps the full grant
    // so a later request can still request the original scopes.
    let effectiveScopes = record.scopes;
    if (scopes && scopes.length > 0) {
      const granted = new Set(record.scopes);
      const unknown = scopes.filter((s) => !granted.has(s));
      if (unknown.length > 0) {
        throw new InvalidScopeError(`Scopes exceed the original grant: ${unknown.join(' ')}`);
      }
      effectiveScopes = scopes;
    }

    const pair = await this.mintTokenPair({
      clientId: record.clientId,
      sub: record.sub,
      accessScopes: effectiveScopes,
      refreshScopes: record.scopes,
      resource: record.resource,
      familyId: record.familyId,
    });
    if (record.supersededByHash === undefined) {
      // Rotate: the old token stays exchangeable for a short grace window
      // (tolerates a client that lost the rotation response), then becomes a
      // reuse-detection tombstone. Bound its retention to TOMBSTONE_TTL_MS so
      // hourly rotations don't accumulate tombstones for the full grant TTL.
      await this.deps.store.updateRefreshToken({
        ...record,
        expiresAt: Math.min(record.expiresAt, now + TOMBSTONE_TTL_MS),
        supersededByHash: sha256Hex(pair.refresh_token!),
        graceUntil: now + REFRESH_GRACE_MS,
      });
    }
    return pair;
  }

  async verifyAccessToken(token: string): Promise<AuthInfo> {
    const tokenHash = sha256Hex(token);
    const record = await this.deps.store.getAccessToken(tokenHash);
    if (!record) {
      throw new InvalidTokenError('Invalid access token.');
    }
    if (record.expiresAt <= Date.now()) {
      await this.deps.store.deleteAccessToken(tokenHash);
      throw new InvalidTokenError('Access token has expired.');
    }
    const user = await this.deps.store.getUser(record.sub);
    if (!user || user.needsReauth) {
      // The Google grant behind this token is gone; 401 here makes the MCP
      // client re-run the OAuth flow, which repairs the grant.
      throw new InvalidTokenError('Authorization has been revoked. Sign in again.');
    }
    return {
      token,
      clientId: record.clientId,
      scopes: record.scopes,
      // requireBearerAuth compares against Date.now()/1000 — seconds, not ms.
      expiresAt: Math.floor(record.expiresAt / 1000),
      resource: record.resource ? new URL(record.resource) : undefined,
      extra: { sub: record.sub, email: user.email },
    };
  }

  /** Presence makes the SDK mount /revoke. Unknown tokens are a no-op per spec. */
  async revokeToken(
    client: OAuthClientInformationFull,
    request: OAuthTokenRevocationRequest,
  ): Promise<void> {
    const tokenHash = sha256Hex(request.token);
    const access = await this.deps.store.getAccessToken(tokenHash);
    if (access) {
      if (access.clientId === client.client_id) {
        await this.deps.store.deleteAccessToken(tokenHash);
      }
      return;
    }
    const refresh = await this.deps.store.getRefreshToken(tokenHash);
    if (refresh && refresh.clientId === client.client_id) {
      await this.deps.store.revokeTokenFamily(refresh.familyId);
    }
  }

  // -------------------------------------------------------------------------
  // Internals
  // -------------------------------------------------------------------------

  private resolveRequestedScopes(requested: string[] | undefined): string[] {
    const configured = this.deps.config.advertisedScopes;
    if (!requested || requested.length === 0) return [...configured];
    const allowed = new Set(configured);
    const scopes = requested.filter((s) => allowed.has(s));
    if (scopes.length === 0) {
      throw new InvalidScopeError(
        `None of the requested scopes are supported. Supported: ${configured.join(' ')}`,
      );
    }
    return scopes;
  }

  private async mintTokenPair(grant: {
    clientId: string;
    sub: string;
    /** Scope of the issued access token (may be a narrowed subset). */
    accessScopes: string[];
    /** Scope stored on the refresh token — the full grant, so future refreshes
     * can still request the original scopes even after a one-time narrowing. */
    refreshScopes: string[];
    resource?: string;
    familyId: string;
  }): Promise<OAuthTokens> {
    const now = Date.now();
    const accessToken = mintAccessToken();
    const refreshToken = mintRefreshToken();
    await this.deps.store.saveAccessToken({
      tokenHash: sha256Hex(accessToken),
      clientId: grant.clientId,
      sub: grant.sub,
      scopes: grant.accessScopes,
      expiresAt: now + this.deps.config.tokenTtlMs,
      familyId: grant.familyId,
      resource: grant.resource,
    });
    await this.deps.store.saveRefreshToken({
      tokenHash: sha256Hex(refreshToken),
      clientId: grant.clientId,
      sub: grant.sub,
      scopes: grant.refreshScopes,
      familyId: grant.familyId,
      createdAt: now,
      expiresAt: now + REFRESH_TOKEN_TTL_MS,
      resource: grant.resource,
    });
    return {
      access_token: accessToken,
      token_type: 'bearer',
      expires_in: Math.floor(this.deps.config.tokenTtlMs / 1000),
      scope: grant.accessScopes.join(' '),
      refresh_token: refreshToken,
    };
  }
}
