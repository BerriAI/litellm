// ---------------------------------------------------------------------------
// Team mode storage contract.
//
// Team mode turns the HTTP transport into an OAuth 2.1 authorization server
// (two-hop: MCP client ⇄ this server ⇄ Google). This module defines the
// records that flow through that exchange and the pluggable store that holds
// them. Two implementations ship in core: in-memory (tests, ephemeral
// deployments) and a JSON file (the default; Google refresh tokens must
// survive restarts).
//
// Lifetimes and integrity rules the stores enforce:
// - Pending authorizations and authorization codes are short-lived and kept
//   in-memory in BOTH implementations — losing them on restart costs one
//   login retry. Their single-use `consume*` operations are atomic
//   (synchronous Map get+delete), which is what prevents double-minting.
// - MCP access/refresh tokens are stored as SHA-256 hashes only; a stolen
//   store file yields no usable bearer tokens. Google refresh tokens are
//   necessarily cleartext (they must be replayed to Google verbatim) — the
//   file store relies on 0600 permissions, mirroring tokens.json.
// - Hard caps bound every collection so an unauthenticated attacker cannot
//   grow the store without bound.
// ---------------------------------------------------------------------------

import type { OAuthClientInformationFull } from '@modelcontextprotocol/sdk/shared/auth.js';

/** TTL for a pending two-hop authorization (user is at the Google consent screen). */
export const PENDING_AUTH_TTL_MS = 10 * 60 * 1000;
/** TTL for a server-issued authorization code. */
export const AUTH_CODE_TTL_MS = 60 * 1000;
/** Absolute lifetime of an MCP refresh token (renewed on rotation). */
export const REFRESH_TOKEN_TTL_MS = 30 * 24 * 60 * 60 * 1000;
/** After rotation, the superseded refresh token stays exchangeable this long
 * (tolerates the client losing the rotation response). Reuse after the grace
 * window is treated as theft and revokes the whole token family. */
export const REFRESH_GRACE_MS = 60 * 1000;
/** How long a superseded refresh token is retained past rotation purely as a
 * reuse-detection tombstone. Long enough to catch a stolen token replayed after
 * the grace window, short enough to bound tombstone accumulation — an hourly
 * rotating client would otherwise pile up ~720 tombstones over the 30-day grant
 * TTL. Reuse within this horizon still revokes the family; reuse afterwards is
 * indistinguishable from an expired token (a 401 that re-runs the OAuth flow). */
export const TOMBSTONE_TTL_MS = 24 * 60 * 60 * 1000;
/** PKCE-challenge lookups allowed per authorization code before it is burned
 * (bounds brute-forcing a stolen code's verifier at the /token endpoint). */
export const MAX_CHALLENGE_LOOKUPS = 3;

/** Default store caps. Overridable per-store (tests); constants otherwise. */
export interface TeamStoreCaps {
  maxClients: number;
  maxUsers: number;
  maxPendingAuthorizations: number;
  /** Active (non-superseded) token pairs per user; oldest evicted on overflow. */
  maxTokensPerSub: number;
  /** Superseded refresh-token tombstones per user; oldest evicted on overflow.
   * A hard backstop under TOMBSTONE_TTL_MS so a burst of rotations can't grow
   * the store between sweeps. */
  maxTombstonesPerSub: number;
}

export const DEFAULT_TEAM_STORE_CAPS: TeamStoreCaps = {
  maxClients: 100,
  maxUsers: 200,
  maxPendingAuthorizations: 1000,
  maxTokensPerSub: 20,
  maxTombstonesPerSub: 20,
};

/** Thrown when an insert would exceed a hard cap. Callers translate this into
 * an OAuth error / error page; it must never crash the server. */
export class TeamStoreCapacityError extends Error {
  constructor(what: string, cap: number) {
    super(`Team store capacity exceeded: ${what} (limit ${cap}).`);
    this.name = 'TeamStoreCapacityError';
  }
}

/** A two-hop authorization awaiting the Google consent round-trip.
 * Keyed by the state WE send to Google (crypto-random, single-use). */
export interface PendingAuthorization {
  googleState: string;
  clientId: string;
  /** The MCP client's own `state`, echoed verbatim on the final redirect. */
  clientState?: string;
  /** PKCE challenge, persisted verbatim; the SDK performs the S256 check. */
  codeChallenge: string;
  redirectUri: string;
  scopes: string[];
  resource?: string;
  createdAt: number;
  expiresAt: number;
}

/** A server-issued authorization code. Single-use; stored hashed. */
export interface AuthorizationCodeRecord {
  /** sha256 hex of the code; primary key. */
  codeHash: string;
  clientId: string;
  sub: string;
  codeChallenge: string;
  redirectUri: string;
  scopes: string[];
  resource?: string;
  createdAt: number;
  expiresAt: number;
  /** Incremented by peekAuthorizationCode; the record is burned past
   * MAX_CHALLENGE_LOOKUPS. */
  challengeLookups: number;
}

/** A team member, keyed by their Google-stable `sub`. */
export interface TeamUserRecord {
  sub: string;
  email: string;
  /** Cleartext by necessity — replayed to Google verbatim. */
  googleRefreshToken: string;
  googleAccessToken?: string;
  /** ms epoch (AccountRecord convention). */
  googleTokenExpiry?: number;
  grantedScopes: string[];
  /** Set when Google reported invalid_grant; cleared on re-authorization.
   * verifyAccessToken rejects flagged users. */
  needsReauth?: boolean;
  /** ISO timestamps (AccountRecord convention). */
  createdAt: string;
  updatedAt: string;
}

/** A minted MCP access token. Keyed by sha256 of the token. */
export interface AccessTokenRecord {
  tokenHash: string;
  clientId: string;
  sub: string;
  scopes: string[];
  /** ms epoch. Convert to SECONDS for AuthInfo.expiresAt — requireBearerAuth
   * compares against Date.now()/1000; milliseconds make tokens immortal. */
  expiresAt: number;
  /** Rotation family shared with the refresh token minted alongside it, so
   * family revocation kills both. */
  familyId: string;
  /** RFC 8707 resource the token was bound to at the code exchange. */
  resource?: string;
}

/** A minted MCP refresh token. Keyed by sha256 of the token.
 * Superseded records are kept until expiresAt as reuse-detection tombstones. */
export interface RefreshTokenRecord {
  tokenHash: string;
  clientId: string;
  sub: string;
  scopes: string[];
  familyId: string;
  createdAt: number;
  expiresAt: number;
  /** RFC 8707 resource the token family was bound to. */
  resource?: string;
  /** Set on rotation: hash of the successor token. */
  supersededByHash?: string;
  /** Superseded token remains exchangeable until this instant. */
  graceUntil?: number;
}

/**
 * Pluggable team-mode store.
 *
 * Contract notes:
 * - `consume*` methods are atomic single-use: exactly one of N concurrent
 *   callers receives the record; expired records are treated as absent.
 * - `peekAuthorizationCode` is non-consuming (the SDK reads the PKCE challenge
 *   BEFORE the exchange) but counts lookups and burns the code past
 *   MAX_CHALLENGE_LOOKUPS.
 * - upsertUser throws TeamStoreCapacityError at maxUsers; saveClient,
 *   saveAccessToken, and saveRefreshToken instead evict the least-recently-used
 *   / oldest entry when their cap is reached, so an unauthenticated flood can
 *   never permanently lock out legitimate registrations or logins.
 */
export interface TeamStore {
  /** Load/validate backing storage. Must be called once before use. */
  init(): Promise<void>;

  /** Drain any pending durable writes so nothing queued is lost on shutdown.
   * In-memory stores resolve immediately. */
  flush(): Promise<void>;

  getClient(clientId: string): Promise<OAuthClientInformationFull | undefined>;
  /** Persist a registered client. At the cap, evicts the least-recently-used
   * client (by last registration/token-issuance) rather than rejecting, so a
   * burst of unauthenticated DCR registrations cannot permanently exhaust it. */
  saveClient(client: OAuthClientInformationFull): Promise<void>;

  savePendingAuthorization(pending: PendingAuthorization): Promise<void>;
  consumePendingAuthorization(googleState: string): Promise<PendingAuthorization | undefined>;

  saveAuthorizationCode(code: AuthorizationCodeRecord): Promise<void>;
  peekAuthorizationCode(codeHash: string): Promise<AuthorizationCodeRecord | undefined>;
  consumeAuthorizationCode(codeHash: string): Promise<AuthorizationCodeRecord | undefined>;

  getUser(sub: string): Promise<TeamUserRecord | undefined>;
  upsertUser(user: TeamUserRecord): Promise<void>;
  /** Atomic read-modify-write of a single user, serialized against concurrent
   * writers. `mutator` receives the freshest stored record and returns the
   * replacement; a no-op when the user is absent. Prefer this over
   * getUser+upsertUser for partial updates — it avoids a stale snapshot
   * clobbering a concurrent write (e.g. a re-auth landing mid token-refresh). */
  updateUser(
    sub: string,
    mutator: (current: TeamUserRecord) => TeamUserRecord,
  ): Promise<void>;

  saveAccessToken(token: AccessTokenRecord): Promise<void>;
  getAccessToken(tokenHash: string): Promise<AccessTokenRecord | undefined>;
  deleteAccessToken(tokenHash: string): Promise<void>;

  saveRefreshToken(token: RefreshTokenRecord): Promise<void>;
  getRefreshToken(tokenHash: string): Promise<RefreshTokenRecord | undefined>;
  updateRefreshToken(token: RefreshTokenRecord): Promise<void>;

  /** invalid_grant recovery: drop every MCP token for a user so the next
   * request 401s and the client re-runs the OAuth flow. */
  revokeTokensForSub(sub: string): Promise<void>;
  /** Refresh-reuse theft response: drop all tokens in a rotation family. */
  revokeTokenFamily(familyId: string): Promise<void>;

  /** Drop expired pending authorizations, codes, and tokens. */
  sweepExpired(now?: number): Promise<void>;
}
