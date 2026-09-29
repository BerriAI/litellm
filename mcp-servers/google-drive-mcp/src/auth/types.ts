// ---------------------------------------------------------------------------
// Shared types for multi-account authentication.
//
// These are consumed by AccountStore, SessionStore, AccountResolver, and the
// AccountClientFactory. Keep dependencies minimal — this file must be cheap
// to import from anywhere in the auth module.
// ---------------------------------------------------------------------------

/** One user's OAuth credential record. Primary key is `sub` (Google-stable). */
export interface AccountRecord {
  alias: string;
  email: string;
  sub: string;
  accessToken: string;
  refreshToken: string;
  /** Space-separated scope string, verbatim from Google's token response. */
  scope: string;
  tokenType: 'Bearer';
  /** Access-token expiry, ms since epoch. */
  expiryDate: number;
  addedAt: string;
  lastRefreshedAt: string;
  /**
   * True when email/sub were not discoverable at account-add time (e.g. migrated
   * from a v1 token file with no userinfo scope). Cleared on next successful
   * identity resolution.
   */
  pendingIdentity?: boolean;
}

export interface TokenFileV2 {
  version: 2;
  /** Alias of the global default account. Optional — sole-authenticated wins if absent. */
  defaultAccount?: string;
  accounts: Record<string, AccountRecord>;
}

/** Tool operation classification. Drives resolution policy. */
export type ToolOpKind = 'read' | 'write' | 'admin';

/** The resolved targeting for a single tool invocation. */
export interface AccountTargeting {
  kind: 'single' | 'fanout';
  accounts: AccountRecord[];
  resolutionReason:
    | 'explicit-param'
    | 'session-default'
    | 'global-default'
    | 'sole-authenticated'
    | 'merged-eligible';
}

export interface SessionState {
  sessionId: string;
  createdAt: number;
  defaultAccountAlias?: string;
}

/** Auth mode detected at server startup. */
export type AuthMode =
  | 'local-oauth'       // multi-account: disk-backed tokens.json v2
  | 'service-account'   // single synthetic account, no disk persistence
  | 'external-token'    // single synthetic account, no disk persistence
  | 'test';             // unit/integration tests

/** Snapshot suitable for the `manage_accounts list` tool — no secrets. */
export interface RedactedAccountView {
  alias: string;
  email: string;
  sub: string;
  addedAt: string;
  scopesGranted: string[];
  expiresInSec: number | null;
  pendingIdentity: boolean;
  isDefault: boolean;
}

/** Alias validation: lowercase start, alphanumerics + `_`/`-`, 1–32 chars. */
export const ALIAS_PATTERN = /^[a-z0-9][a-z0-9_-]{0,31}$/;

/**
 * Reserved aliases that cannot be chosen via `manage_accounts add`.
 *
 * - `stdio`, `service-account`, `external-token`, `test` are used internally
 *   as session ids or synthetic-account names.
 * - `default` is the auto-assigned alias for the sole account produced by the
 *   v1→v2 migration and by the first-time boot flow; reserving it prevents
 *   users colliding with those installs and also discourages a poor label.
 * - `all` and `*` are held for a future fanout-over-everything shorthand
 *   (Phase 3+), so existing accounts can never shadow the keyword.
 */
export const RESERVED_ALIASES = new Set([
  'stdio',
  'service-account',
  'external-token',
  'test',
  'default',
  'all',
  '*',
]);
