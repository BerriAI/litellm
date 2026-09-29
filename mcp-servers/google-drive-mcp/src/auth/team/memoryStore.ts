// ---------------------------------------------------------------------------
// In-memory TeamStore. Also the base class for FileTeamStore: every durable
// mutation is routed through `mutateDurable`, which the file store overrides
// to serialize through its write queue. Pending authorizations and
// authorization codes are in-memory in BOTH implementations — their single-use
// `consume*` operations are a synchronous Map get+delete, which is what makes
// them atomic (no await between check and removal).
// ---------------------------------------------------------------------------

import type { OAuthClientInformationFull } from '@modelcontextprotocol/sdk/shared/auth.js';
import {
  AccessTokenRecord,
  AuthorizationCodeRecord,
  DEFAULT_TEAM_STORE_CAPS,
  MAX_CHALLENGE_LOOKUPS,
  PendingAuthorization,
  RefreshTokenRecord,
  TeamStore,
  TeamStoreCapacityError,
  TeamStoreCaps,
  TeamUserRecord,
} from './types.js';

export class InMemoryTeamStore implements TeamStore {
  protected clients = new Map<string, OAuthClientInformationFull>();
  /** Recency rank per client — a monotonic tick (seeded from ms epoch, but
   * strictly increasing so two events in the same millisecond still order).
   * Drives LRU eviction at the client cap so idle registrations are reclaimed
   * before active connectors. Persisted alongside clients by the file store. */
  protected clientLastUsed = new Map<string, number>();
  /** Last recency tick handed out; see nextClientTick. */
  private lastClientTick = 0;
  protected users = new Map<string, TeamUserRecord>();
  protected accessTokens = new Map<string, AccessTokenRecord>();
  protected refreshTokens = new Map<string, RefreshTokenRecord>();
  protected readonly caps: TeamStoreCaps;

  private pendingAuthorizations = new Map<string, PendingAuthorization>();
  private authorizationCodes = new Map<string, AuthorizationCodeRecord>();

  constructor(opts: { caps?: Partial<TeamStoreCaps> } = {}) {
    this.caps = { ...DEFAULT_TEAM_STORE_CAPS, ...opts.caps };
  }

  async init(): Promise<void> {
    // Nothing to load.
  }

  async flush(): Promise<void> {
    // In-memory: no durable writes are queued.
  }

  /**
   * Run a mutation of the durable sections (clients/users/tokens). The file
   * store overrides this to wrap the mutation in its serialized
   * merge-from-disk → mutate → atomic-write cycle; here it just runs.
   */
  protected async mutateDurable<T>(mutate: () => T): Promise<T> {
    return mutate();
  }

  // -------------------------------------------------------------------------
  // Registered clients (DCR)
  // -------------------------------------------------------------------------

  async getClient(clientId: string): Promise<OAuthClientInformationFull | undefined> {
    return this.clients.get(clientId);
  }

  async saveClient(client: OAuthClientInformationFull): Promise<void> {
    await this.mutateDurable(() => {
      // At the cap, reclaim the least-recently-used client rather than
      // rejecting: DCR is unauthenticated, so throwing here would let a flood
      // of never-used registrations permanently lock out real connectors. An
      // idle attacker registration is the LRU victim; an active connector
      // (which issues tokens) keeps its lastUsed fresh and survives.
      if (!this.clients.has(client.client_id) && this.clients.size >= this.caps.maxClients) {
        this.evictLruClient();
      }
      this.clients.set(client.client_id, client);
      this.clientLastUsed.set(client.client_id, this.nextClientTick());
    });
  }

  // -------------------------------------------------------------------------
  // Pending authorizations (in-memory only)
  // -------------------------------------------------------------------------

  async savePendingAuthorization(pending: PendingAuthorization): Promise<void> {
    if (this.pendingAuthorizations.size >= this.caps.maxPendingAuthorizations) {
      // Expired entries don't count against a live flow; drop them first.
      this.sweepPendingAndCodes(Date.now());
    }
    if (this.pendingAuthorizations.size >= this.caps.maxPendingAuthorizations) {
      throw new TeamStoreCapacityError(
        'pending authorizations',
        this.caps.maxPendingAuthorizations,
      );
    }
    this.pendingAuthorizations.set(pending.googleState, pending);
  }

  async consumePendingAuthorization(googleState: string): Promise<PendingAuthorization | undefined> {
    const pending = this.pendingAuthorizations.get(googleState);
    if (!pending) return undefined;
    this.pendingAuthorizations.delete(googleState);
    return pending.expiresAt > Date.now() ? pending : undefined;
  }

  // -------------------------------------------------------------------------
  // Authorization codes (in-memory only)
  // -------------------------------------------------------------------------

  async saveAuthorizationCode(code: AuthorizationCodeRecord): Promise<void> {
    this.authorizationCodes.set(code.codeHash, code);
  }

  async peekAuthorizationCode(codeHash: string): Promise<AuthorizationCodeRecord | undefined> {
    const code = this.authorizationCodes.get(codeHash);
    if (!code) return undefined;
    if (code.expiresAt <= Date.now()) {
      this.authorizationCodes.delete(codeHash);
      return undefined;
    }
    code.challengeLookups += 1;
    if (code.challengeLookups > MAX_CHALLENGE_LOOKUPS) {
      this.authorizationCodes.delete(codeHash);
      return undefined;
    }
    return code;
  }

  async consumeAuthorizationCode(codeHash: string): Promise<AuthorizationCodeRecord | undefined> {
    const code = this.authorizationCodes.get(codeHash);
    if (!code) return undefined;
    this.authorizationCodes.delete(codeHash);
    return code.expiresAt > Date.now() ? code : undefined;
  }

  // -------------------------------------------------------------------------
  // Users
  // -------------------------------------------------------------------------

  async getUser(sub: string): Promise<TeamUserRecord | undefined> {
    return this.users.get(sub);
  }

  async upsertUser(user: TeamUserRecord): Promise<void> {
    await this.mutateDurable(() => {
      if (!this.users.has(user.sub) && this.users.size >= this.caps.maxUsers) {
        throw new TeamStoreCapacityError('team users', this.caps.maxUsers);
      }
      this.users.set(user.sub, user);
    });
  }

  async updateUser(
    sub: string,
    mutator: (current: TeamUserRecord) => TeamUserRecord,
  ): Promise<void> {
    await this.mutateDurable(() => {
      // Read the freshest record INSIDE the serialized mutation (the file store
      // merges from disk first), so a concurrent writer's update to the same
      // user is not clobbered by a stale caller snapshot.
      const current = this.users.get(sub);
      if (!current) return;
      this.users.set(sub, mutator(current));
    });
  }

  // -------------------------------------------------------------------------
  // MCP tokens
  // -------------------------------------------------------------------------

  async saveAccessToken(token: AccessTokenRecord): Promise<void> {
    await this.mutateDurable(() => {
      this.evictOldestIfAtCap(this.accessTokens, token.sub, (t) => t.expiresAt);
      this.touchClient(token.clientId);
      this.accessTokens.set(token.tokenHash, token);
    });
  }

  async getAccessToken(tokenHash: string): Promise<AccessTokenRecord | undefined> {
    return this.accessTokens.get(tokenHash);
  }

  async deleteAccessToken(tokenHash: string): Promise<void> {
    await this.mutateDurable(() => {
      this.accessTokens.delete(tokenHash);
    });
  }

  async saveRefreshToken(token: RefreshTokenRecord): Promise<void> {
    await this.mutateDurable(() => {
      this.evictOldestIfAtCap(
        this.refreshTokens,
        token.sub,
        (t) => t.createdAt,
        (t) => t.supersededByHash === undefined,
      );
      // Independent backstop on tombstones: TOMBSTONE_TTL_MS bounds their
      // lifetime, but a burst of rotations between sweeps could still pile up,
      // so hard-cap the retained tombstones per user too.
      this.evictOldestIfAtCap(
        this.refreshTokens,
        token.sub,
        (t) => t.createdAt,
        (t) => t.supersededByHash !== undefined,
        this.caps.maxTombstonesPerSub,
      );
      this.touchClient(token.clientId);
      this.refreshTokens.set(token.tokenHash, token);
    });
  }

  async getRefreshToken(tokenHash: string): Promise<RefreshTokenRecord | undefined> {
    return this.refreshTokens.get(tokenHash);
  }

  async updateRefreshToken(token: RefreshTokenRecord): Promise<void> {
    await this.mutateDurable(() => {
      this.refreshTokens.set(token.tokenHash, token);
    });
  }

  async revokeTokensForSub(sub: string): Promise<void> {
    await this.mutateDurable(() => {
      for (const [hash, token] of this.accessTokens) {
        if (token.sub === sub) this.accessTokens.delete(hash);
      }
      for (const [hash, token] of this.refreshTokens) {
        if (token.sub === sub) this.refreshTokens.delete(hash);
      }
    });
  }

  async revokeTokenFamily(familyId: string): Promise<void> {
    await this.mutateDurable(() => {
      for (const [hash, token] of this.accessTokens) {
        if (token.familyId === familyId) this.accessTokens.delete(hash);
      }
      for (const [hash, token] of this.refreshTokens) {
        if (token.familyId === familyId) this.refreshTokens.delete(hash);
      }
    });
  }

  async sweepExpired(now: number = Date.now()): Promise<void> {
    this.sweepPendingAndCodes(now);
    await this.mutateDurable(() => {
      for (const [hash, token] of this.accessTokens) {
        if (token.expiresAt <= now) this.accessTokens.delete(hash);
      }
      // Superseded refresh tokens are kept as reuse-detection tombstones until
      // expiresAt — bounded to TOMBSTONE_TTL_MS past rotation (see provider), so
      // post-grace reuse within the detection horizon still revokes the family
      // while the store stays bounded.
      for (const [hash, token] of this.refreshTokens) {
        if (token.expiresAt <= now) this.refreshTokens.delete(hash);
      }
    });
  }

  // -------------------------------------------------------------------------
  // Internals
  // -------------------------------------------------------------------------

  private sweepPendingAndCodes(now: number): void {
    for (const [state, pending] of this.pendingAuthorizations) {
      if (pending.expiresAt <= now) this.pendingAuthorizations.delete(state);
    }
    for (const [hash, code] of this.authorizationCodes) {
      if (code.expiresAt <= now) this.authorizationCodes.delete(hash);
    }
  }

  /** Evict the caller's oldest token when their per-sub cap is reached, so one
   * reconnect-looping client cannot grow the store without bound. */
  private evictOldestIfAtCap<T extends { sub: string }>(
    map: Map<string, T>,
    sub: string,
    ageOf: (token: T) => number,
    counts: (token: T) => boolean = () => true,
    cap: number = this.caps.maxTokensPerSub,
  ): void {
    let count = 0;
    let oldestHash: string | undefined;
    let oldestAge = Infinity;
    for (const [hash, token] of map) {
      if (token.sub !== sub || !counts(token)) continue;
      count += 1;
      const age = ageOf(token);
      if (age < oldestAge) {
        oldestAge = age;
        oldestHash = hash;
      }
    }
    if (count >= cap && oldestHash !== undefined) {
      map.delete(oldestHash);
    }
  }

  /** Evict the least-recently-used client (registered or issued a token least
   * recently). A client missing a lastUsed entry is treated as the oldest. */
  private evictLruClient(): void {
    let oldestId: string | undefined;
    let oldestSeen = Infinity;
    for (const id of this.clients.keys()) {
      const seen = this.clientLastUsed.get(id) ?? 0;
      if (seen < oldestSeen) {
        oldestSeen = seen;
        oldestId = id;
      }
    }
    if (oldestId !== undefined) {
      this.clients.delete(oldestId);
      this.clientLastUsed.delete(oldestId);
    }
  }

  /** Mark a client active (bumps its LRU position). No-op for unknown clients
   * so token records for an already-evicted client don't resurrect its meta. */
  private touchClient(clientId: string): void {
    if (this.clients.has(clientId)) {
      this.clientLastUsed.set(clientId, this.nextClientTick());
    }
  }

  /** Next recency tick: wall-clock, but forced strictly above the previous tick
   * so events within one millisecond still order deterministically. */
  private nextClientTick(): number {
    this.lastClientTick = Math.max(Date.now(), this.lastClientTick + 1);
    return this.lastClientTick;
  }

  /** After loading persisted client recency ticks, advance the counter past the
   * highest so new ticks always outrank restored ones. */
  protected seedClientClock(): void {
    for (const v of this.clientLastUsed.values()) {
      if (v > this.lastClientTick) this.lastClientTick = v;
    }
  }
}
