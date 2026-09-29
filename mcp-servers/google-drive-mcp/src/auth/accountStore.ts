// ---------------------------------------------------------------------------
// AccountStore — disk-backed multi-account token persistence.
//
// - Owns `tokens.json` (v2 schema) with atomic-rename writes.
// - Serializes concurrent writes through an in-process queue.
// - Handles one-shot migration from v1 (single-account flat OAuth credentials)
//   to v2 on first load. A backup is left at `tokens.json.v1-backup-<ts>`.
// - Synthetic-mode accounts (service account / external token / test) live
//   in-memory only and are never written to disk.
// ---------------------------------------------------------------------------

import * as crypto from 'crypto';
import * as fs from 'fs/promises';
import * as path from 'path';
import {
  AccountRecord,
  AuthMode,
  RedactedAccountView,
  TokenFileV2,
} from './types.js';
import { DEFAULT_SCOPES, splitScopes } from './scopes.js';
import {
  describeErrorForLog,
  getAdditionalLegacyPaths,
  getLegacyTokenPath,
  getSecureTokenPath,
} from './utils.js';

function emptyFile(): TokenFileV2 {
  return { version: 2, accounts: {} };
}

export class AccountStore {
  private readonly filePath: string;
  private readonly mode: AuthMode;
  private data: TokenFileV2 = emptyFile();
  private syntheticClients = new Map<string, unknown>();
  private writeQueue: Promise<void> = Promise.resolve();
  private tmpSeq = 0;

  constructor(opts?: { filePath?: string; mode?: AuthMode }) {
    this.filePath = opts?.filePath ?? getSecureTokenPath();
    this.mode = opts?.mode ?? 'local-oauth';
  }

  getMode(): AuthMode {
    return this.mode;
  }

  getFilePath(): string {
    return this.filePath;
  }

  /**
   * Load the token file from disk (or migrate v1/legacy → v2 on first run).
   * Safe to call multiple times; re-reads the file each time.
   */
  async reload(): Promise<void> {
    if (this.mode !== 'local-oauth') {
      return;
    }

    try {
      const content = await fs.readFile(this.filePath, 'utf-8');
      const parsed = JSON.parse(content) as Record<string, unknown>;

      if (
        parsed &&
        parsed.version === 2 &&
        parsed.accounts &&
        typeof parsed.accounts === 'object' &&
        !Array.isArray(parsed.accounts)
      ) {
        this.data = parsed as unknown as TokenFileV2;
        return;
      }

      if (looksLikeV1(parsed)) {
        await this.migrateFromV1(parsed as V1TokenShape);
        return;
      }

      throw new Error(
        `Unrecognized tokens.json format (version=${parsed?.version ?? 'unknown'}). ` +
          `Expected a v2 file or a legacy single-account OAuth credential.`,
      );
    } catch (err: unknown) {
      if (isENOENT(err)) {
        // No current file — check legacy paths before giving up.
        const migrated = await this.tryMigrateLegacyPaths();
        if (migrated) {
          return;
        }
        this.data = emptyFile();
        return;
      }
      throw err;
    }
  }

  /**
   * Move an unreadable/corrupt token file aside (non-destructive) and reset to an
   * empty in-memory state, so the boot flow can self-heal instead of failing every
   * request. Returns the backup path, or null if there was nothing to move.
   */
  async quarantineCorruptFile(): Promise<string | null> {
    let moved: string | null = null;
    if (this.mode === 'local-oauth') {
      const backup = `${this.filePath}.corrupt-${Date.now()}`;
      try {
        await fs.rename(this.filePath, backup);
        moved = backup;
      } catch (err: unknown) {
        if (!isENOENT(err)) {
          console.error(`AccountStore: could not quarantine ${this.filePath}:`, err);
        }
      }
    }
    this.data = emptyFile();
    return moved;
  }

  list(): AccountRecord[] {
    return Object.values(this.data.accounts);
  }

  get(alias: string): AccountRecord | undefined {
    return this.data.accounts[alias];
  }

  getDefault(): string | undefined {
    return this.data.defaultAccount;
  }

  async upsert(record: AccountRecord): Promise<void> {
    return this.enqueue(() => {
      this.data.accounts[record.alias] = record;
    });
  }

  async remove(alias: string): Promise<void> {
    return this.enqueue(() => {
      delete this.data.accounts[alias];
      if (this.data.defaultAccount === alias) {
        delete this.data.defaultAccount;
      }
    });
  }

  async setDefault(alias: string | null): Promise<void> {
    return this.enqueue(() => {
      if (alias === null) {
        delete this.data.defaultAccount;
        return;
      }
      if (!this.data.accounts[alias]) {
        throw new Error(`Cannot set default: alias "${alias}" does not exist.`);
      }
      this.data.defaultAccount = alias;
    });
  }

  /**
   * Seed a synthetic (non-file-backed) account — used for service-account,
   * external-token, and test modes. Never written to disk.
   */
  setSyntheticAccount(record: AccountRecord, client: unknown): void {
    this.data.accounts[record.alias] = record;
    this.data.defaultAccount = record.alias;
    this.syntheticClients.set(record.alias, client);
  }

  getSyntheticClient(alias: string): unknown | undefined {
    return this.syntheticClients.get(alias);
  }

  listRedacted(): RedactedAccountView[] {
    const defaultAlias = this.data.defaultAccount;
    const now = Date.now();
    return this.list().map((r) => ({
      alias: r.alias,
      email: r.email,
      sub: r.sub,
      addedAt: r.addedAt,
      scopesGranted: splitScopes(r.scope),
      expiresInSec: r.expiryDate ? Math.floor((r.expiryDate - now) / 1000) : null,
      pendingIdentity: !!r.pendingIdentity,
      isDefault: r.alias === defaultAlias,
    }));
  }

  // -------------------------------------------------------------------------
  // Internal helpers
  // -------------------------------------------------------------------------

  private enqueue(mutate: () => void): Promise<void> {
    const next = this.writeQueue.then(async () => {
      // Read-modify-write: re-read the freshest on-disk state before applying the
      // mutation so a concurrent process's accounts (or a refresh token it just
      // rotated) survive instead of being clobbered by our stale in-memory snapshot.
      if (this.mode === 'local-oauth') {
        await this.mergeFromDisk();
      }
      mutate();
      if (this.mode === 'local-oauth') {
        await this.atomicWrite();
      }
    });
    // Swallow rejections on the chain so a failed write doesn't poison subsequent writes.
    this.writeQueue = next.catch(() => undefined);
    return next;
  }

  private async atomicWrite(): Promise<void> {
    const dir = path.dirname(this.filePath);
    await fs.mkdir(dir, { recursive: true });
    // Unique temp name per writer so concurrent processes sharing this file never
    // race on the same scratch path (rename stays atomic).
    const tmp = `${this.filePath}.tmp.${process.pid}.${this.tmpSeq++}`;
    await fs.writeFile(tmp, JSON.stringify(this.data, null, 2), { mode: 0o600 });
    await fs.rename(tmp, this.filePath);
  }

  /**
   * Re-read the on-disk file into memory ahead of a queued write. Best-effort:
   * a missing file resets us to empty; an unreadable or foreign-shaped file
   * leaves the in-memory state untouched (a queued write must not be derailed by
   * a transient bad read, and migration only ever happens in reload()).
   */
  private async mergeFromDisk(): Promise<void> {
    try {
      const content = await fs.readFile(this.filePath, 'utf-8');
      const parsed = JSON.parse(content) as Record<string, unknown>;
      if (
        parsed &&
        parsed.version === 2 &&
        parsed.accounts &&
        typeof parsed.accounts === 'object' &&
        !Array.isArray(parsed.accounts)
      ) {
        this.data = parsed as unknown as TokenFileV2;
      }
    } catch (err: unknown) {
      if (isENOENT(err)) {
        this.data = emptyFile();
        return;
      }
      // describeErrorForLog: a JSON.parse SyntaxError echoes fragments of the
      // token file into its message.
      console.error(
        `AccountStore: could not re-read ${this.filePath} before write; keeping in-memory state. ` +
          `(${describeErrorForLog(err)})`,
      );
    }
  }

  private async migrateFromV1(v1: V1TokenShape): Promise<void> {
    const record = buildRecordFromV1(v1);
    this.data = {
      version: 2,
      defaultAccount: record.alias,
      accounts: { [record.alias]: record },
    };
    const backup = `${this.filePath}.v1-backup-${Date.now()}`;
    try {
      await fs.rename(this.filePath, backup);
      console.error(`Migrated v1 tokens.json → v2 format. Backup saved to ${backup}`);
    } catch (err) {
      console.error('Warning: could not rename v1 token file for backup:', err);
    }
    await this.atomicWrite();
  }

  private async tryMigrateLegacyPaths(): Promise<boolean> {
    const legacyPaths = [getLegacyTokenPath(), ...getAdditionalLegacyPaths()];
    for (const legacyPath of legacyPaths) {
      try {
        const content = await fs.readFile(legacyPath, 'utf-8');
        const parsed = JSON.parse(content) as Record<string, unknown>;
        if (!looksLikeV1(parsed)) continue;
        const record = buildRecordFromV1(parsed as V1TokenShape);
        this.data = {
          version: 2,
          defaultAccount: record.alias,
          accounts: { [record.alias]: record },
        };
        await this.atomicWrite();
        try {
          await fs.unlink(legacyPath);
        } catch (_e) {
          /* best-effort */
        }
        console.error(`Migrated legacy token file ${legacyPath} → ${this.filePath}`);
        return true;
      } catch (err) {
        if (isENOENT(err)) continue;
        // Malformed file at a legacy path — log and continue, don't fail boot.
        console.error(
          `Skipping unreadable legacy token file ${legacyPath}: ${describeErrorForLog(err)}`,
        );
      }
    }
    return false;
  }
}

// ---------------------------------------------------------------------------
// v1 detection & conversion
// ---------------------------------------------------------------------------

// Field nullability matches google-auth-library's `Credentials`, so a fresh
// token response can be passed here directly (see buildRecordFromV1).
interface V1TokenShape {
  access_token?: string | null;
  refresh_token?: string | null;
  scope?: string | null;
  expiry_date?: number | null;
  token_type?: string | null;
}

function looksLikeV1(parsed: unknown): boolean {
  if (!parsed || typeof parsed !== 'object') return false;
  const p = parsed as Record<string, unknown>;
  if (p.version !== undefined) return false;
  return (
    typeof p.access_token === 'string' ||
    typeof p.refresh_token === 'string'
  );
}

/**
 * Convert a flat v1 token shape into a v2 AccountRecord under the reserved
 * alias 'default'. Used by the v1→v2 file migration AND by buildAuthSystem's
 * first-time bootstrap, which feeds the fresh OAuth token response through
 * here so both paths produce an identical record.
 */
export function buildRecordFromV1(v1: V1TokenShape): AccountRecord {
  const refreshToken = v1.refresh_token ?? '';
  const seed = refreshToken || v1.access_token || 'no-token';
  const sub = crypto.createHash('sha256').update(seed).digest('hex').slice(0, 24);
  const now = new Date().toISOString();
  return {
    alias: 'default',
    email: 'unknown',
    sub,
    accessToken: v1.access_token ?? '',
    refreshToken,
    // A v1 file predates scope filtering (the old single-account world used the
    // one token for everything). If it never recorded a scope, fall back to the
    // full default set so migration doesn't newly fail the resolver's scope gate
    // — a total outage from a file that worked the day before.
    scope: v1.scope ?? DEFAULT_SCOPES.join(' '),
    tokenType: 'Bearer',
    expiryDate: v1.expiry_date ?? 0,
    addedAt: now,
    lastRefreshedAt: now,
    pendingIdentity: true,
  };
}

function isENOENT(err: unknown): boolean {
  return (
    err instanceof Error &&
    'code' in err &&
    (err as NodeJS.ErrnoException).code === 'ENOENT'
  );
}
