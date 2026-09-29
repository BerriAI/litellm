// ---------------------------------------------------------------------------
// File-backed TeamStore: team-store.json, mode 0600.
//
// Persists the durable sections (registered clients, users, MCP token hashes)
// with the same write discipline as AccountStore: a serialized queue where
// each mutation re-reads the freshest on-disk state, applies, then writes via
// temp-file + atomic rename. Pending authorizations and authorization codes
// stay in-memory (see memoryStore.ts) — losing them on restart costs one
// login retry.
//
// The file holds Google refresh tokens in cleartext (they must be replayed to
// Google verbatim) — it is the deployment's most sensitive asset. MCP tokens
// are stored only as SHA-256 hashes.
// ---------------------------------------------------------------------------

import * as fs from 'fs/promises';
import * as path from 'path';
import type { OAuthClientInformationFull } from '@modelcontextprotocol/sdk/shared/auth.js';
import { describeErrorForLog } from '../utils.js';
import { InMemoryTeamStore } from './memoryStore.js';
import {
  AccessTokenRecord,
  RefreshTokenRecord,
  TeamStoreCaps,
  TeamUserRecord,
} from './types.js';

/** Message thrown when parsed JSON does not match the store shape; also the
 * signal isCorruptionError uses to classify the failure. */
const BAD_SHAPE_MESSAGE = 'unrecognized team-store file shape';

interface TeamStoreFileV1 {
  version: 1;
  clients: Record<string, OAuthClientInformationFull>;
  /** Per-client last-used timestamps (ms epoch) backing LRU eviction. Optional
   * for backward compatibility: pre-existing files without it load fine and the
   * next write adds it. */
  clientMeta?: Record<string, number>;
  users: Record<string, TeamUserRecord>;
  accessTokens: Record<string, AccessTokenRecord>;
  refreshTokens: Record<string, RefreshTokenRecord>;
}

function isENOENT(err: unknown): boolean {
  return (
    typeof err === 'object' && err !== null && (err as { code?: unknown }).code === 'ENOENT'
  );
}

/** True only for genuine CONTENT corruption (unparseable JSON or wrong shape),
 * which is safe to quarantine. IO/permission failures (EACCES, EIO, …) carry an
 * errno `code` and are deliberately NOT corruption — the file may be intact. */
function isCorruptionError(err: unknown): boolean {
  return (
    err instanceof SyntaxError ||
    (err instanceof Error && err.message === BAD_SHAPE_MESSAGE)
  );
}

function looksLikeV1(parsed: unknown): parsed is TeamStoreFileV1 {
  if (typeof parsed !== 'object' || parsed === null) return false;
  const p = parsed as Record<string, unknown>;
  return (
    p.version === 1 &&
    ['clients', 'users', 'accessTokens', 'refreshTokens'].every(
      (key) => typeof p[key] === 'object' && p[key] !== null && !Array.isArray(p[key]),
    )
  );
}

export class FileTeamStore extends InMemoryTeamStore {
  private writeQueue: Promise<unknown> = Promise.resolve();
  private tmpSeq = 0;

  constructor(
    private readonly filePath: string,
    opts: { caps?: Partial<TeamStoreCaps> } = {},
  ) {
    super(opts);
  }

  getFilePath(): string {
    return this.filePath;
  }

  override async init(): Promise<void> {
    try {
      const content = await fs.readFile(this.filePath, 'utf-8');
      const parsed = JSON.parse(content) as unknown;
      if (!looksLikeV1(parsed)) {
        throw new Error(BAD_SHAPE_MESSAGE);
      }
      this.loadSections(parsed);
    } catch (err) {
      if (isENOENT(err)) {
        // No file yet — normal first boot; fall through and create it.
      } else if (isCorruptionError(err)) {
        // The file's CONTENT is unusable (bad JSON / wrong shape). Move it aside
        // (non-destructive) and start empty — affected team members re-authorize.
        const backup = `${this.filePath}.corrupt-${Date.now()}`;
        const moved = await fs
          .rename(this.filePath, backup)
          .then(() => true)
          .catch(() => false);
        console.error(
          `TeamStore: ${this.filePath} was corrupt (${describeErrorForLog(err)}). ` +
            (moved ? `Moved it to ${backup}. ` : '') +
            'Starting fresh — team members will need to re-authorize.',
        );
      } else {
        // IO/permission error (EACCES, EIO, EMFILE, …): the file is likely
        // intact, just momentarily unreadable. Quarantining it here would
        // destroy every member's Google grant over a transient glitch, so fail
        // loudly and leave the file untouched.
        throw new Error(
          `Team store at ${this.filePath} could not be read (${describeErrorForLog(err)}). ` +
            'Refusing to start rather than risk discarding stored credentials — ' +
            'fix the file permissions/mount and restart.',
        );
      }
    }
    // Establish the file (0600) and prove the path is writable — a broken
    // store path must fail at startup, not on the first user's sign-in.
    await this.atomicWrite();
  }

  override async flush(): Promise<void> {
    // Drain the queue tail so writes enqueued before shutdown reach disk. The
    // queue swallows rejections, so this never throws.
    await this.writeQueue;
  }

  protected override mutateDurable<T>(mutate: () => T): Promise<T> {
    const next = this.writeQueue.then(async () => {
      // Read-modify-write: re-read the freshest on-disk state before applying
      // the mutation so a concurrent writer's records survive instead of being
      // clobbered by our stale in-memory snapshot.
      await this.mergeFromDisk();
      const result = mutate();
      await this.atomicWrite();
      return result;
    });
    // Swallow rejections on the chain so a failed write doesn't poison
    // subsequent writes.
    this.writeQueue = next.catch(() => undefined);
    return next;
  }

  // -------------------------------------------------------------------------
  // Internals
  // -------------------------------------------------------------------------

  private loadSections(data: TeamStoreFileV1): void {
    this.clients = new Map(Object.entries(data.clients));
    // Rebuild the LRU map for exactly the clients on disk. A client missing a
    // meta entry (older file, or hand-edited) is seeded to "now" so it is not
    // immediately treated as the eviction victim on first load.
    const meta = data.clientMeta ?? {};
    const now = Date.now();
    this.clientLastUsed = new Map(
      Object.keys(data.clients).map((id) => [
        id,
        typeof meta[id] === 'number' ? meta[id] : now,
      ]),
    );
    this.seedClientClock();
    this.users = new Map(Object.entries(data.users));
    this.accessTokens = new Map(Object.entries(data.accessTokens));
    this.refreshTokens = new Map(Object.entries(data.refreshTokens));
  }

  private serialize(): TeamStoreFileV1 {
    return {
      version: 1,
      clients: Object.fromEntries(this.clients),
      clientMeta: Object.fromEntries(this.clientLastUsed),
      users: Object.fromEntries(this.users),
      accessTokens: Object.fromEntries(this.accessTokens),
      refreshTokens: Object.fromEntries(this.refreshTokens),
    };
  }

  private async mergeFromDisk(): Promise<void> {
    try {
      const content = await fs.readFile(this.filePath, 'utf-8');
      const parsed = JSON.parse(content) as unknown;
      if (looksLikeV1(parsed)) {
        this.loadSections(parsed);
      }
      // Foreign-shaped content: keep the in-memory state — a queued write must
      // not be derailed by a transient bad read.
    } catch (err) {
      if (isENOENT(err)) {
        // File removed out from under us (operator reset): start empty.
        this.loadSections({ version: 1, clients: {}, users: {}, accessTokens: {}, refreshTokens: {} });
        return;
      }
      console.error(
        `TeamStore: could not re-read ${this.filePath} before write; keeping in-memory state. ` +
          `(${describeErrorForLog(err)})`,
      );
    }
  }

  private async atomicWrite(): Promise<void> {
    const dir = path.dirname(this.filePath);
    await fs.mkdir(dir, { recursive: true });
    // Unique temp name per writer so concurrent processes sharing this file
    // never race on the same scratch path (rename stays atomic).
    const tmp = `${this.filePath}.tmp.${process.pid}.${this.tmpSeq++}`;
    await fs.writeFile(tmp, JSON.stringify(this.serialize(), null, 2), { mode: 0o600 });
    await fs.rename(tmp, this.filePath);
  }
}
