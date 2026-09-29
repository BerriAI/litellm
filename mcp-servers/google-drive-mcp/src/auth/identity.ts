// ---------------------------------------------------------------------------
// Live identity verification
// ---------------------------------------------------------------------------
//
// Nothing else in the server ever asks Google "who am I?". Without that, an
// auth-mode/config mismatch (e.g. GOOGLE_APPLICATION_CREDENTIALS silently
// forcing an empty service account over the user's tokens.json — see issue
// #137) is invisible: every Drive call returns empty with no error. This
// helper resolves the identity the live Drive client is actually acting as.

import type { drive_v3 } from 'googleapis';

/** The account the live Drive client is actually authenticated as. */
export interface EffectiveIdentity {
  emailAddress: string | null;
  displayName: string | null;
  /** Bytes of Drive storage used, as reported by the API (string per Drive API). */
  storageUsage: string | null;
  /** Storage limit in bytes, or null for unlimited / pooled-quota accounts. */
  storageLimit: string | null;
  /** Present only when the identity lookup failed; the message is itself diagnostic. */
  error?: string;
}

/**
 * How long to wait for `about.get` before giving up. A diagnostic must return
 * promptly even when Google is unreachable, so this is deliberately far shorter
 * than the 120s default API timeout — a hung identity lookup would otherwise
 * hang the very tool you reach for when the network is misbehaving.
 */
const DEFAULT_IDENTITY_TIMEOUT_MS = 10_000;

/** The identity shape returned when the lookup can't produce a real answer. */
function errorIdentity(error: string): EffectiveIdentity {
  return {
    emailAddress: null,
    displayName: null,
    storageUsage: null,
    storageLimit: null,
    error,
  };
}

/**
 * Ask the Drive API which account the current client is acting as, via
 * `about.get`. Requires only a `drive`/`drive.readonly` scope, both of which
 * are in DEFAULT_SCOPES. Never throws — a failure (including a missing client
 * or a timeout) is returned as `{ error }` so callers (e.g. authGetStatus) can
 * surface it without crashing. `drive` may be undefined when the caller has no
 * usable client (no account, or a client that failed to build).
 */
export async function getEffectiveIdentity(
  drive: drive_v3.Drive | undefined,
  timeoutMs: number = DEFAULT_IDENTITY_TIMEOUT_MS,
): Promise<EffectiveIdentity> {
  if (!drive) {
    return errorIdentity(
      "No authenticated Drive client (no account, or the default account's client could not be built).",
    );
  }
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await drive.about.get(
      { fields: 'user(displayName,emailAddress),storageQuota(limit,usage)' },
      { signal: controller.signal },
    );
    const user = res.data.user;
    const quota = res.data.storageQuota;
    return {
      emailAddress: user?.emailAddress ?? null,
      displayName: user?.displayName ?? null,
      storageUsage: quota?.usage ?? null,
      storageLimit: quota?.limit ?? null,
    };
  } catch (e: unknown) {
    if (controller.signal.aborted) {
      return errorIdentity(`Identity lookup timed out after ${timeoutMs}ms`);
    }
    return errorIdentity(e instanceof Error ? e.message : String(e));
  } finally {
    clearTimeout(timer);
  }
}
