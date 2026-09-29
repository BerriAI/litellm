// Shared retry/timeout core, kept as plain JS so it can be consumed two ways:
//  - by the bundled app, via the typed re-export in ./retry.ts, and
//  - directly by standalone Node scripts (e.g. scripts/registry-metadata.js) that
//    run without a build step or node_modules.
// This file is the single source of truth for the retry policy; JSDoc gives the
// app full types through ./retry.ts. It depends only on Node globals, so plain
// `node` can import it with zero dependencies.

// 429 rate-limit and 503/504 transient gateway errors are conventionally
// retryable. 500/502 are deliberately excluded: documents.batchUpdate is
// non-idempotent, and a 5xx after the request reached the server may mean it
// was partially applied — retrying could double-apply edits.
const RETRYABLE_STATUS = new Set([429, 503, 504]);
// UND_ERR_CONNECT_TIMEOUT and UND_ERR_SOCKET are undici's (native fetch)
// analogues of ETIMEDOUT-before-connect and ECONNRESET — same hazard profile.
const RETRYABLE_CODES = new Set([
  'ETIMEDOUT',
  'ECONNRESET',
  'ENOTFOUND',
  'EAI_AGAIN',
  'UND_ERR_CONNECT_TIMEOUT',
  'UND_ERR_SOCKET',
]);
const MAX_DELAY_MS = 30_000;

export class TimeoutError extends Error {
  code = 'ETIMEDOUT';
  /**
   * @param {number} ms
   * @param {string} op
   */
  constructor(ms, op) {
    super(`${op} timed out after ${ms}ms`);
    this.name = 'TimeoutError';
  }
}

function httpStatus(err) {
  const s = err?.response?.status ?? err?.status;
  return typeof s === 'number' ? s : undefined;
}

// gaxios 7 wraps native-fetch transport failures as GaxiosError(code:
// undefined) -> cause: TypeError('fetch failed') -> cause: the undici error
// carrying the real code, so the flat err.code check alone would miss every
// transport error. The walk is depth-bounded to survive cyclic cause chains.
function hasRetryableCode(err) {
  for (let e = err, depth = 0; e && depth < 5; e = e.cause, depth++) {
    if (typeof e.code === 'string' && RETRYABLE_CODES.has(e.code)) return true;
  }
  return false;
}

function isRetryable(err) {
  if (!err) return false;
  const status = httpStatus(err);
  if (status !== undefined && RETRYABLE_STATUS.has(status)) return true;
  return hasRetryableCode(err);
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function runAttempt(fn, controller, timeoutMs, opLabel) {
  const p = fn(controller.signal);
  if (timeoutMs <= 0) return await p;
  // The abort cancels cooperative callers (e.g. gaxios `{ signal }`) so a
  // timed-out request does not keep running; the race guarantees the attempt
  // still rejects even if the caller ignores the signal.
  p.catch(() => {});
  let timer;
  const timeout = new Promise((_, reject) => {
    timer = setTimeout(() => {
      // Reject before aborting so the race settles with TimeoutError rather
      // than the caller's (less informative) cancellation error.
      reject(new TimeoutError(timeoutMs, opLabel));
      controller.abort();
    }, timeoutMs);
  });
  try {
    return await Promise.race([p, timeout]);
  } finally {
    if (timer) clearTimeout(timer);
  }
}

/**
 * Runs `fn` with a per-attempt timeout and exponential backoff.
 *
 * `fn` receives an AbortSignal that callers SHOULD forward to the underlying
 * request (e.g. gaxios `{ signal }`) so a timed-out attempt is actually
 * cancelled rather than left running — important for non-idempotent calls
 * where a stray in-flight request racing a retry could double-apply.
 *
 * @template T
 * @param {(signal: AbortSignal) => Promise<T>} fn
 * @param {Pick<import('./cliArgs.js').RuntimeConfig, 'apiTimeout' | 'retryMax' | 'retryBaseDelay'>} cfg
 * @param {string} [opLabel]
 * @param {(message: string, data?: unknown) => void} [log]
 * @returns {Promise<T>}
 */
export async function withRetry(fn, cfg, opLabel = 'operation', log = () => {}) {
  let lastErr;
  for (let attempt = 0; attempt <= cfg.retryMax; attempt++) {
    const controller = new AbortController();
    try {
      return await runAttempt(fn, controller, cfg.apiTimeout, opLabel);
    } catch (err) {
      lastErr = err;
      if (!isRetryable(err) || attempt === cfg.retryMax) throw err;
      const backoff = Math.min(cfg.retryBaseDelay * 2 ** attempt, MAX_DELAY_MS);
      const delay = backoff + Math.floor(Math.random() * 200);
      log(`[${opLabel}] retry ${attempt + 1}/${cfg.retryMax} in ${delay}ms`, {
        reason: err?.message,
      });
      await sleep(delay);
    }
  }
  throw lastErr;
}
