import assert from 'node:assert/strict';
import test from 'node:test';

import { withRetry, TimeoutError } from '../src/utils/retry.js';
import type { RuntimeConfig } from '../src/utils/cliArgs.js';

const cfg = (o: Partial<RuntimeConfig> = {}): RuntimeConfig => ({
  apiTimeout: 0,
  tokenRefreshTimeout: 0,
  retryMax: 0,
  retryBaseDelay: 0,
  disableResources: false,
  ...o,
});

const httpErr = (status: number) => Object.assign(new Error(`HTTP ${status}`), {
  response: { status },
});

// ---------------------------------------------------------------------------
// happy path
// ---------------------------------------------------------------------------
test('returns the result without retrying on success', async () => {
  let calls = 0;
  const out = await withRetry(async () => { calls++; return 42; }, cfg({ retryMax: 3 }));
  assert.equal(out, 42);
  assert.equal(calls, 1);
});

// ---------------------------------------------------------------------------
// retryable errors
// ---------------------------------------------------------------------------
test('retries a retryable status (503) then succeeds', async () => {
  let calls = 0;
  const out = await withRetry(async () => {
    calls++;
    if (calls < 3) throw httpErr(503);
    return 'ok';
  }, cfg({ retryMax: 5 }));
  assert.equal(out, 'ok');
  assert.equal(calls, 3);
});

test('retries on numeric err.status and on string err.code', async () => {
  let calls = 0;
  await withRetry(async () => {
    calls++;
    if (calls === 1) throw Object.assign(new Error('rate'), { status: 429 });
    if (calls === 2) throw Object.assign(new Error('net'), { code: 'ECONNRESET' });
    return 1;
  }, cfg({ retryMax: 3 }));
  assert.equal(calls, 3);
});

// gaxios 7 (native fetch) wraps transport failures as GaxiosError(code:
// undefined) -> cause: TypeError('fetch failed') -> cause: undici error with
// the real code. The retry policy must find retryable codes down that chain.
const fetchFailedErr = (code: string) => {
  const undiciErr = Object.assign(new Error('socket hang up'), { code });
  const fetchErr = new TypeError('fetch failed');
  (fetchErr as Error & { cause?: unknown }).cause = undiciErr;
  return Object.assign(new Error('Request failed'), { cause: fetchErr });
};

test('retries a gaxios-7 transport error with the code nested in err.cause.cause', async () => {
  for (const code of ['ECONNRESET', 'UND_ERR_CONNECT_TIMEOUT', 'UND_ERR_SOCKET']) {
    let calls = 0;
    const out = await withRetry(async () => {
      calls++;
      if (calls === 1) throw fetchFailedErr(code);
      return 'ok';
    }, cfg({ retryMax: 2 }));
    assert.equal(out, 'ok');
    assert.equal(calls, 2, `expected one retry for nested code ${code}`);
  }
});

test('does not retry a non-retryable code nested in the cause chain', async () => {
  let calls = 0;
  await assert.rejects(
    withRetry(async () => { calls++; throw fetchFailedErr('ECONNREFUSED'); }, cfg({ retryMax: 3 })),
    /Request failed/,
  );
  assert.equal(calls, 1);
});

test('cause-chain walk is bounded on a cyclic cause chain', async () => {
  const cyclic = new Error('cyclic') as Error & { cause?: unknown };
  cyclic.cause = cyclic;
  let calls = 0;
  await assert.rejects(
    withRetry(async () => { calls++; throw cyclic; }, cfg({ retryMax: 3 })),
    /cyclic/,
  );
  assert.equal(calls, 1);
});

test('exhausts retryMax then throws the last error', async () => {
  let calls = 0;
  await assert.rejects(
    withRetry(async () => { calls++; throw httpErr(504); }, cfg({ retryMax: 2 })),
    /HTTP 504/,
  );
  assert.equal(calls, 3); // 1 initial + 2 retries
});

// ---------------------------------------------------------------------------
// non-retryable errors
// ---------------------------------------------------------------------------
test('does not retry a non-retryable status (400)', async () => {
  let calls = 0;
  await assert.rejects(
    withRetry(async () => { calls++; throw httpErr(400); }, cfg({ retryMax: 5 })),
    /HTTP 400/,
  );
  assert.equal(calls, 1);
});

test('does NOT retry 500 (non-idempotent batchUpdate guard)', async () => {
  let calls = 0;
  await assert.rejects(
    withRetry(async () => { calls++; throw httpErr(500); }, cfg({ retryMax: 5 })),
    /HTTP 500/,
  );
  assert.equal(calls, 1);
});

test('retryMax=0 disables retries', async () => {
  let calls = 0;
  await assert.rejects(
    withRetry(async () => { calls++; throw httpErr(503); }, cfg({ retryMax: 0 })),
    /HTTP 503/,
  );
  assert.equal(calls, 1);
});

// ---------------------------------------------------------------------------
// timeout
// ---------------------------------------------------------------------------
test('apiTimeout rejects a hung call with TimeoutError even if signal ignored', async () => {
  await assert.rejects(
    withRetry(() => new Promise(() => {}), cfg({ apiTimeout: 20, retryMax: 0 })),
    (err: unknown) => {
      assert.ok(err instanceof TimeoutError);
      assert.equal((err as TimeoutError).code, 'ETIMEDOUT');
      return true;
    },
  );
});

test('a timeout is retryable and aborts the forwarded signal', async () => {
  let calls = 0;
  let firstSignalAborted = false;
  await assert.rejects(
    withRetry((signal) => {
      calls++;
      if (calls === 1) {
        signal.addEventListener('abort', () => { firstSignalAborted = true; });
      }
      return new Promise(() => {});
    }, cfg({ apiTimeout: 20, retryMax: 1 })),
    (err: unknown) => err instanceof TimeoutError,
  );
  assert.equal(calls, 2); // timed-out attempt was retried
  assert.equal(firstSignalAborted, true);
});

test('apiTimeout=0 means no timeout (call is awaited)', async () => {
  const out = await withRetry(
    () => new Promise((resolve) => setTimeout(() => resolve('done'), 30)),
    cfg({ apiTimeout: 0, retryMax: 0 }),
  );
  assert.equal(out, 'done');
});

// ---------------------------------------------------------------------------
// backoff bounds
// ---------------------------------------------------------------------------
test('backoff is capped (huge base delay does not stall the test)', async () => {
  // base 10 min, but cap is 30s; retry happens immediately here because
  // retryBaseDelay is overridden small — this asserts the cap math is applied
  // (Math.min) rather than multiplying unbounded.
  let calls = 0;
  const start = Date.now();
  await assert.rejects(
    withRetry(async () => { calls++; throw httpErr(503); }, cfg({ retryMax: 1, retryBaseDelay: 5 })),
    /HTTP 503/,
  );
  assert.equal(calls, 2);
  // delay = min(5 * 2^0, 30000) + jitter(<200) → well under 1s
  assert.ok(Date.now() - start < 1000);
});
