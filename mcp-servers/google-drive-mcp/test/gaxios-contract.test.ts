import assert from 'node:assert/strict';
import test from 'node:test';
import http from 'node:http';
import { Readable } from 'node:stream';
import { mkdtempSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { GoogleAuth, OAuth2Client } from 'google-auth-library';

// ---------------------------------------------------------------------------
// Contract tests for the HTTP client underneath googleapis / google-auth-library.
//
// Our code duck-types gaxios responses and errors (`res.data`, `res.headers`,
// `err.status`, `err.code`) rather than importing its types, so `tsc` cannot
// tell us when a dependency bump changes those shapes — the upgrade to gaxios 7
// silently broke `getGoogleDocImage` exactly that way, and unit tests missed it
// because their doubles were built in the *old* shape.
//
// These tests pin the real library's behavior against a local server (no
// network, no credentials). When a future bump changes a shape, the failure
// here names the assumption and the code that depends on it.
// ---------------------------------------------------------------------------

/** A local server standing in for a Google endpoint. */
async function withServer(
  handler: (req: http.IncomingMessage, res: http.ServerResponse) => void,
  run: (baseUrl: string) => Promise<void>,
): Promise<void> {
  const server = http.createServer(handler);
  await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
  const { port } = server.address() as { port: number };
  try {
    await run(`http://127.0.0.1:${port}`);
  } finally {
    await new Promise<void>((resolve) => server.close(() => resolve()));
  }
}

/** An OAuth2Client is what tool code calls `.request()` on (`ctx.authClient`). */
function client(): OAuth2Client {
  const c = new OAuth2Client();
  c.setCredentials({ access_token: 'test-token-not-a-secret' });
  return c;
}

// ---------------------------------------------------------------------------
// Response headers
// ---------------------------------------------------------------------------
test('response.headers is a Headers instance, so bracket access does NOT work', async () => {
  await withServer(
    (_req, res) => { res.writeHead(200, { 'content-type': 'image/png' }); res.end('bytes'); },
    async (base) => {
      const res = await client().request({ url: `${base}/img` });

      // The contract src/utils/streams.ts#getResponseHeader exists to satisfy.
      assert.equal(
        typeof (res.headers as unknown as Headers).get, 'function',
        'headers must expose the Headers API; getResponseHeader() relies on .get()',
      );
      assert.equal(
        (res.headers as unknown as Record<string, unknown>)['content-type'], undefined,
        'bracket access must stay unsupported — if this starts working, gaxios reverted to '
        + 'plain-object headers and getResponseHeader() should be re-checked',
      );
      assert.equal((res.headers as unknown as Headers).get('content-type'), 'image/png');
    },
  );
});

// ---------------------------------------------------------------------------
// Stream responses
// ---------------------------------------------------------------------------
test('responseType "stream" yields a Node Readable (gaxios defaults to node-fetch v3, not native fetch)', async () => {
  await withServer(
    (_req, res) => { res.writeHead(200, { 'content-type': 'text/plain' }); res.end('payload'); },
    async (base) => {
      const res = await client().request({ url: `${base}/file`, responseType: 'stream' });

      // gaxios returns `Response.body` verbatim. Its default fetch in Node is
      // node-fetch v3, whose body is a Node stream — so `.on()` handlers work.
      // Under *native* fetch this would be a web ReadableStream instead, which
      // is why src/utils/streams.ts#toNodeReadable normalizes both. If this
      // assertion ever fails, that helper stopped being merely defensive and is
      // now load-bearing for every `responseType: 'stream'` caller.
      assert.ok(
        res.data instanceof Readable,
        'expected a Node Readable; a web ReadableStream here means the fetch '
        + 'implementation changed — audit every responseType:"stream" call site',
      );
      assert.equal(typeof (res.data as Readable).on, 'function');
    },
  );
});

// ---------------------------------------------------------------------------
// Error shapes — what isInvalidGrant() and retry.core.js read
// ---------------------------------------------------------------------------
test('an HTTP error exposes numeric status and a parsed JSON body', async () => {
  await withServer(
    (_req, res) => {
      res.writeHead(400, { 'content-type': 'application/json' });
      res.end(JSON.stringify({ error: 'invalid_grant', error_description: 'Token has been revoked.' }));
    },
    async (base) => {
      await assert.rejects(
        () => client().request({ url: `${base}/token`, method: 'POST' }),
        (err: any) => {
          // src/auth/accountClientFactory.ts#isInvalidGrant reads response.data.error
          assert.equal(err.response?.data?.error, 'invalid_grant',
            'isInvalidGrant() depends on the JSON body being parsed onto response.data');
          // src/utils/retry.core.js#httpStatus reads response.status ?? status
          assert.equal(err.status, 400, 'retry.core.js httpStatus() depends on a numeric top-level status');
          assert.equal(err.response?.status, 400);
          return true;
        },
      );
    },
  );
});

test('a transport failure exposes a string code (top level and in the cause chain)', async () => {
  // Port 1 has nothing listening, so this fails before any HTTP exchange.
  await assert.rejects(
    () => client().request({ url: 'http://127.0.0.1:1/unreachable' }),
    (err: any) => {
      // retry.core.js#isRetryable walks the chain; today the code is already at
      // the top, but the walk also covers wrappers that only set it on a cause.
      const codes: unknown[] = [];
      for (let e = err, depth = 0; e && depth < 5; e = e.cause, depth++) codes.push(e.code);
      assert.ok(
        codes.some((c) => typeof c === 'string' && c.length > 0),
        `expected a string error code somewhere in the cause chain, got ${JSON.stringify(codes)}`,
      );
      assert.equal(err.status, undefined, 'a transport failure has no HTTP status');
      return true;
    },
  );
});

// ---------------------------------------------------------------------------
// Credentials loading
// ---------------------------------------------------------------------------
test('GoogleAuth does NOT reject a malformed key file — validateCredentialsFile() must', async () => {
  const dir = mkdtempSync(join(tmpdir(), 'gdrive-mcp-contract-'));
  const badKey = join(dir, 'bad.json');
  writeFileSync(badKey, '{ truncated json ');

  // google-auth-library v10 resolves a credential-less client here, where v9
  // threw. That silent success is why src/auth/externalAuth.ts validates the
  // file itself before constructing GoogleAuth. If this ever throws, upstream
  // fixed it and our guard becomes belt-and-braces (still fine to keep).
  const auth = new GoogleAuth({ keyFile: badKey, scopes: ['https://www.googleapis.com/auth/drive'] });
  const resolved = await auth.getClient().then(() => true, () => false);
  assert.equal(resolved, true,
    'upstream behavior changed: GoogleAuth now rejects malformed key files — '
    + 'revisit validateCredentialsFile() in src/auth/externalAuth.ts');
});
