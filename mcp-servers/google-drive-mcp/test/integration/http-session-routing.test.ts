import assert from 'node:assert/strict';
import { describe, it, before, after } from 'node:test';
import type { Server as HttpServer } from 'node:http';
import { google } from 'googleapis';
import { createAllMocks } from '../helpers/mock-google-apis.js';

// ---------------------------------------------------------------------------
// HTTP per-session account routing (finding 10).
//
// Each HTTP session must resolve per-session state under its real Mcp-Session-Id,
// not a shared 'stdio' key. We give two sessions distinct per-session default
// accounts and assert a bare tool call (no `account`) routes to each session's
// own default. Before the fix both sessions resolved under 'stdio' — the
// per-session default was never found, so both fell through to the global
// default and neither per-account marker appeared.
// ---------------------------------------------------------------------------

const MCP_HEADERS = {
  'Content-Type': 'application/json',
  Accept: 'application/json, text/event-stream',
};

let _serverModule: any = null;
async function getServerModule() {
  if (!_serverModule) _serverModule = await import('../../src/index.js');
  return _serverModule;
}

/** Parse an SSE or JSON response and return the first JSON-RPC message. */
async function parseResponse(res: Response): Promise<any> {
  const contentType = res.headers.get('content-type') || '';
  const text = await res.text();
  if (contentType.includes('text/event-stream')) {
    for (const line of text.split('\n')) {
      if (line.startsWith('data: ')) return JSON.parse(line.slice(6));
    }
    throw new Error('No data line found in SSE response');
  }
  return JSON.parse(text);
}

function startServer(app: any): Promise<{ httpServer: HttpServer; baseUrl: string }> {
  return new Promise((resolve) => {
    const httpServer = app.listen(0, '127.0.0.1', () => {
      const addr = httpServer.address();
      const baseUrl =
        addr && typeof addr === 'object' ? `http://127.0.0.1:${addr.port}` : '';
      resolve({ httpServer, baseUrl });
    });
  });
}

async function initSession(baseUrl: string): Promise<string> {
  const res = await fetch(`${baseUrl}/mcp`, {
    method: 'POST',
    headers: MCP_HEADERS,
    body: JSON.stringify({
      jsonrpc: '2.0',
      method: 'initialize',
      params: {
        protocolVersion: '2025-03-26',
        capabilities: {},
        clientInfo: { name: 'test', version: '1.0.0' },
      },
      id: 1,
    }),
  });
  assert.equal(res.status, 200);
  const sid = res.headers.get('mcp-session-id')!;
  await res.text();
  // Complete the handshake so subsequent tool calls are accepted.
  await fetch(`${baseUrl}/mcp`, {
    method: 'POST',
    headers: { ...MCP_HEADERS, 'mcp-session-id': sid },
    body: JSON.stringify({ jsonrpc: '2.0', method: 'notifications/initialized' }),
  });
  return sid;
}

async function callSearch(baseUrl: string, sid: string): Promise<any> {
  const res = await fetch(`${baseUrl}/mcp`, {
    method: 'POST',
    headers: { ...MCP_HEADERS, 'mcp-session-id': sid },
    body: JSON.stringify({
      jsonrpc: '2.0',
      method: 'tools/call',
      params: { name: 'search', arguments: { query: 'q' } },
      id: 2,
    }),
  });
  assert.equal(res.status, 200);
  return parseResponse(res);
}

describe('HTTP per-session account routing', () => {
  let httpServer: HttpServer;
  let baseUrl: string;
  let sessions: Map<string, any>;
  let mod: any;
  let originalDriveFactory: typeof google.drive;
  const authCalls: Array<{ marker: string | undefined }> = [];

  before(async () => {
    const mocks = createAllMocks();
    (google as any).drive = mocks.google.drive;
    (google as any).docs = mocks.google.docs;
    (google as any).sheets = mocks.google.sheets;
    (google as any).slides = mocks.google.slides;
    (google as any).calendar = mocks.google.calendar;

    mod = await getServerModule();
    mod._setAuthClientForTesting({ request: async () => ({ data: 'mock' }) });
    // Two extra synthetic accounts on top of the default 'test' account.
    await mod._addSyntheticAccountForTesting('alpha', { _marker: 'alpha-client' });
    await mod._addSyntheticAccountForTesting('beta', { _marker: 'beta-client' });

    // Record the auth marker for every drive-service construction.
    originalDriveFactory = google.drive as any;
    (google as any).drive = (opts: any) => {
      authCalls.push({ marker: opts?.auth?._marker });
      return originalDriveFactory(opts);
    };

    const result = mod.createHttpApp('127.0.0.1');
    sessions = result.sessions;
    const started = await startServer(result.app);
    httpServer = started.httpServer;
    baseUrl = started.baseUrl;
  });

  after(async () => {
    (google as any).drive = originalDriveFactory;
    // Reset to a single 'test' account so subsequent test files start clean.
    mod._setAuthClientForTesting({});
    for (const [, s] of sessions) {
      try { await s.transport.close(); } catch { /* ignore */ }
      try { await s.server.close(); } catch { /* ignore */ }
    }
    sessions.clear();
    await new Promise<void>((resolve) => httpServer.close(() => resolve()));
  });

  it('routes each session to its own per-session default account', async () => {
    const sidA = await initSession(baseUrl);
    const sidB = await initSession(baseUrl);
    assert.notEqual(sidA, sidB);

    // Seed a distinct per-session default for each (no tool writes this yet, so
    // set it directly on the SessionStore the resolver reads).
    const sys = mod._getAuthSystemForTesting()!;
    sys.sessions.getOrCreate(sidA).defaultAccountAlias = 'alpha';
    sys.sessions.getOrCreate(sidB).defaultAccountAlias = 'beta';

    // Session A → alpha
    authCalls.length = 0;
    const aBody = await callSearch(baseUrl, sidA);
    assert.notEqual(aBody.result?.isError, true);
    let markers = authCalls.map((c) => c.marker).filter((m) => m !== undefined);
    assert.ok(
      markers.includes('alpha-client'),
      `session A should route to alpha, saw: ${JSON.stringify(markers)}`,
    );
    assert.ok(
      !markers.includes('beta-client'),
      `session A must not route to beta, saw: ${JSON.stringify(markers)}`,
    );

    // Session B → beta (proves the two sessions don't share one key)
    authCalls.length = 0;
    const bBody = await callSearch(baseUrl, sidB);
    assert.notEqual(bBody.result?.isError, true);
    markers = authCalls.map((c) => c.marker).filter((m) => m !== undefined);
    assert.ok(
      markers.includes('beta-client'),
      `session B should route to beta, saw: ${JSON.stringify(markers)}`,
    );
    assert.ok(
      !markers.includes('alpha-client'),
      `session B must not route to alpha, saw: ${JSON.stringify(markers)}`,
    );
  });
});
