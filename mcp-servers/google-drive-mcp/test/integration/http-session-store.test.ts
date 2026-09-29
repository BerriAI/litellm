import assert from 'node:assert/strict';
import { describe, it, before, after } from 'node:test';
import type { Server as HttpServer } from 'node:http';
import { google } from 'googleapis';
import { createAllMocks } from '../helpers/mock-google-apis.js';

// ---------------------------------------------------------------------------
// HTTP transport ↔ SessionStore integration.
//
// The dispatch layer threads `Mcp-Session-Id` into AccountResolver, so the
// AuthSystem's SessionStore must mirror the HTTP session lifecycle: created
// on initialize, removed on DELETE / transport.close / idle timeout. This
// test verifies that wiring directly so a regression in index.ts (forgetting
// to call sessions.delete on a teardown branch) would fail loudly.
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

async function setupMocks() {
  const mocks = createAllMocks();
  (google as any).drive = mocks.google.drive;
  (google as any).docs = mocks.google.docs;
  (google as any).sheets = mocks.google.sheets;
  (google as any).slides = mocks.google.slides;
  (google as any).calendar = mocks.google.calendar;
  const mod = await getServerModule();
  mod._setAuthClientForTesting({ request: async () => ({ data: 'mock' }) });
  return mod;
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
  return sid;
}

describe('HTTP session lifecycle ↔ AuthSystem.sessions', () => {
  let httpServer: HttpServer;
  let baseUrl: string;
  let httpSessions: Map<string, any>;
  let mod: any;

  before(async () => {
    mod = await setupMocks();
    const result = mod.createHttpApp('127.0.0.1');
    httpSessions = result.sessions;
    const started = await startServer(result.app);
    httpServer = started.httpServer;
    baseUrl = started.baseUrl;
  });

  after(async () => {
    for (const [, s] of httpSessions) {
      await s.transport.close();
      await s.server.close();
    }
    httpSessions.clear();
    await new Promise<void>((resolve) => httpServer.close(() => resolve()));
  });

  it('initialize POST registers the session in AuthSystem.sessions', async () => {
    const sys = mod._getAuthSystemForTesting()!;
    const before = sys.sessions.size();
    const sid = await initSession(baseUrl);
    assert.ok(sys.sessions.get(sid), 'expected SessionStore to track the new session');
    assert.equal(sys.sessions.size(), before + 1);
  });

  it('DELETE removes the session from AuthSystem.sessions', async () => {
    const sys = mod._getAuthSystemForTesting()!;
    const sid = await initSession(baseUrl);
    assert.ok(sys.sessions.get(sid));

    const delRes = await fetch(`${baseUrl}/mcp`, {
      method: 'DELETE',
      headers: { 'mcp-session-id': sid },
    });
    assert.equal(delRes.status, 200);
    assert.equal(sys.sessions.get(sid), undefined, 'SessionStore should drop the session on DELETE');
  });

  it('transport.close (e.g. peer disconnect) clears AuthSystem.sessions entry', async () => {
    const sys = mod._getAuthSystemForTesting()!;
    const sid = await initSession(baseUrl);
    assert.ok(sys.sessions.get(sid));

    const session = httpSessions.get(sid)!;
    await session.transport.close();

    // The transport.onclose handler runs synchronously-ish; give it a microtask.
    await new Promise((r) => setTimeout(r, 10));
    assert.equal(sys.sessions.get(sid), undefined, 'transport.close should propagate to SessionStore');
  });
});

// ---------------------------------------------------------------------------
// Idle timeout propagation
// ---------------------------------------------------------------------------

describe('HTTP idle timeout ↔ AuthSystem.sessions', () => {
  let httpServer: HttpServer;
  let baseUrl: string;
  let httpSessions: Map<string, any>;
  let mod: any;

  before(async () => {
    mod = await setupMocks();
    const result = mod.createHttpApp('127.0.0.1', { sessionIdleTimeoutMs: 50 });
    httpSessions = result.sessions;
    const started = await startServer(result.app);
    httpServer = started.httpServer;
    baseUrl = started.baseUrl;
  });

  after(async () => {
    for (const [, s] of httpSessions) {
      try { await s.transport.close(); } catch { /* ignore */ }
      try { await s.server.close(); } catch { /* ignore */ }
    }
    httpSessions.clear();
    await new Promise<void>((resolve) => httpServer.close(() => resolve()));
  });

  it('idle eviction removes the entry from AuthSystem.sessions', async () => {
    const sys = mod._getAuthSystemForTesting()!;
    const sid = await initSession(baseUrl);
    assert.ok(sys.sessions.get(sid));

    // Wait past the 50ms idle timeout + a small buffer.
    await new Promise((r) => setTimeout(r, 200));

    assert.equal(sys.sessions.get(sid), undefined, 'idle eviction should clear SessionStore entry');
  });
});
