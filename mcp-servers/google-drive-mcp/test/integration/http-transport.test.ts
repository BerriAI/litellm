import assert from 'node:assert/strict';
import { describe, it, before, after, mock } from 'node:test';
import http from 'node:http';
import type { Server as HttpServer } from 'node:http';
import { google } from 'googleapis';
import { createAllMocks } from '../helpers/mock-google-apis.js';
import { setTimeout as delay } from 'node:timers/promises';

let _serverModule: any = null;

async function getServerModule() {
  if (!_serverModule) {
    _serverModule = await import('../../src/index.js');
  }
  return _serverModule;
}

const MCP_HEADERS = {
  'Content-Type': 'application/json',
  'Accept': 'application/json, text/event-stream',
};

/** Parse an SSE or JSON response and return the first JSON-RPC message. */
async function parseResponse(res: Response): Promise<any> {
  const contentType = res.headers.get('content-type') || '';
  const text = await res.text();
  if (contentType.includes('text/event-stream')) {
    // Extract the first `data:` line
    for (const line of text.split('\n')) {
      if (line.startsWith('data: ')) {
        return JSON.parse(line.slice(6));
      }
    }
    throw new Error('No data line found in SSE response');
  }
  return JSON.parse(text);
}

async function setupMocks() {
  const mocks = createAllMocks();
  (google as any).drive = mocks.google.drive;
  (google as any).docs = mocks.google.docs;
  (google as any).sheets = mocks.google.sheets;
  (google as any).slides = mocks.google.slides;
  (google as any).calendar = mocks.google.calendar;

  const mod = await getServerModule();
  mod._setAuthClientForTesting({
    request: async () => ({ data: 'mock-auth-request-response' }),
  });
  return mod;
}

async function initializeSession(baseUrl: string): Promise<string> {
  const res = await fetch(`${baseUrl}/mcp`, {
    method: 'POST',
    headers: MCP_HEADERS,
    body: JSON.stringify({
      jsonrpc: '2.0',
      method: 'initialize',
      params: { protocolVersion: '2025-03-26', capabilities: {}, clientInfo: { name: 'test-client', version: '1.0.0' } },
      id: 1,
    }),
  });
  assert.equal(res.status, 200);
  const sessionId = res.headers.get('mcp-session-id')!;
  assert.ok(sessionId);
  await res.text();
  return sessionId;
}

function startServer(app: any): Promise<{ httpServer: HttpServer; baseUrl: string }> {
  return new Promise((resolve) => {
    const httpServer = app.listen(0, '127.0.0.1', () => {
      const addr = httpServer.address();
      let baseUrl = '';
      if (addr && typeof addr === 'object') {
        baseUrl = `http://127.0.0.1:${addr.port}`;
      }
      resolve({ httpServer, baseUrl });
    });
  });
}

async function cleanupServer(httpServer: HttpServer, sessions: Map<string, any>) {
  for (const [, session] of sessions) {
    await session.transport.close();
    await session.server.close();
  }
  sessions.clear();
  await new Promise<void>((resolve) => httpServer.close(() => resolve()));
}

describe('HTTP transport', () => {
  let httpServer: HttpServer;
  let baseUrl: string;
  let sessions: Map<string, any>;

  before(async () => {
    const mod = await setupMocks();
    const result = mod.createHttpApp('127.0.0.1');
    sessions = result.sessions;
    const started = await startServer(result.app);
    httpServer = started.httpServer;
    baseUrl = started.baseUrl;
  });

  after(async () => {
    await cleanupServer(httpServer, sessions);
  });

  it('responds to initialize POST and returns session ID', async () => {
    const res = await fetch(`${baseUrl}/mcp`, {
      method: 'POST',
      headers: MCP_HEADERS,
      body: JSON.stringify({
        jsonrpc: '2.0',
        method: 'initialize',
        params: { protocolVersion: '2025-03-26', capabilities: {}, clientInfo: { name: 'test-client', version: '1.0.0' } },
        id: 1,
      }),
    });

    assert.equal(res.status, 200);
    const sessionId = res.headers.get('mcp-session-id');
    assert.ok(sessionId, 'response should include mcp-session-id header');

    const body = await parseResponse(res);
    assert.equal(body.jsonrpc, '2.0');
    assert.equal(body.id, 1);
    assert.ok(body.result, 'response should have a result');
    assert.ok(body.result.serverInfo, 'result should contain serverInfo');
    assert.equal(body.result.serverInfo.name, 'google-drive-mcp');
  });

  it('reuses session ID for subsequent requests', async () => {
    // Initialize
    const initRes = await fetch(`${baseUrl}/mcp`, {
      method: 'POST',
      headers: MCP_HEADERS,
      body: JSON.stringify({
        jsonrpc: '2.0',
        method: 'initialize',
        params: { protocolVersion: '2025-03-26', capabilities: {}, clientInfo: { name: 'test-client', version: '1.0.0' } },
        id: 1,
      }),
    });
    const sessionId = initRes.headers.get('mcp-session-id')!;
    assert.ok(sessionId);
    // Consume init response
    await initRes.text();

    // Send initialized notification
    await fetch(`${baseUrl}/mcp`, {
      method: 'POST',
      headers: { ...MCP_HEADERS, 'mcp-session-id': sessionId },
      body: JSON.stringify({
        jsonrpc: '2.0',
        method: 'notifications/initialized',
      }),
    });

    // List tools using same session
    const toolsRes = await fetch(`${baseUrl}/mcp`, {
      method: 'POST',
      headers: { ...MCP_HEADERS, 'mcp-session-id': sessionId },
      body: JSON.stringify({
        jsonrpc: '2.0',
        method: 'tools/list',
        params: {},
        id: 2,
      }),
    });

    assert.equal(toolsRes.status, 200);
    const body = await parseResponse(toolsRes);
    assert.equal(body.id, 2);
    assert.ok(Array.isArray(body.result?.tools), 'should return tools array');
    assert.ok(body.result.tools.length > 0, 'should have at least one tool');
  });

  it('returns 400 for non-initialize request without session', async () => {
    const res = await fetch(`${baseUrl}/mcp`, {
      method: 'POST',
      headers: MCP_HEADERS,
      body: JSON.stringify({
        jsonrpc: '2.0',
        method: 'tools/list',
        params: {},
        id: 1,
      }),
    });

    assert.equal(res.status, 400);
    const body = await parseResponse(res);
    assert.ok(body.error, 'should have error');
  });

  it('returns 400 for GET without session ID', async () => {
    const res = await fetch(`${baseUrl}/mcp`);
    assert.equal(res.status, 400);
  });

  it('DELETE closes session', async () => {
    // Initialize a session
    const initRes = await fetch(`${baseUrl}/mcp`, {
      method: 'POST',
      headers: MCP_HEADERS,
      body: JSON.stringify({
        jsonrpc: '2.0',
        method: 'initialize',
        params: { protocolVersion: '2025-03-26', capabilities: {}, clientInfo: { name: 'test-client', version: '1.0.0' } },
        id: 1,
      }),
    });
    const sessionId = initRes.headers.get('mcp-session-id')!;
    assert.ok(sessionId);
    await initRes.text();

    // DELETE the session
    const delRes = await fetch(`${baseUrl}/mcp`, {
      method: 'DELETE',
      headers: { 'mcp-session-id': sessionId },
    });
    assert.equal(delRes.status, 200);

    // Subsequent request with same session should fail
    const postRes = await fetch(`${baseUrl}/mcp`, {
      method: 'POST',
      headers: { ...MCP_HEADERS, 'mcp-session-id': sessionId },
      body: JSON.stringify({
        jsonrpc: '2.0',
        method: 'tools/list',
        params: {},
        id: 2,
      }),
    });
    // Session is gone: 404 tells the client to start a new session.
    assert.equal(postRes.status, 404);
  });
});

// ---------------------------------------------------------------------------
// B1. Session isolation
// ---------------------------------------------------------------------------
describe('HTTP transport — session isolation', () => {
  let httpServer: HttpServer;
  let baseUrl: string;
  let sessions: Map<string, any>;

  before(async () => {
    const mod = await setupMocks();
    const result = mod.createHttpApp('127.0.0.1');
    sessions = result.sessions;
    const started = await startServer(result.app);
    httpServer = started.httpServer;
    baseUrl = started.baseUrl;
  });

  after(async () => {
    await cleanupServer(httpServer, sessions);
  });

  it('two sessions work independently', async () => {
    const sidA = await initializeSession(baseUrl);
    const sidB = await initializeSession(baseUrl);
    assert.notEqual(sidA, sidB);

    for (const sid of [sidA, sidB]) {
      const res = await fetch(`${baseUrl}/mcp`, {
        method: 'POST',
        headers: { ...MCP_HEADERS, 'mcp-session-id': sid },
        body: JSON.stringify({ jsonrpc: '2.0', method: 'tools/list', params: {}, id: 2 }),
      });
      assert.equal(res.status, 200);
      const body = await parseResponse(res);
      assert.ok(Array.isArray(body.result?.tools));
    }
  });

  it('deleting one session does not affect the other', async () => {
    const sidA = await initializeSession(baseUrl);
    const sidB = await initializeSession(baseUrl);

    // Delete A
    const delRes = await fetch(`${baseUrl}/mcp`, {
      method: 'DELETE',
      headers: { 'mcp-session-id': sidA },
    });
    assert.equal(delRes.status, 200);

    // B still works
    const bRes = await fetch(`${baseUrl}/mcp`, {
      method: 'POST',
      headers: { ...MCP_HEADERS, 'mcp-session-id': sidB },
      body: JSON.stringify({ jsonrpc: '2.0', method: 'tools/list', params: {}, id: 3 }),
    });
    assert.equal(bRes.status, 200);

    // A is gone
    const aRes = await fetch(`${baseUrl}/mcp`, {
      method: 'POST',
      headers: { ...MCP_HEADERS, 'mcp-session-id': sidA },
      body: JSON.stringify({ jsonrpc: '2.0', method: 'tools/list', params: {}, id: 4 }),
    });
    assert.equal(aRes.status, 404);
  });
});

// ---------------------------------------------------------------------------
// B2. Session idle timeout
// ---------------------------------------------------------------------------
describe('HTTP transport — session idle timeout', () => {
  let httpServer: HttpServer;
  let baseUrl: string;
  let sessions: Map<string, any>;

  before(async () => {
    const mod = await setupMocks();
    // Wide margins on purpose: with the whole suite running in parallel,
    // event-loop lag of tens of ms is routine and tighter values flake.
    const result = mod.createHttpApp('127.0.0.1', { sessionIdleTimeoutMs: 400 });
    sessions = result.sessions;
    const started = await startServer(result.app);
    httpServer = started.httpServer;
    baseUrl = started.baseUrl;
  });

  after(async () => {
    await cleanupServer(httpServer, sessions);
  });

  it('idle session is evicted after timeout', async () => {
    const sid = await initializeSession(baseUrl);
    assert.ok(sessions.has(sid));

    await delay(1000);

    const res = await fetch(`${baseUrl}/mcp`, {
      method: 'POST',
      headers: { ...MCP_HEADERS, 'mcp-session-id': sid },
      body: JSON.stringify({ jsonrpc: '2.0', method: 'tools/list', params: {}, id: 2 }),
    });
    // Streamable HTTP: an unknown session id is 404, and the body carries the
    // SDK's own code/message so clients see one shape for "start over".
    assert.equal(res.status, 404);
    const body = await parseResponse(res);
    assert.equal(body.error.code, -32001);
    assert.equal(body.error.message, 'Session not found');
    assert.equal(sessions.has(sid), false);
  });

  it('a client can recover from an evicted session by re-initializing', async () => {
    const sid = await initializeSession(baseUrl);
    await delay(1000);
    assert.equal(sessions.has(sid), false);

    // The spec says the client sends a fresh initialize without a session id...
    const fresh = await initializeSession(baseUrl);
    assert.notEqual(fresh, sid);
    assert.ok(sessions.has(fresh));

    // ...but a client that leaves the stale id on the initialize is tolerated
    // too, rather than being bounced with a second 404.
    const lenient = await fetch(`${baseUrl}/mcp`, {
      method: 'POST',
      headers: { ...MCP_HEADERS, 'mcp-session-id': sid },
      body: JSON.stringify({
        jsonrpc: '2.0',
        method: 'initialize',
        params: { protocolVersion: '2025-03-26', capabilities: {}, clientInfo: { name: 'test-client', version: '1.0.0' } },
        id: 1,
      }),
    });
    assert.equal(lenient.status, 200);
    const lenientSid = lenient.headers.get('mcp-session-id');
    await lenient.text();
    assert.ok(lenientSid && lenientSid !== sid);
    assert.ok(sessions.has(lenientSid!));
  });

  it('activity resets the idle timer', async () => {
    const sid = await initializeSession(baseUrl);

    // Wait less than timeout, then send a request to reset timer
    await delay(150);
    const midRes = await fetch(`${baseUrl}/mcp`, {
      method: 'POST',
      headers: { ...MCP_HEADERS, 'mcp-session-id': sid },
      body: JSON.stringify({ jsonrpc: '2.0', method: 'tools/list', params: {}, id: 2 }),
    });
    assert.equal(midRes.status, 200);
    await midRes.text();

    // Wait again — total time since last activity < timeout
    await delay(150);
    const res = await fetch(`${baseUrl}/mcp`, {
      method: 'POST',
      headers: { ...MCP_HEADERS, 'mcp-session-id': sid },
      body: JSON.stringify({ jsonrpc: '2.0', method: 'tools/list', params: {}, id: 3 }),
    });
    assert.equal(res.status, 200);
    assert.ok(sessions.has(sid));
  });
});

// ---------------------------------------------------------------------------
// B3. Server.close() on shutdown
// ---------------------------------------------------------------------------
describe('HTTP transport — server.close()', () => {
  let httpServer: HttpServer;
  let baseUrl: string;
  let sessions: Map<string, any>;

  before(async () => {
    const mod = await setupMocks();
    const result = mod.createHttpApp('127.0.0.1');
    sessions = result.sessions;
    const started = await startServer(result.app);
    httpServer = started.httpServer;
    baseUrl = started.baseUrl;
  });

  after(async () => {
    await cleanupServer(httpServer, sessions);
  });

  it('DELETE calls server.close()', async () => {
    const sid = await initializeSession(baseUrl);
    const session = sessions.get(sid)!;
    const closeSpy = mock.fn(session.server.close.bind(session.server));
    session.server.close = closeSpy;

    await fetch(`${baseUrl}/mcp`, {
      method: 'DELETE',
      headers: { 'mcp-session-id': sid },
    });

    assert.equal(closeSpy.mock.callCount(), 1);
  });

  it('transport onclose cleans up session from map', async () => {
    const sid = await initializeSession(baseUrl);
    assert.ok(sessions.has(sid));

    const session = sessions.get(sid)!;
    await session.transport.close();

    // onclose handler should have removed it
    assert.equal(sessions.has(sid), false);
  });
});

// ---------------------------------------------------------------------------
// B4. Error handling
// ---------------------------------------------------------------------------
describe('HTTP transport — error handling', () => {
  let httpServer: HttpServer;
  let baseUrl: string;
  let sessions: Map<string, any>;

  before(async () => {
    const mod = await setupMocks();
    const result = mod.createHttpApp('127.0.0.1');
    sessions = result.sessions;
    const started = await startServer(result.app);
    httpServer = started.httpServer;
    baseUrl = started.baseUrl;
  });

  after(async () => {
    await cleanupServer(httpServer, sessions);
  });

  it('POST with invalid JSON returns 400', async () => {
    const res = await fetch(`${baseUrl}/mcp`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Accept': 'application/json, text/event-stream' },
      body: 'not json',
    });
    assert.ok([400, 500].includes(res.status), `expected 400 or 500, got ${res.status}`);
  });

  it('POST with valid JSON-RPC but unknown method (no session) returns 400', async () => {
    const res = await fetch(`${baseUrl}/mcp`, {
      method: 'POST',
      headers: MCP_HEADERS,
      body: JSON.stringify({ jsonrpc: '2.0', method: 'foo', id: 1 }),
    });
    assert.equal(res.status, 400);
    const body = await parseResponse(res);
    assert.ok(body.error);
  });

  it('DELETE with non-existent session ID returns 404', async () => {
    const res = await fetch(`${baseUrl}/mcp`, {
      method: 'DELETE',
      headers: { 'mcp-session-id': 'non-existent-uuid' },
    });
    assert.equal(res.status, 404);
    const body = await parseResponse(res);
    assert.equal(body.error.code, -32001);
  });

  it('GET with non-existent session ID returns 404', async () => {
    const res = await fetch(`${baseUrl}/mcp`, {
      method: 'GET',
      headers: { 'mcp-session-id': 'non-existent-uuid' },
    });
    assert.equal(res.status, 404);
    const body = await parseResponse(res);
    assert.equal(body.error.code, -32001);
  });
});

// ---------------------------------------------------------------------------
// B5. DNS rebinding protection
// ---------------------------------------------------------------------------
describe('HTTP transport — DNS rebinding protection', () => {
  let httpServer: HttpServer;
  let baseUrl: string;
  let sessions: Map<string, any>;
  let port: number;

  before(async () => {
    const mod = await setupMocks();
    const result = mod.createHttpApp('127.0.0.1');
    sessions = result.sessions;
    const started = await startServer(result.app);
    httpServer = started.httpServer;
    baseUrl = started.baseUrl;
    const addr = httpServer.address();
    port = (addr && typeof addr === 'object') ? addr.port : 0;
  });

  after(async () => {
    await cleanupServer(httpServer, sessions);
  });

  it('request with spoofed Host header is rejected', async () => {
    const body = JSON.stringify({
      jsonrpc: '2.0',
      method: 'initialize',
      params: { protocolVersion: '2025-03-26', capabilities: {}, clientInfo: { name: 'test', version: '1.0.0' } },
      id: 1,
    });
    const status = await new Promise<number>((resolve, reject) => {
      const req = http.request({
        hostname: '127.0.0.1',
        port,
        path: '/mcp',
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json, text/event-stream',
          'Host': 'evil.com',
          'Content-Length': Buffer.byteLength(body),
        },
      }, (res) => {
        res.resume();
        resolve(res.statusCode!);
      });
      req.on('error', reject);
      req.write(body);
      req.end();
    });
    assert.equal(status, 403);
  });

  it('request with correct Host header succeeds', async () => {
    const res = await fetch(`${baseUrl}/mcp`, {
      method: 'POST',
      headers: MCP_HEADERS,
      body: JSON.stringify({
        jsonrpc: '2.0',
        method: 'initialize',
        params: { protocolVersion: '2025-03-26', capabilities: {}, clientInfo: { name: 'test', version: '1.0.0' } },
        id: 1,
      }),
    });
    assert.equal(res.status, 200);
  });
});

// ---------------------------------------------------------------------------
// B7. server.close() symmetry — each session gets its own Server instance
// ---------------------------------------------------------------------------
describe('HTTP transport — server instance uniqueness', () => {
  let httpServer: HttpServer;
  let baseUrl: string;
  let sessions: Map<string, any>;

  before(async () => {
    const mod = await setupMocks();
    const result = mod.createHttpApp('127.0.0.1');
    sessions = result.sessions;
    const started = await startServer(result.app);
    httpServer = started.httpServer;
    baseUrl = started.baseUrl;
  });

  after(async () => {
    await cleanupServer(httpServer, sessions);
  });

  it('each HTTP session gets its own Server instance', async () => {
    const sidA = await initializeSession(baseUrl);
    const sidB = await initializeSession(baseUrl);

    const serverA = sessions.get(sidA)!.server;
    const serverB = sessions.get(sidB)!.server;
    assert.notEqual(serverA, serverB);
  });
});
