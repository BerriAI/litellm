import assert from 'node:assert/strict';
import { after, before, describe, it, mock } from 'node:test';
import { createHash, randomBytes } from 'node:crypto';
import type { Server as HttpServer } from 'node:http';
import { google } from 'googleapis';

import { createAllMocks } from '../helpers/mock-google-apis.js';
import type { TeamConfig } from '../../src/auth/team/config.js';
import type { GoogleIdp, GoogleTokenResult } from '../../src/auth/team/googleIdp.js';
import { InMemoryTeamStore } from '../../src/auth/team/memoryStore.js';
import { createTeamRuntime, type TeamRuntime } from '../../src/auth/team/runtime.js';

// ---------------------------------------------------------------------------
// Team-mode dispatch: bearer-guarded /mcp, per-user tool routing, session↔user
// binding, and the invalid_grant self-healing path — all over real HTTP with a
// faked Google IdP and mocked Google APIs.
// ---------------------------------------------------------------------------

const DRIVE_SCOPE = 'https://www.googleapis.com/auth/drive';
const DOCS_SCOPE = 'https://www.googleapis.com/auth/documents';

const MCP_HEADERS = {
  'Content-Type': 'application/json',
  Accept: 'application/json, text/event-stream',
};

function makeConfig(): TeamConfig {
  return {
    issuerUrl: new URL('http://127.0.0.1:3100'),
    googleRedirectUri: 'http://127.0.0.1:3100/oauth/google/callback',
    allowedDomains: [],
    allowedRedirectUris: [],
    tokenTtlMs: 3600_000,
    store: 'memory',
    storePath: '/unused',
    allowedHosts: ['127.0.0.1', 'localhost', '[::1]'],
    googleScopes: [DRIVE_SCOPE, DOCS_SCOPE, 'openid', 'https://www.googleapis.com/auth/userinfo.email'],
    advertisedScopes: [DRIVE_SCOPE, DOCS_SCOPE],
    ...{},
  };
}

interface FakeUser {
  sub: string;
  email: string;
  hd?: string;
  scope: string;
}

class FakeIdp implements GoogleIdp {
  current!: FakeUser;
  buildConsentUrl(state: string): string {
    return `https://fake-google.example/consent?state=${encodeURIComponent(state)}`;
  }
  async exchangeCode(code: string): Promise<GoogleTokenResult> {
    if (code !== 'fake-google-code') throw new Error('unexpected google code');
    return {
      tokens: {
        refresh_token: `g-refresh-${this.current.sub}`,
        access_token: `g-access-${this.current.sub}`,
        scope: this.current.scope,
        expiry_date: Date.now() + 3600_000,
      },
      identity: {
        sub: this.current.sub,
        email: this.current.email,
        ...(this.current.hd ? { hd: this.current.hd } : {}),
      },
    };
  }
  async revokeGrant(): Promise<void> {}
}

function pkcePair(): { verifier: string; challenge: string } {
  const verifier = randomBytes(32).toString('base64url');
  const challenge = createHash('sha256').update(verifier).digest('base64url');
  return { verifier, challenge };
}

const CLIENT_REDIRECT = 'https://client.example/callback';

async function parseResponse(res: globalThis.Response): Promise<any> {
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

describe('Team mode — bearer-guarded per-user dispatch', () => {
  let httpServer: HttpServer;
  let sessions: Map<string, any>;
  let runtime: TeamRuntime;
  let idp: FakeIdp;
  let store: InMemoryTeamStore;
  let mod: any;
  let baseUrl: string;
  let clientId: string;
  /** Auth clients handed to google.drive(), in call order. */
  let driveAuths: any[];

  before(async () => {
    mod = await import('../../src/index.js');

    const mocks = createAllMocks();
    driveAuths = [];
    (google as any).drive = (opts: any) => {
      driveAuths.push(opts?.auth);
      return mocks.google.drive();
    };
    (google as any).docs = mocks.google.docs;
    (google as any).sheets = mocks.google.sheets;
    (google as any).slides = mocks.google.slides;
    (google as any).calendar = mocks.google.calendar;

    idp = new FakeIdp();
    store = new InMemoryTeamStore();
    runtime = await createTeamRuntime(makeConfig(), { store, idp });
    mod._setTeamRuntimeForTesting(runtime);

    const created = mod.createHttpApp('127.0.0.1', { teamAuth: runtime });
    sessions = created.sessions;
    await new Promise<void>((resolve) => {
      httpServer = created.app.listen(0, '127.0.0.1', () => resolve());
    });
    const addr = httpServer.address() as { port: number };
    baseUrl = `http://127.0.0.1:${addr.port}`;

    const reg = await fetch(`${baseUrl}/register`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        redirect_uris: [CLIENT_REDIRECT],
        token_endpoint_auth_method: 'none',
      }),
    });
    clientId = (await reg.json()).client_id;
  });

  after(async () => {
    mod._setTeamRuntimeForTesting(null);
    runtime.stop();
    for (const [, session] of sessions) {
      await session.transport.close();
      await session.server.close();
    }
    sessions.clear();
    await new Promise<void>((resolve) => httpServer.close(() => resolve()));
  });

  /** Run the whole OAuth dance for a fake user; returns their MCP bearer.
   * Pass requestedScope to narrow the issued token below the user's grant. */
  async function signIn(user: FakeUser, requestedScope?: string): Promise<string> {
    idp.current = user;
    const { verifier, challenge } = pkcePair();
    const authUrl = new URL(`${baseUrl}/authorize`);
    authUrl.searchParams.set('client_id', clientId);
    authUrl.searchParams.set('response_type', 'code');
    authUrl.searchParams.set('redirect_uri', CLIENT_REDIRECT);
    authUrl.searchParams.set('code_challenge', challenge);
    authUrl.searchParams.set('code_challenge_method', 'S256');
    authUrl.searchParams.set('state', 'st');
    if (requestedScope) authUrl.searchParams.set('scope', requestedScope);
    const authRes = await fetch(authUrl, { redirect: 'manual' });
    const googleState = new URL(authRes.headers.get('location')!).searchParams.get('state')!;

    const cbRes = await fetch(
      `${baseUrl}/oauth/google/callback?state=${googleState}&code=fake-google-code`,
      { redirect: 'manual' },
    );
    const code = new URL(cbRes.headers.get('location')!).searchParams.get('code')!;

    const tokenRes = await fetch(`${baseUrl}/token`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body: new URLSearchParams({
        grant_type: 'authorization_code',
        client_id: clientId,
        code,
        code_verifier: verifier,
        redirect_uri: CLIENT_REDIRECT,
      }).toString(),
    });
    assert.equal(tokenRes.status, 200);
    return (await tokenRes.json()).access_token;
  }

  async function initSession(bearer: string): Promise<string> {
    const res = await fetch(`${baseUrl}/mcp`, {
      method: 'POST',
      headers: { ...MCP_HEADERS, Authorization: `Bearer ${bearer}` },
      body: JSON.stringify({
        jsonrpc: '2.0',
        method: 'initialize',
        params: {
          protocolVersion: '2025-03-26',
          capabilities: {},
          clientInfo: { name: 'test', version: '1.0' },
        },
        id: 1,
      }),
    });
    assert.equal(res.status, 200);
    const sid = res.headers.get('mcp-session-id')!;
    await res.text();
    assert.ok(sid);
    return sid;
  }

  async function mcpPost(bearer: string | undefined, sessionId: string | undefined, body: unknown) {
    return fetch(`${baseUrl}/mcp`, {
      method: 'POST',
      headers: {
        ...MCP_HEADERS,
        ...(bearer ? { Authorization: `Bearer ${bearer}` } : {}),
        ...(sessionId ? { 'mcp-session-id': sessionId } : {}),
      },
      body: JSON.stringify(body),
    });
  }

  function toolsCall(name: string, args: Record<string, unknown>, id = 2) {
    return { jsonrpc: '2.0', method: 'tools/call', params: { name, arguments: args }, id };
  }

  it('rejects unauthenticated /mcp requests with 401 + resource_metadata pointer', async () => {
    for (const method of ['POST', 'GET', 'DELETE'] as const) {
      const res = await fetch(`${baseUrl}/mcp`, {
        method,
        headers: method === 'POST' ? MCP_HEADERS : { Accept: 'text/event-stream' },
        ...(method === 'POST' ? { body: JSON.stringify({ jsonrpc: '2.0', method: 'ping', id: 1 }) } : {}),
      });
      assert.equal(res.status, 401, `${method} /mcp must 401 without a bearer`);
      const www = res.headers.get('www-authenticate') ?? '';
      assert.ok(
        www.includes('resource_metadata="http://127.0.0.1:3100/.well-known/oauth-protected-resource"'),
        `WWW-Authenticate must point at the resource metadata (got: ${www})`,
      );
      await res.text();
    }

    const bad = await mcpPost('mcp_at_forged-token', undefined, { jsonrpc: '2.0', method: 'ping', id: 1 });
    assert.equal(bad.status, 401);
    await bad.text();
  });

  it('lists tools without the account parameter and without manage_accounts', async () => {
    const bearer = await signIn({ sub: 'sub-alice', email: 'alice@corp.example', scope: DRIVE_SCOPE });
    const sid = await initSession(bearer);
    const res = await fetch(`${baseUrl}/mcp`, {
      method: 'POST',
      headers: { ...MCP_HEADERS, Authorization: `Bearer ${bearer}`, 'mcp-session-id': sid },
      body: JSON.stringify({ jsonrpc: '2.0', method: 'tools/list', id: 2 }),
    });
    const body = await parseResponse(res);
    const tools = body.result.tools;
    assert.ok(tools.length > 50, 'tool catalog present');
    assert.equal(tools.find((t: any) => t.name === 'manage_accounts'), undefined);
    const search = tools.find((t: any) => t.name === 'search');
    assert.ok(search);
    assert.equal(search.inputSchema.properties.account, undefined, 'no account param in team mode');
  });

  it('routes each user\'s calls to their own Google client', async () => {
    const alice = await signIn({ sub: 'sub-alice', email: 'alice@corp.example', scope: DRIVE_SCOPE });
    const bob = await signIn({ sub: 'sub-bob', email: 'bob@corp.example', scope: DRIVE_SCOPE });
    const aliceSid = await initSession(alice);
    const bobSid = await initSession(bob);

    driveAuths.length = 0;
    const r1 = await mcpPost(alice, aliceSid, toolsCall('search', { query: 'alpha' }));
    const b1 = await parseResponse(r1);
    assert.notEqual(b1.result.isError, true, JSON.stringify(b1.result?.content));
    const r2 = await mcpPost(bob, bobSid, toolsCall('search', { query: 'beta' }));
    const b2 = await parseResponse(r2);
    assert.notEqual(b2.result.isError, true);

    const refreshTokens = driveAuths.map((a) => a?.credentials?.refresh_token);
    assert.deepEqual(refreshTokens, ['g-refresh-sub-alice', 'g-refresh-sub-bob']);

    // Cached per sub: a second call builds no new drive client.
    await (await mcpPost(alice, aliceSid, toolsCall('search', { query: 'again' }, 3))).text();
    assert.equal(driveAuths.length, 2);
  });

  it('attributes dispatch and session logs to the acting user (sub + email)', async () => {
    const bearer = await signIn({ sub: 'sub-carol', email: 'carol@corp.example', scope: DRIVE_SCOPE });
    // log() writes through console.error; capture it around a session + call.
    const errMock = mock.method(console, 'error', () => {});
    try {
      const sid = await initSession(bearer);
      await (await mcpPost(bearer, sid, toolsCall('search', { query: 'audit' }))).text();
      const lines = errMock.mock.calls.map((c) => c.arguments.join(' '));

      const created = lines.find((l) => l.includes('New session created'));
      assert.ok(created, 'session creation is logged');
      assert.ok(
        created!.includes('"sub":"sub-carol"') && created!.includes('"email":"carol@corp.example"'),
        `session line names the user (got: ${created})`,
      );

      const dispatch = lines.find(
        (l) => l.includes('Handling tool request (team)') && l.includes('"tool":"search"'),
      );
      assert.ok(dispatch, 'team dispatch is logged');
      assert.ok(
        dispatch!.includes('"sub":"sub-carol"') && dispatch!.includes('"email":"carol@corp.example"'),
        `dispatch line names the user (got: ${dispatch})`,
      );
    } finally {
      errMock.mock.restore();
    }
  });

  it('rejects the account argument and manage_accounts outright', async () => {
    const bearer = await signIn({ sub: 'sub-alice', email: 'alice@corp.example', scope: DRIVE_SCOPE });
    const sid = await initSession(bearer);

    const withAccount = await parseResponse(
      await mcpPost(bearer, sid, toolsCall('search', { query: 'x', account: 'sub-bob' })),
    );
    assert.equal(withAccount.result.isError, true);
    assert.match(withAccount.result.content[0].text, /'account' parameter is not available in team mode/);

    const manage = await parseResponse(
      await mcpPost(bearer, sid, toolsCall('manage_accounts', { action: 'list' }, 3)),
    );
    assert.equal(manage.result.isError, true);
    assert.match(manage.result.content[0].text, /manage_accounts is not available in team mode/);
  });

  it('binds sessions to the initializing user: a foreign bearer cannot touch them', async () => {
    const alice = await signIn({ sub: 'sub-alice', email: 'alice@corp.example', scope: DRIVE_SCOPE });
    const bob = await signIn({ sub: 'sub-bob', email: 'bob@corp.example', scope: DRIVE_SCOPE });
    const aliceSid = await initSession(alice);

    // POST with a non-initialize body: indistinguishable from an unknown
    // session, which is a 404 "Session not found".
    const post = await mcpPost(bob, aliceSid, { jsonrpc: '2.0', method: 'tools/list', id: 2 });
    assert.equal(post.status, 404);
    const postBody = await post.json();
    assert.equal(postBody.error.code, -32001);
    assert.equal(postBody.error.message, 'Session not found');

    // GET (SSE stream) and DELETE: same 404 as an unknown session.
    const get = await fetch(`${baseUrl}/mcp`, {
      headers: { Accept: 'text/event-stream', Authorization: `Bearer ${bob}`, 'mcp-session-id': aliceSid },
    });
    assert.equal(get.status, 404);
    await get.text();

    const del = await fetch(`${baseUrl}/mcp`, {
      method: 'DELETE',
      headers: { Authorization: `Bearer ${bob}`, 'mcp-session-id': aliceSid },
    });
    assert.equal(del.status, 404);
    await del.text();

    // Alice's session survived the hijack attempts.
    const alive = await mcpPost(alice, aliceSid, { jsonrpc: '2.0', method: 'tools/list', id: 3 });
    assert.equal(alive.status, 200);
    await alive.text();
  });

  it('enforces per-tool scope gates against the user\'s actually-granted scopes', async () => {
    const bearer = await signIn({
      sub: 'sub-docs-only',
      email: 'docs@corp.example',
      scope: 'https://www.googleapis.com/auth/documents',
    });
    const sid = await initSession(bearer);
    const body = await parseResponse(await mcpPost(bearer, sid, toolsCall('search', { query: 'x' })));
    assert.equal(body.result.isError, true);
    assert.match(body.result.content[0].text, /lacks the required scope/);
    assert.match(body.result.content[0].text, /Reconnect this connector/);
  });

  it('enforces the bearer token\'s own scopes, not just the Google grant', async () => {
    // The user's Google grant covers BOTH drive and docs, but the client
    // requests a token narrowed to docs only. A drive tool must be refused on
    // the TOKEN scope even though the underlying grant would allow it —
    // otherwise the advertised scope model is a no-op at the resource server.
    const bearer = await signIn(
      { sub: 'sub-narrow', email: 'narrow@corp.example', scope: `${DRIVE_SCOPE} ${DOCS_SCOPE}` },
      DOCS_SCOPE,
    );
    const sid = await initSession(bearer);
    const body = await parseResponse(await mcpPost(bearer, sid, toolsCall('search', { query: 'x' })));
    assert.equal(body.result.isError, true);
    assert.match(body.result.content[0].text, /access token is not authorized/);
  });

  it('hides local single-user auth tools in team mode and rejects them if called', async () => {
    const bearer = await signIn({ sub: 'sub-alice', email: 'alice@corp.example', scope: DRIVE_SCOPE });
    const sid = await initSession(bearer);

    const listRes = await fetch(`${baseUrl}/mcp`, {
      method: 'POST',
      headers: { ...MCP_HEADERS, Authorization: `Bearer ${bearer}`, 'mcp-session-id': sid },
      body: JSON.stringify({ jsonrpc: '2.0', method: 'tools/list', id: 2 }),
    });
    const tools = (await parseResponse(listRes)).result.tools;
    assert.equal(tools.find((t: any) => t.name === 'authGetStatus'), undefined, 'authGetStatus hidden');
    assert.equal(tools.find((t: any) => t.name === 'authListScopes'), undefined, 'authListScopes hidden');

    const called = await parseResponse(await mcpPost(bearer, sid, toolsCall('authGetStatus', {}, 4)));
    assert.equal(called.result.isError, true);
    assert.match(called.result.content[0].text, /not available in team mode/);
  });

  it('never touches the local account system', async () => {
    assert.equal(mod._getAuthSystemForTesting(), null, 'team dispatch must not build an AuthSystem');
  });

  it('self-heals a revoked Google grant: tool error, then 401, then re-auth works', async () => {
    const bearer = await signIn({ sub: 'sub-revoked', email: 'rev@corp.example', scope: DRIVE_SCOPE });
    const sid = await initSession(bearer);

    // Expire the cached client's credentials and make its refresh fail with
    // invalid_grant, as Google does after the user revokes access.
    const client = await runtime.clientFactory.getClient('sub-revoked');
    client.setCredentials({ ...client.credentials, expiry_date: Date.now() - 60_000 });
    (client as unknown as { refreshAccessToken: () => Promise<never> }).refreshAccessToken =
      async () => {
        // gaxios-7-realistic shape: top-level `status` plus the parsed body.
        const err = new Error('invalid_grant') as Error & {
          status?: number;
          response?: { status?: number; data?: { error?: string } };
        };
        err.status = 400;
        err.response = { status: 400, data: { error: 'invalid_grant' } };
        throw err;
      };

    // 1) The in-flight call surfaces an actionable error…
    const failing = await parseResponse(await mcpPost(bearer, sid, toolsCall('search', { query: 'x' })));
    assert.equal(failing.result.isError, true);
    assert.match(failing.result.content[0].text, /expired or been revoked/);
    assert.match(failing.result.content[0].text, /Reconnect this connector/);

    // 2) …the user's MCP tokens were revoked, so the next request 401s with
    // the discovery pointer that makes the client re-run OAuth…
    const next = await mcpPost(bearer, sid, toolsCall('search', { query: 'x' }, 3));
    assert.equal(next.status, 401);
    assert.ok((next.headers.get('www-authenticate') ?? '').includes('resource_metadata='));
    await next.text();

    // 3) …and completing a fresh sign-in clears the flag and works again.
    const fresh = await signIn({ sub: 'sub-revoked', email: 'rev@corp.example', scope: DRIVE_SCOPE });
    assert.equal((await store.getUser('sub-revoked'))?.needsReauth, undefined);
    const freshSid = await initSession(fresh);
    const ok = await parseResponse(await mcpPost(fresh, freshSid, toolsCall('search', { query: 'x' })));
    assert.notEqual(ok.result.isError, true);
  });
});
