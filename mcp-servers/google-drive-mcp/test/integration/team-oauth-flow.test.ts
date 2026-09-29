import assert from 'node:assert/strict';
import { after, before, describe, it } from 'node:test';
import { createHash, randomBytes } from 'node:crypto';
import type { Server as HttpServer } from 'node:http';

import type { TeamConfig } from '../../src/auth/team/config.js';
import type { GoogleIdp, GoogleTokenResult } from '../../src/auth/team/googleIdp.js';
import { InMemoryTeamStore } from '../../src/auth/team/memoryStore.js';
import { createTeamRuntime, type TeamRuntime } from '../../src/auth/team/runtime.js';

// ---------------------------------------------------------------------------
// Full two-hop OAuth flow over real HTTP: discovery → dynamic client
// registration → /authorize (PKCE) → Google callback (faked IdP) → /token →
// refresh → revoke. Uses the same in-process harness as http-transport.test.ts.
// ---------------------------------------------------------------------------

const DRIVE_SCOPE = 'https://www.googleapis.com/auth/drive';

function makeConfig(overrides: Partial<TeamConfig> = {}): TeamConfig {
  return {
    issuerUrl: new URL('http://127.0.0.1:3100'),
    googleRedirectUri: 'http://127.0.0.1:3100/oauth/google/callback',
    allowedDomains: [],
    allowedRedirectUris: [],
    tokenTtlMs: 3600_000,
    store: 'memory',
    storePath: '/unused',
    allowedHosts: ['127.0.0.1', 'localhost', '[::1]'],
    googleScopes: [DRIVE_SCOPE, 'openid', 'https://www.googleapis.com/auth/userinfo.email'],
    advertisedScopes: [DRIVE_SCOPE],
    ...overrides,
  };
}

class FakeIdp implements GoogleIdp {
  identity = { sub: 'google-sub-alice', email: 'alice@example.com', hd: 'example.com' };
  tokens = {
    refresh_token: 'g-refresh-alice',
    access_token: 'g-access-alice',
    scope: `${DRIVE_SCOPE} openid`,
    expiry_date: Date.now() + 3600_000,
  };
  revoked: string[] = [];
  exchangeCalls = 0;

  buildConsentUrl(state: string): string {
    return `https://fake-google.example/consent?state=${encodeURIComponent(state)}`;
  }
  async exchangeCode(code: string): Promise<GoogleTokenResult> {
    this.exchangeCalls += 1;
    if (code !== 'fake-google-code') throw new Error('unexpected google code');
    return { tokens: { ...this.tokens }, identity: { ...this.identity } };
  }
  async revokeGrant(token: string): Promise<void> {
    this.revoked.push(token);
  }
}

function pkcePair(): { verifier: string; challenge: string } {
  const verifier = randomBytes(32).toString('base64url');
  const challenge = createHash('sha256').update(verifier).digest('base64url');
  return { verifier, challenge };
}

const CLIENT_REDIRECT = 'https://client.example/callback';

interface Flow {
  baseUrl: string;
  clientId: string;
}

/** Register a public client via DCR. */
async function registerClient(baseUrl: string): Promise<string> {
  const res = await fetch(`${baseUrl}/register`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      client_name: 'test-connector',
      redirect_uris: [CLIENT_REDIRECT],
      token_endpoint_auth_method: 'none',
      grant_types: ['authorization_code', 'refresh_token'],
      response_types: ['code'],
    }),
  });
  assert.equal(res.status, 201);
  const body = await res.json();
  assert.ok(body.client_id);
  assert.equal(body.client_secret, undefined, 'public client gets no secret');
  return body.client_id;
}

/** Drive /authorize → Google consent redirect; returns our Google-hop state. */
async function startAuthorize(
  flow: Flow,
  challenge: string,
  clientState = 'mcp-client-state',
): Promise<string> {
  const url = new URL(`${flow.baseUrl}/authorize`);
  url.searchParams.set('client_id', flow.clientId);
  url.searchParams.set('response_type', 'code');
  url.searchParams.set('redirect_uri', CLIENT_REDIRECT);
  url.searchParams.set('code_challenge', challenge);
  url.searchParams.set('code_challenge_method', 'S256');
  url.searchParams.set('state', clientState);
  url.searchParams.set('scope', DRIVE_SCOPE);
  const res = await fetch(url, { redirect: 'manual' });
  assert.equal(res.status, 302);
  const location = new URL(res.headers.get('location')!);
  assert.equal(location.origin, 'https://fake-google.example');
  const state = location.searchParams.get('state');
  assert.ok(state);
  return state!;
}

/** Complete the Google callback; returns the redirect back to the client. */
async function completeCallback(flow: Flow, googleState: string): Promise<URL> {
  const url = new URL(`${flow.baseUrl}/oauth/google/callback`);
  url.searchParams.set('state', googleState);
  url.searchParams.set('code', 'fake-google-code');
  const res = await fetch(url, { redirect: 'manual' });
  assert.equal(res.status, 302);
  await res.text();
  return new URL(res.headers.get('location')!);
}

async function exchangeCode(
  flow: Flow,
  code: string,
  verifier: string,
  redirectUri = CLIENT_REDIRECT,
): Promise<globalThis.Response> {
  return fetch(`${flow.baseUrl}/token`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body: new URLSearchParams({
      grant_type: 'authorization_code',
      client_id: flow.clientId,
      code,
      code_verifier: verifier,
      redirect_uri: redirectUri,
    }).toString(),
  });
}

describe('Team mode — OAuth 2.1 authorization server flow', () => {
  let httpServer: HttpServer;
  let sessions: Map<string, any>;
  let runtime: TeamRuntime;
  let idp: FakeIdp;
  let store: InMemoryTeamStore;
  let flow: Flow;

  before(async () => {
    const mod = await import('../../src/index.js');
    idp = new FakeIdp();
    store = new InMemoryTeamStore();
    runtime = await createTeamRuntime(makeConfig(), { store, idp });
    const created = mod.createHttpApp('127.0.0.1', { teamAuth: runtime });
    sessions = created.sessions;
    await new Promise<void>((resolve) => {
      httpServer = created.app.listen(0, '127.0.0.1', () => resolve());
    });
    const addr = httpServer.address();
    const baseUrl = `http://127.0.0.1:${(addr as { port: number }).port}`;
    flow = { baseUrl, clientId: await registerClient(baseUrl) };
  });

  after(async () => {
    runtime.stop();
    for (const [, session] of sessions) {
      await session.transport.close();
      await session.server.close();
    }
    sessions.clear();
    await new Promise<void>((resolve) => httpServer.close(() => resolve()));
  });

  it('serves AS and protected-resource metadata', async () => {
    const asRes = await fetch(`${flow.baseUrl}/.well-known/oauth-authorization-server`);
    assert.equal(asRes.status, 200);
    const as = await asRes.json();
    assert.equal(as.issuer, 'http://127.0.0.1:3100/');
    assert.ok(as.authorization_endpoint.endsWith('/authorize'));
    assert.ok(as.token_endpoint.endsWith('/token'));
    assert.ok(as.registration_endpoint.endsWith('/register'));
    assert.ok(as.revocation_endpoint.endsWith('/revoke'));
    assert.deepEqual(as.code_challenge_methods_supported, ['S256']);
    assert.deepEqual(as.scopes_supported, [DRIVE_SCOPE]);

    const prmRes = await fetch(`${flow.baseUrl}/.well-known/oauth-protected-resource`);
    assert.equal(prmRes.status, 200);
    const prm = await prmRes.json();
    assert.deepEqual(prm.authorization_servers, ['http://127.0.0.1:3100/']);
  });

  it('completes the full two-hop flow and issues working tokens', async () => {
    const { verifier, challenge } = pkcePair();
    const googleState = await startAuthorize(flow, challenge, 'state-xyz');

    const back = await completeCallback(flow, googleState);
    assert.equal(back.origin + back.pathname, CLIENT_REDIRECT);
    assert.equal(back.searchParams.get('state'), 'state-xyz', 'client state echoed');
    const code = back.searchParams.get('code');
    assert.ok(code!.startsWith('mcp_ac_'));

    const tokenRes = await exchangeCode(flow, code!, verifier);
    assert.equal(tokenRes.status, 200);
    const tokens = await tokenRes.json();
    assert.ok(tokens.access_token.startsWith('mcp_at_'));
    assert.ok(tokens.refresh_token.startsWith('mcp_rt_'));
    assert.equal(tokens.expires_in, 3600);

    // The user was persisted with the Google-granted identity + tokens.
    const user = await store.getUser('google-sub-alice');
    assert.equal(user?.email, 'alice@example.com');
    assert.equal(user?.googleRefreshToken, 'g-refresh-alice');

    // The bearer round-trips through the provider used by requireBearerAuth.
    const info = await runtime.provider.verifyAccessToken(tokens.access_token);
    assert.equal(info.extra?.sub, 'google-sub-alice');
    assert.ok(typeof info.expiresAt === 'number' && info.expiresAt < 1e11);

    // Refresh rotates.
    const refreshRes = await fetch(`${flow.baseUrl}/token`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body: new URLSearchParams({
        grant_type: 'refresh_token',
        client_id: flow.clientId,
        refresh_token: tokens.refresh_token,
      }).toString(),
    });
    assert.equal(refreshRes.status, 200);
    const rotated = await refreshRes.json();
    assert.notEqual(rotated.refresh_token, tokens.refresh_token);

    // Revoke the rotated access token → verification fails.
    const revokeRes = await fetch(`${flow.baseUrl}/revoke`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body: new URLSearchParams({
        client_id: flow.clientId,
        token: rotated.access_token,
      }).toString(),
    });
    assert.equal(revokeRes.status, 200);
    await assert.rejects(() => runtime.provider.verifyAccessToken(rotated.access_token));
  });

  it('rejects the exchange with a wrong PKCE verifier, then burns the code on reuse', async () => {
    const { challenge, verifier } = pkcePair();
    const googleState = await startAuthorize(flow, challenge);
    const code = (await completeCallback(flow, googleState)).searchParams.get('code')!;

    const bad = await exchangeCode(flow, code, 'wrong-verifier-wrong-verifier-wrong-verifier');
    assert.equal(bad.status, 400);
    const badBody = await bad.json();
    assert.equal(badBody.error, 'invalid_grant');

    // The failed PKCE check did not consume the code; the real verifier works…
    const good = await exchangeCode(flow, code, verifier);
    assert.equal(good.status, 200);
    await good.json();

    // …exactly once.
    const replay = await exchangeCode(flow, code, verifier);
    assert.equal(replay.status, 400);
  });

  it('re-validates redirect_uri at the token exchange', async () => {
    const { challenge, verifier } = pkcePair();
    const googleState = await startAuthorize(flow, challenge);
    const code = (await completeCallback(flow, googleState)).searchParams.get('code')!;

    const res = await exchangeCode(flow, code, verifier, 'https://attacker.example/cb');
    assert.equal(res.status, 400);
  });

  it('accepts a token exchange that omits redirect_uri (OAuth 2.1 client)', async () => {
    const { challenge, verifier } = pkcePair();
    const googleState = await startAuthorize(flow, challenge);
    const code = (await completeCallback(flow, googleState)).searchParams.get('code')!;

    // OAuth 2.1 clients drop redirect_uri from the token request (PKCE binds the
    // code). The SDK passes it through as undefined; the provider must not
    // reject a legal omission — previously this dead-ended sign-in.
    const res = await fetch(`${flow.baseUrl}/token`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body: new URLSearchParams({
        grant_type: 'authorization_code',
        client_id: flow.clientId,
        code,
        code_verifier: verifier,
      }).toString(),
    });
    assert.equal(res.status, 200);
    const tokens = await res.json();
    assert.ok(tokens.access_token.startsWith('mcp_at_'));
  });

  it('sends offline + forced-consent parameters to Google', async () => {
    // The consent URL is produced by GoogleIdpClient in production; here the
    // fake only carries state — assert the pending side instead: the flow
    // above proves state round-trips. This test pins the real builder.
    const { GoogleIdpClient } = await import('../../src/auth/team/googleIdp.js');
    const real = new GoogleIdpClient({
      clientId: 'cid',
      clientSecret: 'secret',
      redirectUri: 'https://server.example/oauth/google/callback',
      scopes: [DRIVE_SCOPE, 'openid'],
      hdHint: 'example.com',
    });
    const url = new URL(real.buildConsentUrl('the-state'));
    assert.equal(url.searchParams.get('access_type'), 'offline');
    assert.equal(url.searchParams.get('prompt'), 'consent select_account');
    assert.equal(url.searchParams.get('state'), 'the-state');
    assert.equal(url.searchParams.get('hd'), 'example.com');
    assert.ok(url.searchParams.get('scope')!.includes(DRIVE_SCOPE));
  });

  it('returns a 400 page (never a redirect) for unknown or replayed callback state', async () => {
    const bogus = await fetch(
      `${flow.baseUrl}/oauth/google/callback?state=bogus&code=fake-google-code`,
      { redirect: 'manual' },
    );
    assert.equal(bogus.status, 400);
    assert.equal(bogus.headers.get('location'), null);
    const page = await bogus.text();
    assert.ok(!page.includes('bogus'), 'callback inputs must not be reflected');

    // Replay: a consumed state behaves exactly like an unknown one.
    const { challenge } = pkcePair();
    const googleState = await startAuthorize(flow, challenge);
    await completeCallback(flow, googleState);
    const replay = await fetch(
      `${flow.baseUrl}/oauth/google/callback?state=${googleState}&code=fake-google-code`,
      { redirect: 'manual' },
    );
    assert.equal(replay.status, 400);
    assert.equal(replay.headers.get('location'), null);
  });

  it('propagates a user denial at Google as access_denied to the client', async () => {
    const { challenge } = pkcePair();
    const googleState = await startAuthorize(flow, challenge, 'denied-state');
    const res = await fetch(
      `${flow.baseUrl}/oauth/google/callback?state=${googleState}&error=access_denied`,
      { redirect: 'manual' },
    );
    assert.equal(res.status, 302);
    const back = new URL(res.headers.get('location')!);
    assert.equal(back.origin + back.pathname, CLIENT_REDIRECT);
    assert.equal(back.searchParams.get('error'), 'access_denied');
    assert.equal(back.searchParams.get('state'), 'denied-state');
    assert.equal(back.searchParams.get('code'), null);
  });
});

describe('Team mode — hosted-domain allowlist', () => {
  let httpServer: HttpServer;
  let runtime: TeamRuntime;
  let idp: FakeIdp;
  let store: InMemoryTeamStore;
  let flow: Flow;

  before(async () => {
    const mod = await import('../../src/index.js');
    idp = new FakeIdp();
    store = new InMemoryTeamStore();
    runtime = await createTeamRuntime(makeConfig({ allowedDomains: ['corp.example'] }), {
      store,
      idp,
    });
    const created = mod.createHttpApp('127.0.0.1', { teamAuth: runtime });
    await new Promise<void>((resolve) => {
      httpServer = created.app.listen(0, '127.0.0.1', () => resolve());
    });
    const addr = httpServer.address();
    const baseUrl = `http://127.0.0.1:${(addr as { port: number }).port}`;
    flow = { baseUrl, clientId: await registerClient(baseUrl) };
  });

  after(async () => {
    runtime.stop();
    await new Promise<void>((resolve) => httpServer.close(() => resolve()));
  });

  it('rejects a wrong-domain account, revokes the fresh grant, and persists nothing', async () => {
    idp.identity = { sub: 'sub-evil', email: 'mallory@evil.example', hd: 'evil.example' };
    const { challenge } = pkcePair();
    const googleState = await startAuthorize(flow, challenge, 'dom-state');
    const back = await completeCallback(flow, googleState);
    assert.equal(back.searchParams.get('error'), 'access_denied');
    assert.equal(back.searchParams.get('state'), 'dom-state');
    assert.equal(await store.getUser('sub-evil'), undefined);
    assert.deepEqual(idp.revoked, ['g-refresh-alice']);
  });

  it('rejects a consumer account (no hd claim) — the allowlist fails closed', async () => {
    idp.identity = { sub: 'sub-gmail', email: 'someone@gmail.com' } as FakeIdp['identity'];
    const { challenge } = pkcePair();
    const googleState = await startAuthorize(flow, challenge);
    const back = await completeCallback(flow, googleState);
    assert.equal(back.searchParams.get('error'), 'access_denied');
    assert.equal(await store.getUser('sub-gmail'), undefined);
  });

  it('accepts an allowed-domain account', async () => {
    idp.identity = { sub: 'sub-ok', email: 'bob@corp.example', hd: 'corp.example' };
    const { challenge, verifier } = pkcePair();
    const googleState = await startAuthorize(flow, challenge);
    const back = await completeCallback(flow, googleState);
    const code = back.searchParams.get('code');
    assert.ok(code);
    const res = await exchangeCode(flow, code!, verifier);
    assert.equal(res.status, 200);
    assert.equal((await store.getUser('sub-ok'))?.email, 'bob@corp.example');
  });

  it('preserves an existing member\'s stored grant when a re-auth is rejected by the allowlist', async () => {
    // 1) An allowed-domain member signs in and is persisted.
    idp.identity = { sub: 'sub-member', email: 'member@corp.example', hd: 'corp.example' };
    {
      const { challenge, verifier } = pkcePair();
      const googleState = await startAuthorize(flow, challenge);
      const code = (await completeCallback(flow, googleState)).searchParams.get('code')!;
      assert.equal((await exchangeCode(flow, code, verifier)).status, 200);
    }
    const before = await store.getUser('sub-member');
    assert.ok(before?.googleRefreshToken);
    const revokedBefore = idp.revoked.length;

    // 2) A later re-auth arrives WITHOUT the hd claim (a transient Google
    //    omission). The sign-in is rejected, but because a record already
    //    exists, the member's working grant must NOT be revoked at Google
    //    (which would destroy the shared refresh token) nor flagged for reauth.
    idp.identity = { sub: 'sub-member', email: 'member@corp.example' } as FakeIdp['identity'];
    const { challenge } = pkcePair();
    const googleState = await startAuthorize(flow, challenge, 'reauth-state');
    const back = await completeCallback(flow, googleState);
    assert.equal(back.searchParams.get('error'), 'access_denied');

    const after = await store.getUser('sub-member');
    assert.equal(after?.googleRefreshToken, before?.googleRefreshToken, 'stored grant intact');
    assert.notEqual(after?.needsReauth, true, 'existing member not flagged for reauth');
    assert.equal(idp.revoked.length, revokedBefore, 'existing member grant not revoked at Google');
  });
});
