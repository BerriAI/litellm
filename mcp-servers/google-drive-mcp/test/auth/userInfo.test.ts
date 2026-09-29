import assert from 'node:assert/strict';
import test from 'node:test';
import { fetchUserInfo } from '../../src/auth/userInfo.js';

// We only need the shape the helper calls: client.request({ url, method, signal }).
// A fake OAuth2Client-ish object is plenty — the helper never touches real OAuth.

function fakeClient(impl: (req: unknown) => { data: unknown }) {
  return {
    request(req: unknown) {
      return Promise.resolve(impl(req));
    },
  } as unknown as import('google-auth-library').OAuth2Client;
}

test('fetchUserInfo returns {sub, email} when both present', async () => {
  const client = fakeClient(() => ({ data: { sub: '12345', email: 'ann@example.com' } }));
  const info = await fetchUserInfo(client);
  assert.deepEqual(info, { sub: '12345', email: 'ann@example.com' });
});

test('fetchUserInfo throws when sub is missing', async () => {
  const client = fakeClient(() => ({ data: { email: 'ann@example.com' } }));
  await assert.rejects(() => fetchUserInfo(client), /missing sub or email/);
});

test('fetchUserInfo throws when email is missing', async () => {
  const client = fakeClient(() => ({ data: { sub: '12345' } }));
  await assert.rejects(() => fetchUserInfo(client), /missing sub or email/);
});

test('fetchUserInfo propagates network errors', async () => {
  const client = {
    request: () => Promise.reject(new Error('boom')),
  } as unknown as import('google-auth-library').OAuth2Client;
  await assert.rejects(() => fetchUserInfo(client), /boom/);
});
