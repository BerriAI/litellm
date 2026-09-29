import assert from 'node:assert/strict';
import test from 'node:test';

import { getEffectiveIdentity } from '../../src/auth/identity.js';

test('getEffectiveIdentity returns an error identity (never throws) when the drive client is missing', async () => {
  // The client-less admin context (zero accounts, or a default whose client
  // could not be built) must yield a diagnosable identity, not a crash (#137).
  const identity = await getEffectiveIdentity(undefined);
  assert.equal(identity.emailAddress, null);
  assert.ok(identity.error && /No authenticated Drive client/.test(identity.error));
});

test('getEffectiveIdentity resolves the acting account from about.get', async () => {
  const fakeDrive: any = {
    about: {
      get: async () => ({
        data: {
          user: { displayName: 'Ada L', emailAddress: 'ada@example.com' },
          storageQuota: { limit: '100', usage: '10' },
        },
      }),
    },
  };
  const identity = await getEffectiveIdentity(fakeDrive);
  assert.equal(identity.emailAddress, 'ada@example.com');
  assert.equal(identity.displayName, 'Ada L');
  assert.equal(identity.storageLimit, '100');
  assert.equal(identity.storageUsage, '10');
  assert.equal(identity.error, undefined);
});

test('getEffectiveIdentity returns the underlying error when about.get rejects', async () => {
  const fakeDrive: any = {
    about: { get: async () => { throw new Error('Insufficient Permission'); } },
  };
  const identity = await getEffectiveIdentity(fakeDrive);
  assert.equal(identity.emailAddress, null);
  assert.equal(identity.error, 'Insufficient Permission');
});

test('getEffectiveIdentity times out instead of hanging when about.get never resolves', async () => {
  // Mirrors gaxios: a request that respects the abort signal rejects when the
  // timeout fires, so the diagnostic returns promptly on an unreachable network.
  const fakeDrive: any = {
    about: {
      get: (_params: unknown, opts: { signal: AbortSignal }) =>
        new Promise((_resolve, reject) => {
          opts.signal.addEventListener('abort', () => reject(new Error('aborted')));
        }),
    },
  };
  const identity = await getEffectiveIdentity(fakeDrive, 20);
  assert.equal(identity.emailAddress, null);
  assert.ok(identity.error && /timed out after 20ms/.test(identity.error));
});
