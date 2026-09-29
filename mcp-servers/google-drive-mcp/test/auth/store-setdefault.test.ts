import assert from 'node:assert/strict';
import test from 'node:test';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';

import { AccountStore } from '../../src/auth/accountStore.js';

// Guards that `setDefault` is applied both to disk and to the in-memory view a
// same-process read observes, when awaited sequentially. (The `manage_accounts
// set_default` handler awaits this write before responding, so a request/response
// client always observes the new default on its next call. Reads are synchronous
// and bypass the write queue, so a caller that fires a concurrent read WITHOUT
// awaiting the set_default response may transiently see the old value — that is a
// client ordering concern, not a store defect. This test pins the sequential
// contract the store must uphold.)
test('setDefault updates both disk and in-memory getDefault when awaited', async () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'gdmcp-setdefault-'));
  const fp = path.join(dir, 'tokens.json');
  const rec = (alias: string, sub: string) => ({
    alias, email: `${alias}@x`, sub,
    accessToken: 'x', refreshToken: 'y', scope: '', tokenType: 'Bearer',
    expiryDate: 9999999999999, addedAt: '2026-01-01T00:00:00.000Z', pendingIdentity: false,
  });
  fs.writeFileSync(fp, JSON.stringify({
    version: 2, defaultAccount: 'default',
    accounts: { default: rec('default', '1'), work: rec('work', '2') },
  }));
  try {
    const store = new AccountStore({ filePath: fp, mode: 'local-oauth' });
    await store.reload();
    assert.equal(store.getDefault(), 'default', 'precondition: default is the default');

    await store.setDefault('work');
    assert.equal(store.getDefault(), 'work', 'in-memory getDefault reflects the new default');
    assert.equal(
      JSON.parse(fs.readFileSync(fp, 'utf-8')).defaultAccount,
      'work',
      'the new default is persisted to disk',
    );

    await store.setDefault(null);
    assert.equal(store.getDefault(), undefined, 'clearing the default is reflected in memory');
    assert.equal(
      JSON.parse(fs.readFileSync(fp, 'utf-8')).defaultAccount,
      undefined,
      'cleared default is persisted (field removed)',
    );
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});
