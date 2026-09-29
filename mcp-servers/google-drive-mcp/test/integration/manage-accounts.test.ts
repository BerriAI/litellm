import assert from 'node:assert/strict';
import { describe, it, before, after } from 'node:test';
import { setupTestServer, callTool, type TestContext } from '../helpers/setup-server.js';
import { resolveSetDefaultTarget, awaitConsentCompletion } from '../../src/tools/drive.js';

// The test harness seeds an AccountStore in 'test' mode with a single synthetic
// account aliased 'test'. manage_accounts list should see it; add/remove/
// set_default should refuse because those require local-oauth mode.

describe('manage_accounts (test-mode harness)', () => {
  let ctx: TestContext;

  before(async () => {
    ctx = await setupTestServer();
  });

  after(async () => {
    await ctx.cleanup();
  });

  it("list returns the synthetic 'test' account", async () => {
    const result = await callTool(ctx.client, 'manage_accounts', { action: 'list' });
    assert.notEqual(result.isError, true, `list errored: ${JSON.stringify(result)}`);
    const text = result.content[0].text!;
    const payload = JSON.parse(text);
    assert.equal(payload.mode, 'test');
    assert.equal(payload.defaultAccount, 'test');
    assert.ok(Array.isArray(payload.accounts));
    assert.equal(payload.accounts.length, 1);
    assert.equal(payload.accounts[0].alias, 'test');
    assert.equal(payload.accounts[0].isDefault, true);
    // Must NOT leak tokens
    assert.equal((payload.accounts[0] as Record<string, unknown>).accessToken, undefined);
    assert.equal((payload.accounts[0] as Record<string, unknown>).refreshToken, undefined);
  });

  it('add refuses in non-local-oauth modes with an explanation', async () => {
    const result = await callTool(ctx.client, 'manage_accounts', {
      action: 'add',
      account_id: 'work',
    });
    assert.equal(result.isError, true);
    assert.match(
      result.content[0].text!,
      /only supported in local-OAuth mode.*test/,
    );
  });

  it('add rejects invalid aliases before attempting a flow', async () => {
    // Uppercase → invalid per ALIAS_PATTERN
    const bad = await callTool(ctx.client, 'manage_accounts', {
      action: 'add',
      account_id: 'Work',
    });
    assert.equal(bad.isError, true);
    // Note: mode-check runs first; to exercise the alias validator independently
    // we'd need to run in local-oauth mode. For now, any error is acceptable so
    // long as `add` doesn't silently succeed.
  });

  it('remove refuses in non-local-oauth modes', async () => {
    const result = await callTool(ctx.client, 'manage_accounts', {
      action: 'remove',
      account_id: 'test',
    });
    assert.equal(result.isError, true);
    assert.match(result.content[0].text!, /only supported in local-OAuth mode/);
  });

  it('set_default refuses in non-local-oauth modes', async () => {
    const result = await callTool(ctx.client, 'manage_accounts', {
      action: 'set_default',
      account_id: 'test',
    });
    assert.equal(result.isError, true);
    assert.match(result.content[0].text!, /only supported in local-OAuth mode/);
  });

  it('missing action returns a zod validation error', async () => {
    const result = await callTool(ctx.client, 'manage_accounts', {});
    assert.equal(result.isError, true);
  });

  it('unknown action is rejected by the enum validator', async () => {
    const result = await callTool(ctx.client, 'manage_accounts', { action: 'delete' });
    assert.equal(result.isError, true);
  });
});

describe('per-tool account parameter', () => {
  let ctx: TestContext;

  before(async () => {
    ctx = await setupTestServer();
  });

  after(async () => {
    await ctx.cleanup();
  });

  it("accepts account='test' and dispatches the tool", async () => {
    // `search` is a non-admin read tool; the harness's synthetic 'test' account
    // covers all default scopes so resolution succeeds.
    const result = await callTool(ctx.client, 'search', {
      query: 'anything',
      account: 'test',
    });
    // The mock google.drive returns {files:[]} by default.
    assert.notEqual(result.isError, true, `search errored: ${JSON.stringify(result)}`);
  });

  it('rejects unknown aliases with guidance', async () => {
    const result = await callTool(ctx.client, 'search', {
      query: 'anything',
      account: 'nope',
    });
    assert.equal(result.isError, true);
    assert.match(result.content[0].text!, /Unknown account/);
    assert.match(result.content[0].text!, /manage_accounts list/);
  });

  it('strips account from args before the handler sees them', async () => {
    ctx.mocks.drive.tracker.reset();
    const result = await callTool(ctx.client, 'search', {
      query: 'needle',
      account: 'test',
    });
    assert.notEqual(result.isError, true);
    const listCalls = ctx.mocks.drive.tracker.getCalls('files.list');
    assert.ok(listCalls.length > 0, 'expected at least one drive.files.list call');
    const firstArg = listCalls[0].args[0] ?? {};
    assert.equal(
      (firstArg as Record<string, unknown>).account,
      undefined,
      'account should have been stripped before reaching the Drive API',
    );
  });

  it('admin tool (authGetStatus) does not accept the account param in its schema', async () => {
    const { tools } = await ctx.client.listTools();
    const authStatus = tools.find((t) => t.name === 'authGetStatus');
    assert.ok(authStatus);
    assert.equal(
      (authStatus.inputSchema as { properties?: Record<string, unknown> })?.properties?.account,
      undefined,
    );
  });

  it('manage_accounts itself does not accept the account param', async () => {
    const { tools } = await ctx.client.listTools();
    const mgr = tools.find((t) => t.name === 'manage_accounts');
    assert.ok(mgr);
    assert.equal(
      (mgr.inputSchema as { properties?: Record<string, unknown> })?.properties?.account,
      undefined,
    );
  });

  it('a representative non-admin tool has the account param injected', async () => {
    const { tools } = await ctx.client.listTools();
    const search = tools.find((t) => t.name === 'search');
    assert.ok(search);
    const props = (search.inputSchema as { properties?: Record<string, unknown> })?.properties ?? {};
    assert.ok(props.account, "'search' should carry the injected account param");
    const accountProp = props.account as { type?: string; description?: string };
    assert.equal(accountProp.type, 'string');
    assert.match(accountProp.description ?? '', /manage_accounts list/);
  });
});

// Regression (finding 3): after the last account is removed, the admin context
// must not throw — manage_accounts still needs to work so the user can re-add one.
describe('manage_accounts with zero accounts', () => {
  let ctx: TestContext;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  let mod: any;

  before(async () => {
    ctx = await setupTestServer();
    mod = await import('../../src/index.js');
    // Empty the store the admin context would otherwise resolve a default from.
    const sys = mod._getAuthSystemForTesting();
    await sys.store.remove('test');
  });

  after(async () => {
    // Restore the standard synthetic account so later tests are unaffected.
    mod._setAuthClientForTesting({ request: async () => ({ data: 'restored' }) });
    await ctx.cleanup();
  });

  it('list returns an empty list instead of a lockout error', async () => {
    const result = await callTool(ctx.client, 'manage_accounts', { action: 'list' });
    assert.notEqual(result.isError, true, `list errored: ${JSON.stringify(result)}`);
    const payload = JSON.parse(result.content[0].text!);
    assert.equal(payload.defaultAccount, null);
    assert.deepEqual(payload.accounts, []);
  });
});

// Finding 13: `set_default` must treat the literal alias "null" as a real account and
// only clear the default on JSON null / undefined / empty. resolveSetDefaultTarget is the seam.
describe('resolveSetDefaultTarget (finding 13)', () => {
  it('treats the literal string "null" as a real alias, not the clear sentinel', () => {
    assert.equal(resolveSetDefaultTarget('null'), 'null');
  });

  it('clears the default for JSON null, undefined, or empty string', () => {
    assert.equal(resolveSetDefaultTarget(null), null);
    assert.equal(resolveSetDefaultTarget(undefined), null);
    assert.equal(resolveSetDefaultTarget(''), null);
  });

  it('passes an ordinary alias through', () => {
    assert.equal(resolveSetDefaultTarget('work'), 'work');
  });
});

// Finding 13 (schema): the tool documents passing JSON null to `set_default` to clear the
// default, so the zod schema must accept null. In the test-mode harness set_default is
// refused for being outside local-oauth mode — which proves null cleared validation and
// reached the handler (before the fix it failed zod with "Expected string, received null").
describe('manage_accounts set_default accepts JSON null (finding 13)', () => {
  let ctx: TestContext;

  before(async () => {
    ctx = await setupTestServer();
  });

  after(async () => {
    await ctx.cleanup();
  });

  it('null account_id reaches the mode guard, not a zod type error', async () => {
    const result = await callTool(ctx.client, 'manage_accounts', {
      action: 'set_default',
      account_id: null,
    });
    assert.equal(result.isError, true);
    assert.match(result.content[0].text!, /only supported in local-OAuth mode/);
    assert.doesNotMatch(result.content[0].text!, /Expected string|Invalid|received null/);
  });
});

// Finding 12: an abandoned `add` must tear down the embedded OAuth server on timeout so
// its callback port isn't leaked. awaitConsentCompletion is the seam.
describe('awaitConsentCompletion (finding 12)', () => {
  it('cancels the auth server exactly once on timeout', async () => {
    let cancelled = 0;
    const never = new Promise<{ alias: string }>(() => {});
    await assert.rejects(
      awaitConsentCompletion({ completion: never, cancel: async () => { cancelled += 1; } }, 10),
      /Timed out .* waiting for OAuth consent/,
    );
    assert.equal(cancelled, 1, 'cancel() must run once on timeout');
  });

  it('returns the record and does not cancel on success', async () => {
    let cancelled = 0;
    const rec = { alias: 'work' };
    const got = await awaitConsentCompletion(
      { completion: Promise.resolve(rec), cancel: async () => { cancelled += 1; } },
      60_000,
    );
    assert.equal(got, rec);
    assert.equal(cancelled, 0, 'success path leaves teardown to addAccountFlow');
  });

  it('cancels once and rethrows when completion rejects', async () => {
    let cancelled = 0;
    await assert.rejects(
      awaitConsentCompletion(
        { completion: Promise.reject(new Error('boom')), cancel: async () => { cancelled += 1; } },
        60_000,
      ),
      /boom/,
    );
    assert.equal(cancelled, 1);
  });
});
