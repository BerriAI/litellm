import { test } from 'node:test';
import assert from 'node:assert/strict';
import { register } from './hooks/register.js';

function harness(options = { gatewayUrl: 'https://gateway.example' }, key = 'test-gateway-key') {
  let handler;
  let failure;
  const on = (name, fn) => {
    assert.equal(name, 'http.fetch');
    handler = fn;
    return { catch(fn) { failure = fn; } };
  };
  register(on, options);
  const calls = [];
  let keyReads = 0;
  const response = { value: { status: 200, isStreaming: true, text: '', headers: {} } };
  const next = Object.assign(async event => { calls.push(event); return response; },
    { origin: { plugin: 'engine', tier: 'core' } });
  const api = { env: { get: async () => { keyReads++; return key; } } };
  return { calls, next, response, failure: () => failure(),
    run: async event => {
      try { return await handler(api, event, next); }
      catch { return failure(); }
    }, keyReads: () => keyReads };
}

const event = (url = 'https://api.anthropic.com/v1/messages') => ({ url, init: {
  method: 'POST', headers: { Authorization: 'native-secret', 'X-Api-Key': 'native-key',
    Cookie: 'session=secret', 'X-LiteLLM-Api-Key': 'untrusted', 'anthropic-beta': 'feature',
    'content-type': 'application/json' },
  body: '{"model":"native-model","messages":[],"tools":[],"thinking":{"type":"adaptive"}}',
  compress: 'gzip', auth: 'opaque-handle', socketPath: '/tmp/old.sock',
} });

test('routes messages and token counts while preserving body, query and streaming response', async () => {
  for (const path of ['/v1/messages', '/v1/messages/count_tokens']) {
    const h = harness();
    const input = event('https://api.anthropic.com' + path + '?beta=true');
    assert.equal(await h.run(input), h.response);
    assert.equal(h.calls[0].url, 'https://gateway.example' + path + '?beta=true');
    assert.equal(h.calls[0].init.body, input.init.body);
    assert.equal(h.calls[0].init.compress, input.init.compress);
    assert.deepEqual(h.calls[0].init.headers, { 'anthropic-beta': 'feature',
      'content-type': 'application/json', 'x-litellm-api-key': 'Bearer test-gateway-key' });
    assert.equal(h.calls[0].init.auth, undefined);
    assert.equal(h.calls[0].init.socketPath, undefined);
    assert.equal(input.init.headers.Authorization, 'native-secret');
  }
});

test('native services retain the exact event and never read gateway credentials', async () => {
  const h = harness({}, undefined);
  for (const url of ['https://api.anthropic.com/api/oauth/usage',
    'https://api.anthropic.com/v1/sessions/example', 'https://claude.ai/api/organizations/example',
    'https://api.anthropic.com/api/future-feature']) {
    const input = event(url);
    assert.equal(await h.run(input), h.response);
    assert.equal(h.calls.at(-1), input);
  }
  assert.equal(h.keyReads(), 0);
});

test('known inference from other mods also routes while their native service calls remain unchanged', async () => {
  for (const origin of [{ plugin: 'other', tier: 'user' }, { plugin: 'engine', tier: 'user' }]) {
    const h = harness(); h.next.origin = origin;
    const input = event(); await h.run(input);
    assert.equal(h.calls[0].url, 'https://gateway.example/v1/messages');
    assert.equal(h.calls[0].init.redirect, 'manual');
    const service = event('https://api.anthropic.com/api/oauth/usage');
    await h.run(service); assert.equal(h.calls[1], service); assert.equal(h.keyReads(), 1);
  }
});

test('missing or unsafe gateway credentials deny inference without forwarding', async () => {
  for (const key of ['', 'sk-ant-oat01-test-only', 'token\r\ninjected: value', 'token value']) {
    const h = harness(undefined, key);
    assert.ok((await h.run(event())).deny); assert.equal(h.calls.length, 0);
  }
});

test('unsafe gateway configuration denies inference without forwarding', async () => {
  for (const gatewayUrl of [undefined, 'invalid', 'http://gateway.example',
    'https://api.anthropic.com', 'https://user:secret@gateway.example',
    'https://gateway.example?key=secret', 'https://gateway.example/path//ambiguous', 'https://gateway.example#fragment']) {
    const h = harness({ gatewayUrl });
    assert.ok((await h.run(event())).deny); assert.equal(h.calls.length, 0);
  }
});

test('alternate inference origins, unsupported inference routes and wrong methods are denied', async () => {
  for (const url of ['https://other.example/v1/messages', 'https://api.anthropic.com/v1/messages/batches',
    'https://api.anthropic.com/v1/responses', 'https://api.anthropic.com/v1/chat/completions']) {
    const h = harness(); assert.ok((await h.run(event(url))).deny); assert.equal(h.calls.length, 0);
  }
  const h = harness(); const input = event(); input.init.method = 'GET';
  assert.ok((await h.run(input)).deny); assert.equal(h.calls.length, 0);
});

test('encoded supported inference paths route through the gateway', async () => {
  const h = harness(); await h.run(event('https://api.anthropic.com/v1/%6dessages/'));
  assert.equal(h.calls[0].url, 'https://gateway.example/v1/messages');
});

test('gateway authentication and policy denial do not trigger native login recovery', async () => {
  for (const status of [401, 403]) {
    const h = harness(); h.response.value.status = status;
    const result = await h.run(event());
    assert.match(result.deny ?? '', /LiteLLM gateway/);
    assert.match(result.deny, /native Claude login/i);
    assert.equal(result.value, undefined);
    assert.equal(h.calls.length, 1);
  }
});

test('gateway redirects are refused and the plugin does not issue another request', async () => {
  const h = harness(); h.response.value.status = 307;
  assert.ok((await h.run(event())).deny); assert.equal(h.calls.length, 1);
});

test('runtime errors deny rather than replaying the native request', async () => {
  const h = harness(); assert.ok((await h.run(event('not-a-url'))).deny);
  assert.equal(h.calls.length, 0); assert.ok((await h.failure()).deny);
});


test('an existing gateway behind a path prefix retains that prefix', async () => {
  const h = harness({ gatewayUrl: 'https://gateway.example/litellm/' });
  await h.run(event('https://api.anthropic.com/v1/messages?beta=true'));
  assert.equal(h.calls[0].url, 'https://gateway.example/litellm/v1/messages?beta=true');
});

test('environment credential rotation is observed before the next inference', async () => {
  let handler;
  const on = (name, fn) => { handler = fn; return { catch() {} }; };
  register(on, {gatewayUrl: 'https://gateway.example'});
  let key = 'first-gateway-credential';
  const sent = [];
  const next = Object.assign(async e => { sent.push(e); return {value:{status:200}}; },
    {origin:{plugin:'engine',tier:'core'}});
  const api = {env:{get:async () => key}};
  await handler(api, event(), next);
  key = 'second-gateway-credential';
  await handler(api, event(), next);
  assert.equal(sent[0].init.headers['x-litellm-api-key'], 'Bearer first-gateway-credential');
  assert.equal(sent[1].init.headers['x-litellm-api-key'], 'Bearer second-gateway-credential');
});


test('another mod using a messages path on an unrelated service is not treated as inference', async () => {
  const h = harness(); h.next.origin = {plugin:'native-integration',tier:'builtin'};
  const input = event('https://service.example/v1/messages');
  assert.equal(await h.run(input), h.response);
  assert.equal(h.calls[0], input);
  assert.equal(h.keyReads(), 0);
});
