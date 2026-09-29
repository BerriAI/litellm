/**
 * Asserts the *actual* server wiring honours the resources opt-out — not just
 * that loadRuntimeConfig() parses the flag (covered in cli-args-config.test.ts).
 *
 * Regression guard: a future refactor that unconditionally advertises
 * `resources: {}` or unconditionally calls registerResourceHandlers() would
 * leave the config tests green while breaking the opt-out. These tests build a
 * real Server via the injectable createMcpServer(config), drive it with a real
 * MCP Client over InMemoryTransport, and check the advertised capability and
 * method dispatch.
 */
import assert from 'node:assert/strict';
import { describe, it, before } from 'node:test';
import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { InMemoryTransport } from '@modelcontextprotocol/sdk/inMemory.js';
import { ErrorCode } from '@modelcontextprotocol/sdk/types.js';
import { google } from 'googleapis';
import { createAllMocks } from '../helpers/mock-google-apis.js';

const BASE_CONFIG = {
  apiTimeout: 120_000,
  tokenRefreshTimeout: 15_000,
  retryMax: 3,
  retryBaseDelay: 1_000,
  disableResources: false,
};

let serverModule: any;

async function connectClient(disableResources: boolean): Promise<Client> {
  const server = serverModule.createMcpServer({ ...BASE_CONFIG, disableResources });
  const [clientTransport, serverTransport] = InMemoryTransport.createLinkedPair();
  await server.connect(serverTransport);
  const client = new Client({ name: 'gating-test-client', version: '1.0.0' });
  await client.connect(clientTransport);
  return client;
}

describe('resources capability gating (server wiring)', () => {
  before(async () => {
    const mocks = createAllMocks();
    (google as any).drive = mocks.google.drive;
    (google as any).docs = mocks.google.docs;
    (google as any).sheets = mocks.google.sheets;
    (google as any).slides = mocks.google.slides;
    (google as any).calendar = mocks.google.calendar;

    serverModule = await import('../../src/index.js');
    serverModule._setAuthClientForTesting({
      request: async () => ({ data: {} }),
    });
  });

  it('advertises resources and serves resources/list when enabled', async () => {
    const client = await connectClient(false);
    try {
      const caps = client.getServerCapabilities();
      assert.ok(caps?.resources, 'resources capability should be advertised');
      assert.ok(caps?.tools, 'tools capability should always be advertised');

      // Handler is registered: this resolves (mock files.list -> empty) rather
      // than failing with MethodNotFound.
      const res = await client.listResources();
      assert.ok(Array.isArray(res.resources), 'resources/list should return a list');
    } finally {
      await client.close();
    }
  });

  it('omits resources and rejects resources/list when disabled', async () => {
    const client = await connectClient(true);
    try {
      const caps = client.getServerCapabilities();
      assert.equal(caps?.resources, undefined, 'resources capability must be omitted');
      assert.ok(caps?.tools, 'tools capability must still be advertised');

      await assert.rejects(
        () => client.listResources(),
        (err: any) => err?.code === ErrorCode.MethodNotFound,
        'resources/list must be unhandled when the capability is disabled',
      );
    } finally {
      await client.close();
    }
  });

  it('keeps tools available regardless of the resources opt-out', async () => {
    const client = await connectClient(true);
    try {
      const { tools } = await client.listTools();
      assert.ok(tools.length > 0, 'tools must remain registered when resources are disabled');
    } finally {
      await client.close();
    }
  });
});
