/**
 * Bootstraps the MCP server with mocked Google APIs for integration testing.
 *
 * Strategy:
 * 1. Import `google` from `googleapis` and monkey-patch service factory methods.
 * 2. Dynamically import the server module — it sees our patched `google` singleton.
 * 3. Call `_setAuthClientForTesting({})` to bypass authentication.
 * 4. Create an InMemoryTransport pair and connect a Client to the Server.
 */

import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { InMemoryTransport } from '@modelcontextprotocol/sdk/inMemory.js';
import { google } from 'googleapis';
import { createAllMocks, type AllMocks } from './mock-google-apis.js';

export interface TestContext {
  client: Client;
  mocks: AllMocks;
  cleanup: () => Promise<void>;
  /** Override the injected auth client's `request` (used by tools that fetch
   *  arbitrary URLs, e.g. getGoogleDocImage). Restore with `resetAuthRequest`. */
  setAuthRequest: (fn: (opts: any) => Promise<any>) => void;
  resetAuthRequest: () => void;
}

const DEFAULT_AUTH_REQUEST = async () => ({ data: 'mock-auth-request-response' });

// We cache the server module so it's only imported once across all test files.
let _serverModule: any = null;

export async function setupTestServer(): Promise<TestContext> {
  const mocks = createAllMocks();

  // Patch the googleapis singleton
  (google as any).drive = mocks.google.drive;
  (google as any).docs = mocks.google.docs;
  (google as any).sheets = mocks.google.sheets;
  (google as any).slides = mocks.google.slides;
  (google as any).calendar = mocks.google.calendar;

  // Import the server module (only once)
  if (!_serverModule) {
    _serverModule = await import('../../src/index.js');
  }

  // Inject fake auth client to bypass authenticate(). The object is mutable and
  // held by reference inside the server, so swapping `.request` here is visible
  // to `ctx.authClient.request` in tool handlers.
  const authClientMock: { request: (opts: any) => Promise<any> } = {
    request: DEFAULT_AUTH_REQUEST,
  };
  _serverModule._setAuthClientForTesting(authClientMock);

  const server = _serverModule.server;

  // Create transport pair
  const [clientTransport, serverTransport] = InMemoryTransport.createLinkedPair();

  // Connect server to its transport
  await server.connect(serverTransport);

  // Create and connect client
  const client = new Client({ name: 'test-client', version: '1.0.0' });
  await client.connect(clientTransport);

  const cleanup = async () => {
    await client.close();
    // Note: We don't close the server itself — it's reused across tests.
    // The serverTransport closing is enough to reset the connection.
  };

  const setAuthRequest = (fn: (opts: any) => Promise<any>) => { authClientMock.request = fn; };
  const resetAuthRequest = () => { authClientMock.request = DEFAULT_AUTH_REQUEST; };

  return { client, mocks, cleanup, setAuthRequest, resetAuthRequest };
}

/**
 * Helper to call a tool and return its result.
 */
export async function callTool(
  client: Client,
  name: string,
  args: Record<string, unknown> = {},
): Promise<{ content: Array<{ type: string; text?: string; data?: string; mimeType?: string }>; isError?: boolean }> {
  const result = await client.callTool({ name, arguments: args });
  return result as any;
}
