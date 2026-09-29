// ---------------------------------------------------------------------------
// Fake Google token endpoint for refresh tests.
//
// Installs a gaxios `adapter` on an OAuth2Client's transporter so the REAL
// `refreshAccessToken()` path runs end to end — request options, request
// interceptors (where the #169 abort signal is attached), gaxios' own retry
// policy, google-auth-library's response handling and 'tokens' event — with
// the network replaced by a per-call handler. Works on google-auth-library 9
// (transporter is a DefaultTransporter whose `defaults` setter forwards to its
// Gaxios) and 10 (transporter is the Gaxios itself); the response shape below
// is the minimum both accept.
// ---------------------------------------------------------------------------

import type { OAuth2Client } from 'google-auth-library';

export interface TokenEndpointCall {
  index: number;
  url: string;
  method?: string;
  /** The AbortSignal gaxios handed to the transport, if any. */
  signal?: AbortSignal;
}

export type TokenEndpointHandler = (
  call: TokenEndpointCall,
) => Promise<{ status?: number; data: unknown }>;

export interface FakeTokenEndpoint {
  calls: TokenEndpointCall[];
  /** Swap at any time; the next request uses the new handler. */
  handler: TokenEndpointHandler;
}

/** Route every request the client's transporter makes through `handler`. */
export function installFakeTokenEndpoint(
  client: OAuth2Client,
  handler: TokenEndpointHandler,
): FakeTokenEndpoint {
  const endpoint: FakeTokenEndpoint = { calls: [], handler };
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const adapter = async (opts: any) => {
    const call: TokenEndpointCall = {
      index: endpoint.calls.length,
      url: String(opts.url),
      method: opts.method,
      signal: opts.signal,
    };
    endpoint.calls.push(call);
    const { status = 200, data } = await endpoint.handler(call);
    return {
      config: opts,
      data,
      headers: new Headers(),
      status,
      statusText: status === 200 ? 'OK' : 'Error',
      request: { responseURL: call.url },
    };
  };
  const transporter = client.transporter as unknown as { defaults?: Record<string, unknown> };
  transporter.defaults = { ...(transporter.defaults ?? {}), adapter };
  return endpoint;
}

/** A successful refresh response body. */
export function tokenResponse(accessToken: string): TokenEndpointHandler {
  return async () => ({
    data: { access_token: accessToken, expires_in: 3600, token_type: 'Bearer' },
  });
}

/** Google's answer to a revoked/expired refresh token. */
export const invalidGrantResponse: TokenEndpointHandler = async () => ({
  status: 400,
  data: { error: 'invalid_grant', error_description: 'Token has been expired or revoked.' },
});

/**
 * Never answer. Rejects only when the transport's signal aborts (as node-fetch
 * / undici do), with an AbortError so neither gaxios generation retries it; a
 * request that carries no signal hangs forever — exactly the pre-#169 failure.
 */
export const stallUntilAborted: TokenEndpointHandler = (call) =>
  new Promise((_, reject) => {
    const { signal } = call;
    if (!signal) return;
    const fail = () =>
      reject(
        signal.reason ?? Object.assign(new Error('The user aborted a request.'), { name: 'AbortError' }),
      );
    if (signal.aborted) fail();
    else signal.addEventListener('abort', fail, { once: true });
  });

/** Stall the first `stalls` requests, then delegate to `then`. */
export function stallThen(then: TokenEndpointHandler, stalls = 1): TokenEndpointHandler {
  return (call) => (call.index < stalls ? stallUntilAborted(call) : then(call));
}
