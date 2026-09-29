// ---------------------------------------------------------------------------
// GET /oauth/google/callback — the return leg of the Google hop.
//
// Consumes the pending authorization (single-use, keyed by our state),
// exchanges the Google code, resolves who consented, enforces the
// hosted-domain allowlist, persists the user, mints our authorization code,
// and finally redirects back to the MCP client's (already SDK-validated)
// redirect URI with the client's original state.
//
// Redirect discipline: before the pending record is validated there is no
// trusted redirect target, so every early failure renders a direct 4xx HTML
// page — never a Location header (open-redirect prevention). Nothing from the
// query string is ever reflected into a page or a log line (the query carries
// the Google authorization code).
// ---------------------------------------------------------------------------

import type { Request, RequestHandler, Response } from 'express';
import { splitScopes } from '../scopes.js';
import { describeErrorForLog } from '../utils.js';
import type { TeamConfig } from './config.js';
import type { GoogleIdp } from './googleIdp.js';
import { mintAuthorizationCode, sha256Hex } from './tokens.js';
import {
  AUTH_CODE_TTL_MS,
  TeamStore,
  TeamStoreCapacityError,
  TeamUserRecord,
} from './types.js';

export interface GoogleCallbackDeps {
  store: TeamStore;
  idp: GoogleIdp;
  config: TeamConfig;
  /** Notified after a user (re-)authorizes, so cached per-user Google clients
   * can be evicted and rebuilt from the fresh grant. */
  onUserAuthorized?: (sub: string) => void;
  /** Redaction rule: only pass identity fields (sub/email/domain), never
   * tokens, codes, or states. */
  log?: (message: string) => void;
}

export function makeGoogleCallbackHandler(deps: GoogleCallbackDeps): RequestHandler {
  const log = deps.log ?? ((message: string) => console.error(`[team-auth] ${message}`));

  return async (req: Request, res: Response) => {
    const state = typeof req.query.state === 'string' ? req.query.state : undefined;
    const code = typeof req.query.code === 'string' ? req.query.code : undefined;
    const googleError = typeof req.query.error === 'string' ? req.query.error : undefined;

    if (!state) {
      renderErrorPage(res);
      return;
    }
    const pending = await deps.store.consumePendingAuthorization(state);
    if (!pending) {
      // Unknown, expired, or replayed state: no trusted redirect target exists.
      renderErrorPage(res);
      return;
    }

    if (googleError) {
      // The user declined at Google (or Google reported a flow error). The
      // specific error value is untrusted input — collapse it to the standard
      // denial code rather than forwarding it.
      redirectToClient(res, pending.redirectUri, {
        error: 'access_denied',
        state: pending.clientState,
      });
      return;
    }
    if (!code) {
      redirectToClient(res, pending.redirectUri, {
        error: 'invalid_request',
        error_description: 'Missing authorization code.',
        state: pending.clientState,
      });
      return;
    }

    let tokens;
    let identity;
    try {
      ({ tokens, identity } = await deps.idp.exchangeCode(code));
    } catch (err) {
      log(`Google code exchange failed: ${describeErrorForLog(err)}`);
      redirectToClient(res, pending.redirectUri, {
        error: 'server_error',
        error_description: 'Could not complete the Google sign-in.',
        state: pending.clientState,
      });
      return;
    }

    // Hosted-domain allowlist: enforce on the hd claim, which Google asserts,
    // not on the email suffix, which the user controls. Consumer accounts have
    // no hd — absence fails closed. A rejected sign-in must not leave a live
    // grant behind.
    const existing = await deps.store.getUser(identity.sub);
    if (deps.config.allowedDomains.length > 0) {
      const hd = identity.hd?.toLowerCase();
      if (!hd || !deps.config.allowedDomains.includes(hd)) {
        log(`Rejected sign-in from outside the allowed domains: ${identity.email} (sub: ${identity.sub})`);
        // Revoke the freshly-issued grant only for a brand-new sub. For an
        // already-stored member, revoking at Google removes their whole
        // user+client grant — destroying the working refresh token we already
        // hold — over what may be a transient `hd` omission. Drop the fresh
        // token instead and leave the stored grant intact (removed-domain
        // members are deprovisioned by an explicit admin action, not here).
        if (!existing && (tokens.refresh_token || tokens.access_token)) {
          try {
            await deps.idp.revokeGrant(tokens.refresh_token ?? tokens.access_token!);
          } catch (err) {
            log(`Failed to revoke rejected sign-in grant (sub: ${identity.sub}): ${describeErrorForLog(err)}`);
          }
        }
        redirectToClient(res, pending.redirectUri, {
          error: 'access_denied',
          error_description: 'This Google account is not part of the allowed domain.',
          state: pending.clientState,
        });
        return;
      }
    }

    const refreshToken = tokens.refresh_token ?? existing?.googleRefreshToken;
    if (!refreshToken) {
      // prompt=consent + access_type=offline should always yield one; without
      // it the user would break within the hour.
      log(`Google returned no refresh token for ${identity.email} (sub: ${identity.sub}); aborting sign-in.`);
      redirectToClient(res, pending.redirectUri, {
        error: 'server_error',
        error_description: 'Google did not issue offline credentials. Try again.',
        state: pending.clientState,
      });
      return;
    }

    const nowIso = new Date().toISOString();
    const user: TeamUserRecord = {
      sub: identity.sub,
      email: identity.email,
      googleRefreshToken: refreshToken,
      googleAccessToken: tokens.access_token ?? undefined,
      googleTokenExpiry: tokens.expiry_date ?? undefined,
      grantedScopes:
        splitScopes(tokens.scope).length > 0
          ? splitScopes(tokens.scope)
          : [...deps.config.googleScopes],
      createdAt: existing?.createdAt ?? nowIso,
      updatedAt: nowIso,
      // needsReauth deliberately omitted: a successful consent clears the flag.
    };
    try {
      await deps.store.upsertUser(user);
    } catch (err) {
      if (err instanceof TeamStoreCapacityError) {
        log(`Rejected sign-in for ${identity.email} (sub: ${identity.sub}): ${err.message}`);
        await deps.idp.revokeGrant(tokens.refresh_token ?? tokens.access_token ?? '');
        renderTeamFullPage(res);
        return;
      }
      throw err;
    }
    deps.onUserAuthorized?.(identity.sub);
    log(`Authorized team member ${identity.email} (sub: ${identity.sub})`);

    const authorizationCode = mintAuthorizationCode();
    const now = Date.now();
    await deps.store.saveAuthorizationCode({
      codeHash: sha256Hex(authorizationCode),
      clientId: pending.clientId,
      sub: identity.sub,
      codeChallenge: pending.codeChallenge,
      redirectUri: pending.redirectUri,
      scopes: pending.scopes,
      resource: pending.resource,
      createdAt: now,
      expiresAt: now + AUTH_CODE_TTL_MS,
      challengeLookups: 0,
    });
    redirectToClient(res, pending.redirectUri, {
      code: authorizationCode,
      state: pending.clientState,
    });
  };
}

function redirectToClient(
  res: Response,
  redirectUri: string,
  params: Record<string, string | undefined>,
): void {
  // The redirect target is the SDK-validated URI stored at /authorize — never
  // anything derived from the callback request.
  const url = new URL(redirectUri);
  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined) url.searchParams.set(key, value);
  }
  res.redirect(302, url.href);
}

function renderErrorPage(res: Response): void {
  sendPage(
    res,
    400,
    'Sign-in link invalid or expired',
    'This sign-in link is invalid, has expired, or was already used. ' +
      'Go back to your MCP client and start the connection again.',
  );
}

function renderTeamFullPage(res: Response): void {
  sendPage(
    res,
    403,
    'Team is full',
    'This server has reached its maximum number of team members. ' +
      'Ask the server administrator to remove unused accounts or raise the limit.',
  );
}

/** Static content only — callback inputs are never reflected into the page. */
function sendPage(res: Response, status: number, title: string, message: string): void {
  res
    .status(status)
    .type('html')
    .send(
      `<!DOCTYPE html>
<html>
<head>
  <title>${title}</title>
  <style>
    body { font-family: system-ui, sans-serif; text-align: center; padding-top: 50px; color: #333; }
    .container { max-width: 480px; margin: 0 auto; padding: 20px; }
    h1 { color: #c62828; font-size: 1.4em; }
  </style>
</head>
<body>
  <div class="container">
    <h1>${title}</h1>
    <p>${message}</p>
  </div>
</body>
</html>`,
    );
}
