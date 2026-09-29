// ---------------------------------------------------------------------------
// Google IdP client for team mode — the second hop of the two-hop flow.
// Wraps consent-URL construction, code exchange, identity lookup, and
// best-effort grant revocation behind a small interface so tests can inject
// a fake without any network.
// ---------------------------------------------------------------------------

import { Credentials, OAuth2Client } from 'google-auth-library';
import { fetchUserInfo, UserInfo } from '../userInfo.js';

export interface GoogleTokenResult {
  tokens: Credentials;
  identity: UserInfo;
}

export interface GoogleIdp {
  /** Consent URL for the given (our-side) state. */
  buildConsentUrl(state: string): string;
  /** Exchange the callback code and resolve who consented. */
  exchangeCode(code: string): Promise<GoogleTokenResult>;
  /** Best-effort revocation of a just-obtained grant (domain-allowlist
   * rejections must not leave a live grant behind). Never throws. */
  revokeGrant(token: string): Promise<void>;
}

export interface GoogleIdpClientOptions {
  clientId: string;
  clientSecret: string;
  redirectUri: string;
  scopes: string[];
  /** Single-domain deployments pre-filter the Google account picker. UX only —
   * enforcement happens on the `hd` claim at the callback. */
  hdHint?: string;
}

const REVOKE_URL = 'https://oauth2.googleapis.com/revoke';

export class GoogleIdpClient implements GoogleIdp {
  constructor(private readonly opts: GoogleIdpClientOptions) {}

  buildConsentUrl(state: string): string {
    // `prompt=consent` is security-critical, not just refresh-token hygiene:
    // with one static Google client serving every dynamically registered MCP
    // client, a silent re-consent would let an attacker-registered client mint
    // tokens as any victim who follows a link (confused deputy). A visible
    // consent screen on every authorization is the MCP-spec-recommended
    // mitigation. `select_account` matches the manage_accounts add flow.
    return this.newFlowClient().generateAuthUrl({
      access_type: 'offline',
      prompt: 'consent select_account',
      scope: [...this.opts.scopes],
      state,
      ...(this.opts.hdHint ? { hd: this.opts.hdHint } : {}),
    });
  }

  async exchangeCode(code: string): Promise<GoogleTokenResult> {
    const client = this.newFlowClient();
    const { tokens } = await client.getToken(code);
    client.setCredentials(tokens);
    const identity = await fetchUserInfo(client);
    return { tokens, identity };
  }

  async revokeGrant(token: string): Promise<void> {
    try {
      await fetch(REVOKE_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        body: new URLSearchParams({ token }).toString(),
      });
    } catch {
      // Best-effort by contract.
    }
  }

  private newFlowClient(): OAuth2Client {
    return new OAuth2Client({
      clientId: this.opts.clientId,
      clientSecret: this.opts.clientSecret,
      redirectUri: this.opts.redirectUri,
    });
  }
}
