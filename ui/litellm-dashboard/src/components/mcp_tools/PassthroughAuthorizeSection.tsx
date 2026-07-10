import React from "react";
import { Button, Form, Input } from "antd";
import { isClientForwardedTokenMode } from "./types";

interface PassthroughOAuthFlow {
  startOAuthFlow: () => void | Promise<void>;
  status: string;
  error: string | null;
  tokenResponse: { access_token?: string; expires_in?: number } | null;
}

/**
 * Browser-only Authorize & Fetch for the client-forwarded token modes
 * (true_passthrough / oauth_delegate). Tokens are never stored: the token
 * obtained here lives in this browser session only, forwarded per-server for
 * the tools preview and allowlist configuration, and is never written to the
 * server row or the per-user credential store. The optional OAuth client
 * credentials cover IdPs without dynamic client registration (e.g. a
 * pre-registered Slack app); unlike the token they ARE saved onto the server
 * as declared config, so internal users' Authorize relays through the org's
 * app instead of dead-ending on upstreams that cannot mint clients.
 */
export default function PassthroughAuthorizeSection({
  authType,
  oauthFlow,
}: {
  authType?: string | null;
  oauthFlow: PassthroughOAuthFlow;
}) {
  if (!isClientForwardedTokenMode(authType)) return null;
  const authorizeButtonLabels: Record<string, string> = {
    authorizing: "Waiting for authorization...",
    exchanging: "Exchanging authorization code...",
  };
  const authorizeButtonLabel = authorizeButtonLabels[oauthFlow.status] ?? "Authorize & Fetch Tools (browser-only)";
  return (
    <div className="rounded-lg border border-dashed border-gray-300 p-4 space-y-2 mb-4">
      <p className="text-sm text-gray-600">
        Callers bring their own upstream token for this auth type, so LiteLLM never stores tokens. To preview tools and
        configure the tool allowlist, authorize against the upstream here: the token stays in this browser session only
        and is never saved to LiteLLM. An OAuth app configured below IS saved with the server, so internal users who
        authorize from the Tools page go through it.
      </p>
      <Form.Item
        label={<span className="text-sm font-medium text-gray-700">OAuth Client ID (optional, saved)</span>}
        name={["credentials", "client_id"]}
        extra="Set this to make everyone authorize through a specific app; required for upstreams without dynamic client registration (e.g. a pre-registered Slack app)."
      >
        <Input.Password
          placeholder="Leave blank to use dynamic client registration"
          className="rounded-lg border-gray-300 focus:border-blue-500 focus:ring-blue-500"
        />
      </Form.Item>
      <Form.Item
        label={<span className="text-sm font-medium text-gray-700">OAuth Client Secret (optional, saved)</span>}
        name={["credentials", "client_secret"]}
      >
        <Input.Password
          placeholder="Leave blank for public clients / PKCE"
          className="rounded-lg border-gray-300 focus:border-blue-500 focus:ring-blue-500"
        />
      </Form.Item>
      <Button
        onClick={oauthFlow.startOAuthFlow}
        disabled={oauthFlow.status === "authorizing" || oauthFlow.status === "exchanging"}
      >
        {authorizeButtonLabel}
      </Button>
      {oauthFlow.error && <p className="text-sm text-red-500">{oauthFlow.error}</p>}
      {oauthFlow.status === "success" && oauthFlow.tokenResponse?.access_token && (
        <p className="text-sm text-green-600">
          Token held for this browser session. Tools can now be previewed and configured; the token was not saved to
          LiteLLM.
        </p>
      )}
    </div>
  );
}
