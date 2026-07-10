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
 * (true_passthrough / oauth_delegate). LiteLLM never stores upstream
 * credentials for these modes, so the token obtained here lives in this
 * browser session only: it is forwarded per-server for the tools preview and
 * allowlist configuration, and is never written to the server row or the
 * per-user credential store. The optional client credentials cover IdPs
 * without dynamic client registration (e.g. a pre-registered Slack app) and
 * ride the temporary authorize session only.
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
        Callers bring their own upstream token for this auth type, so LiteLLM stores no upstream credentials. To preview
        tools and configure the tool allowlist, authorize against the upstream here: the token stays in this browser
        session only and is never saved to LiteLLM.
      </p>
      <Form.Item
        label={<span className="text-sm font-medium text-gray-700">OAuth Client ID (optional, not saved)</span>}
        name={["credentials", "client_id"]}
        extra="Only needed when the upstream does not support dynamic client registration (e.g. a pre-registered Slack app). Used for this browser authorization only."
      >
        <Input.Password
          placeholder="Leave blank to use dynamic client registration"
          className="rounded-lg border-gray-300 focus:border-blue-500 focus:ring-blue-500"
        />
      </Form.Item>
      <Form.Item
        label={<span className="text-sm font-medium text-gray-700">OAuth Client Secret (optional, not saved)</span>}
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
          Token held for this browser session. Tools can now be previewed and configured; nothing was saved to LiteLLM.
        </p>
      )}
    </div>
  );
}
