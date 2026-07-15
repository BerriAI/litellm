import React from "react";
import { Button, Checkbox, Form, Input } from "antd";
import DcrBridgeToggle from "./DcrBridgeToggle";
import { credentialAuthClass, isClientForwardedTokenMode } from "@/components/mcp_tools/types";

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
 *
 * Blank fields follow the same convention as the M2M credential fields. On
 * create they mean "no app configured" (dynamic client registration). On edit
 * they mean "keep existing" ONLY when the credential class is unchanged: the
 * backend merges a partial update within the client-forwarded class, so a
 * true_passthrough <-> oauth_delegate switch keeps the stored app, but a switch
 * from a different class (e.g. oauth2) replaces it, so blanks then mean "no
 * app". Removing a stored app is an explicit checkbox (edit only) that writes
 * an explicit-null credential.
 */
export default function PassthroughAuthorizeSection({
  authType,
  oauthFlow,
  dcrBridgeInitialChecked,
  isEditing = false,
  savedAuthType,
  removeStoredApp = false,
  onRemoveStoredAppChange,
  appMayNotMatchUpstream = false,
}: {
  authType?: string | null;
  oauthFlow: PassthroughOAuthFlow;
  dcrBridgeInitialChecked?: boolean;
  isEditing?: boolean;
  savedAuthType?: string | null;
  removeStoredApp?: boolean;
  onRemoveStoredAppChange?: (remove: boolean) => void;
  appMayNotMatchUpstream?: boolean;
}) {
  if (!isClientForwardedTokenMode(authType)) return null;
  const authorizeButtonLabels: Record<string, string> = {
    authorizing: "Waiting for authorization...",
    exchanging: "Exchanging authorization code...",
  };
  const authorizeButtonLabel = authorizeButtonLabels[oauthFlow.status] ?? "Authorize & Fetch Tools (browser-only)";
  // On edit, "keep existing" only holds when the stored credential class is unchanged; a cross-class
  // switch (e.g. oauth2 -> true_passthrough) replaces credentials, so blanks then mean "no app".
  const classUnchanged = isEditing && credentialAuthClass(savedAuthType) === credentialAuthClass(authType);
  const clientIdPlaceholder = classUnchanged
    ? "Leave blank to keep the currently saved app (if any)"
    : "Leave blank to use dynamic client registration";
  const clientSecretPlaceholder = classUnchanged
    ? "Leave blank to keep the currently saved secret (if any)"
    : "Leave blank for public clients / PKCE";
  const clientIdExtra = classUnchanged
    ? "Set this to make everyone authorize through a specific app; required for upstreams without dynamic client registration (e.g. a pre-registered Slack app)."
    : "Switching the auth type discards the previously saved app; enter a client ID here or leave blank to use dynamic client registration.";
  return (
    <div className="rounded-lg border border-dashed border-gray-300 p-4 space-y-2 mb-4">
      <p className="text-sm text-gray-600">
        Callers bring their own upstream token for this auth type, so LiteLLM never stores tokens. To preview tools and
        configure the tool allowlist, authorize against the upstream here: the token stays in this browser session only
        and is never saved to LiteLLM. An OAuth app configured below IS saved with the server, so internal users who
        authorize from the Tools page go through it.
      </p>
      {appMayNotMatchUpstream && (
        <p className="text-sm text-amber-600">
          You changed the upstream URL or endpoints; the OAuth app entered here was registered for the previous upstream
          and may not be valid. Update the client ID, or clear it to use dynamic client registration.
        </p>
      )}
      <Form.Item
        label={<span className="text-sm font-medium text-gray-700">OAuth Client ID (optional)</span>}
        name={["credentials", "client_id"]}
        extra={clientIdExtra}
      >
        <Input.Password
          placeholder={clientIdPlaceholder}
          disabled={removeStoredApp}
          className="rounded-lg border-gray-300 focus:border-blue-500 focus:ring-blue-500"
        />
      </Form.Item>
      <Form.Item
        label={<span className="text-sm font-medium text-gray-700">OAuth Client Secret (optional)</span>}
        name={["credentials", "client_secret"]}
      >
        <Input.Password
          placeholder={clientSecretPlaceholder}
          disabled={removeStoredApp}
          className="rounded-lg border-gray-300 focus:border-blue-500 focus:ring-blue-500"
        />
      </Form.Item>
      <DcrBridgeToggle authType={authType} initialChecked={dcrBridgeInitialChecked} />
      {isEditing && onRemoveStoredAppChange && (
        <Checkbox checked={removeStoredApp} onChange={(e) => onRemoveStoredAppChange(e.target.checked)}>
          <span className="text-sm text-gray-700">
            Remove the saved OAuth app on save (the server goes back to dynamic client registration)
          </span>
        </Checkbox>
      )}
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
