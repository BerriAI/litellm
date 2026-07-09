"use client";

/**
 * OAuth2 PKCE flow for the *user* connect path.
 *
 * Thin wrapper over useMcpOAuthPkceFlow. Unlike useMcpOAuthFlow (admin
 * create-server form), this targets a server that already exists in the
 * database, so it skips the temp-session cache step. On success it calls
 * storeMCPOAuthUserCredential to persist the token for the user.
 */

import { storeMCPOAuthUserCredential } from "@/components/networking";
import {
  McpOAuthStorageKeys,
  McpOAuthPkceStatus,
  UseMcpOAuthPkceFlowConfig,
  useMcpOAuthPkceFlow,
} from "./useMcpOAuthPkceFlow";

export type UserMcpOAuthStatus = McpOAuthPkceStatus;

interface UseUserMcpOAuthFlowOptions {
  accessToken: string;
  serverId: string;
  serverAlias?: string | null;
  /** Scopes to request, e.g. ["repo", "read:user"] */
  scopes?: string[];
  /** Pre-configured client_id if the MCP server record has one. */
  clientId?: string | null;
  /**
   * Invoked after the credential is persisted server-side. Receives no token by
   * design: unlike useToolsOAuthFlow, this flow stores the token in the backend
   * (storeMCPOAuthUserCredential) rather than handing it to the client, so the
   * caller only needs a "done" signal to refresh UI state.
   */
  onSuccess: () => void;
}

interface UseUserMcpOAuthFlowResult {
  startOAuthFlow: () => Promise<void>;
  status: UserMcpOAuthStatus;
  error: string | null;
}

const STORAGE_KEYS: McpOAuthStorageKeys = {
  flowState: "litellm-user-mcp-oauth-flow-state",
  // User-flow-specific keys to avoid collisions with the admin OAuth flow
  // (useMcpOAuthFlow), which uses "litellm-mcp-oauth-result".
  result: "litellm-user-mcp-oauth-result",
  returnUrl: "litellm-mcp-oauth-return-url",
};

export const useUserMcpOAuthFlow = ({
  accessToken,
  serverId,
  serverAlias,
  scopes,
  clientId,
  onSuccess,
}: UseUserMcpOAuthFlowOptions): UseUserMcpOAuthFlowResult => {
  const config: UseMcpOAuthPkceFlowConfig = {
    accessToken,
    serverId,
    serverAlias,
    scopes,
    clientId,
    storageKeys: STORAGE_KEYS,
    buildReturnUrl: () => {
      const returnUrl = new URL(window.location.href);
      returnUrl.searchParams.set("mcpOauthReturn", "apps");
      return returnUrl.toString();
    },
    // Persist the token for this user via the backend.
    // accessToken comes from props — it is never stored in sessionStorage.
    persistToken: (token, flowState) => {
      const credential = {
        access_token: token.access_token,
        refresh_token: token.refresh_token,
        expires_in: token.expires_in,
        scopes: flowState.scopes,
      };
      return storeMCPOAuthUserCredential(accessToken, flowState.serverId, credential);
    },
    // Explicitly drop the base hook's access_token argument: this flow keeps the
    // token server-side, so the public onSuccess deliberately exposes none.
    onSuccess: () => onSuccess(),
  };
  return useMcpOAuthPkceFlow(config);
};
