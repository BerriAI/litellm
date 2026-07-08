"use client";

/**
 * OAuth2 PKCE flow for the Tools screen re-authentication path.
 *
 * Thin wrapper over useMcpOAuthPkceFlow. Unlike useUserMcpOAuthFlow (chat panel),
 * this flow:
 * - stores the resulting token in sessionStorage via mcpTokenStore only
 * - does NOT call storeMCPOAuthUserCredential (no backend DB write)
 * - uses "litellm-tools-mcp-oauth-result" as its result key to avoid
 *   collisions with the admin and user flows
 *
 * The OAuth callback page (src/app/mcp/oauth/callback/page.tsx) writes
 * to this key so this hook can pick up the result after the redirect.
 */

import { setToken } from "@/utils/mcpTokenStore";
import {
  McpOAuthStorageKeys,
  McpOAuthPkceStatus,
  UseMcpOAuthPkceFlowConfig,
  useMcpOAuthPkceFlow,
} from "./useMcpOAuthPkceFlow";

export type ToolsOAuthStatus = McpOAuthPkceStatus;

interface UseToolsOAuthFlowOptions {
  accessToken: string;
  serverId: string;
  serverAlias?: string | null;
  userId?: string | null;
  scopes?: string[];
  clientId?: string | null;
  /**
   * Invoked after the token is stored in sessionStorage. Receives the access
   * token because the caller holds it in component state to list tools; contrast
   * useUserMcpOAuthFlow, which persists the token server-side and exposes none.
   */
  onSuccess: (accessToken: string) => void;
}

interface UseToolsOAuthFlowResult {
  startOAuthFlow: () => Promise<void>;
  status: ToolsOAuthStatus;
  error: string | null;
}

const STORAGE_KEYS: McpOAuthStorageKeys = {
  flowState: "litellm-tools-mcp-oauth-flow-state",
  result: "litellm-tools-mcp-oauth-result",
  returnUrl: "litellm-mcp-oauth-return-url",
};

export const useToolsOAuthFlow = ({
  accessToken,
  serverId,
  serverAlias,
  userId,
  scopes,
  clientId,
  onSuccess,
}: UseToolsOAuthFlowOptions): UseToolsOAuthFlowResult => {
  const config: UseMcpOAuthPkceFlowConfig = {
    accessToken,
    serverId,
    serverAlias,
    scopes,
    clientId,
    storageKeys: STORAGE_KEYS,
    // Return to the current page (Tools tab) after the OAuth redirect.
    buildReturnUrl: () => window.location.href,
    // Store in sessionStorage only — no backend DB write.
    persistToken: (token, flowState) =>
      setToken(
        flowState.serverId,
        {
          access_token: token.access_token,
          expires_in: token.expires_in,
          refresh_token: token.refresh_token,
          token_type: token.token_type,
        },
        userId,
      ),
    onSuccess,
  };
  return useMcpOAuthPkceFlow(config);
};
