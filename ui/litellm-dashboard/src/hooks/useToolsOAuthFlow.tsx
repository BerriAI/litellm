"use client";

/**
 * OAuth2 PKCE flow for the Tools screen re-authentication path.
 *
 * Unlike useUserMcpOAuthFlow (used in the chat panel), this hook:
 * - stores the resulting token in sessionStorage via mcpTokenStore only
 * - does NOT call storeMCPOAuthUserCredential (no backend DB write)
 * - uses "litellm-tools-mcp-oauth-result" as its result key to avoid
 *   collisions with the admin and user flows
 *
 * The OAuth callback page (src/app/mcp/oauth/callback/page.tsx) writes
 * to this key so this hook can pick up the result after the redirect.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { buildMcpOAuthAuthorizeUrl, exchangeMcpOAuthToken, registerMcpOAuthClient } from "@/components/networking";
import NotificationsManager from "@/components/molecules/notifications_manager";
import { extractErrorMessage } from "@/utils/errorUtils";
import { generateCodeChallenge, generateCodeVerifier } from "@/utils/pkce";
import { getSecureItem, setSecureItem } from "@/utils/secureStorage";
import { setToken } from "@/utils/mcpTokenStore";
import { buildCallbackUrl, clearStorage } from "./mcpOAuthUtils";

export type ToolsOAuthStatus = "idle" | "authorizing" | "exchanging" | "success" | "error";

interface UseToolsOAuthFlowOptions {
  accessToken: string;
  serverId: string;
  serverAlias?: string | null;
  userId?: string | null;
  scopes?: string[];
  clientId?: string | null;
  /**
   * True for the client-forwarded token modes (true_passthrough / oauth_delegate): the gateway
   * mints the OAuth client itself during /authorize and carries it in the sealed state/code, so
   * the browser must not register a client of its own (each browser-side registration creates an
   * extra client at the IdP that the gateway's minted one then supersedes).
   */
  gatewayMintsClient?: boolean;
  onSuccess: (accessToken: string) => void;
}

interface UseToolsOAuthFlowResult {
  startOAuthFlow: () => Promise<void>;
  status: ToolsOAuthStatus;
  error: string | null;
}

const FLOW_STATE_KEY = "litellm-tools-mcp-oauth-flow-state";
const RESULT_KEY = "litellm-tools-mcp-oauth-result";
const RETURN_URL_KEY = "litellm-mcp-oauth-return-url";

type StoredFlowState = {
  state: string;
  codeVerifier: string;
  serverId: string;
  redirectUri: string;
  clientId?: string;
  clientSecret?: string;
  scopes?: string[];
};

export const useToolsOAuthFlow = ({
  accessToken,
  serverId,
  serverAlias,
  userId,
  scopes,
  clientId: preClientId,
  gatewayMintsClient,
  onSuccess,
}: UseToolsOAuthFlowOptions): UseToolsOAuthFlowResult => {
  const [status, setStatus] = useState<ToolsOAuthStatus>("idle");
  const [error, setError] = useState<string | null>(null);
  const processingRef = useRef(false);
  const onSuccessRef = useRef(onSuccess);
  onSuccessRef.current = onSuccess;

  const startOAuthFlow = useCallback(async () => {
    if (typeof window === "undefined") return;
    try {
      setStatus("authorizing");
      setError(null);

      let clientId: string | undefined = preClientId ?? undefined;
      let clientSecret: string | undefined;
      const redirectUri = buildCallbackUrl();

      if (!clientId && !gatewayMintsClient) {
        try {
          const reg = await registerMcpOAuthClient(accessToken, serverId, {
            client_name: serverAlias || serverId,
            grant_types: ["authorization_code", "refresh_token"],
            response_types: ["code"],
            token_endpoint_auth_method: "none",
            // dcr_bridge servers relay this registration upstream and bind the
            // minted client to the browser's own callback; without it the relay
            // rejects the registration and the flow dead-ends clientless.
            redirect_uris: [redirectUri],
          });
          clientId = reg?.client_id;
          clientSecret = reg?.client_secret;
        } catch (_) {
          // Registration is optional; proceed without client_id
        }
      }

      const verifier = generateCodeVerifier();
      const challenge = await generateCodeChallenge(verifier);
      const state = crypto.randomUUID();
      const scopeString = scopes?.filter((s) => s.trim()).join(" ");

      const authorizeUrl = buildMcpOAuthAuthorizeUrl({
        serverId,
        clientId,
        redirectUri,
        state,
        codeChallenge: challenge,
        scope: scopeString,
      });

      const flowState: StoredFlowState = {
        state,
        codeVerifier: verifier,
        serverId,
        redirectUri,
        clientId,
        clientSecret,
        scopes,
      };

      setSecureItem(FLOW_STATE_KEY, JSON.stringify(flowState));
      // Return to the current page (Tools tab) after the OAuth redirect
      setSecureItem(RETURN_URL_KEY, window.location.href);

      window.location.href = authorizeUrl;
    } catch (err) {
      const msg = extractErrorMessage(err);
      setError(msg);
      setStatus("error");
      NotificationsManager.error(msg);
    }
  }, [accessToken, serverId, serverAlias, scopes, preClientId, gatewayMintsClient]);

  const resumeOAuthFlow = useCallback(async () => {
    if (typeof window === "undefined" || processingRef.current) return;

    const storedResult = getSecureItem(RESULT_KEY);
    if (!storedResult) return;

    // The callback page writes to this result key for every OAuth flow (including
    // the admin server-creation flow).  Guard: only proceed if *this* hook's flow
    // state exists, meaning startOAuthFlow() was actually called from the Tools screen.
    // Without this guard, a stale result written during server creation would trigger
    // "OAuth session state was lost" when the user navigates to the Tools tab.
    const rawFlowState = getSecureItem(FLOW_STATE_KEY);
    if (!rawFlowState) return;

    let peeked: StoredFlowState | null = null;
    try {
      peeked = JSON.parse(rawFlowState) as StoredFlowState;
      if (peeked.serverId && peeked.serverId !== serverId) return;
    } catch (_) {}

    processingRef.current = true;
    clearStorage(RESULT_KEY);

    let payload: Record<string, unknown> | null = null;
    let flowState: StoredFlowState | null = null;

    try {
      payload = JSON.parse(storedResult);
      flowState = peeked;
    } catch (_) {
      setError("Failed to resume OAuth flow. Please retry.");
      setStatus("error");
      processingRef.current = false;
      clearStorage(FLOW_STATE_KEY);
      return;
    }

    try {
      if (!flowState?.state || !flowState.codeVerifier || !flowState.serverId) {
        throw new Error("OAuth session state was lost. Please retry.");
      }
      if (!payload?.state || payload.state !== flowState.state) {
        throw new Error("OAuth state mismatch. Please retry.");
      }
      if (payload.error) {
        throw new Error((payload.error_description as string) || (payload.error as string));
      }
      if (!payload.code) {
        throw new Error("Authorization code missing in callback.");
      }

      setStatus("exchanging");
      const token = await exchangeMcpOAuthToken({
        serverId: flowState.serverId,
        code: payload.code as string,
        clientId: flowState.clientId,
        clientSecret: flowState.clientSecret,
        codeVerifier: flowState.codeVerifier,
        redirectUri: flowState.redirectUri,
        accessToken,
      });

      // Store in sessionStorage only — no backend DB write
      setToken(
        flowState.serverId,
        {
          access_token: token.access_token,
          expires_in: token.expires_in,
          refresh_token: token.refresh_token,
          token_type: token.token_type,
        },
        userId,
      );

      setStatus("success");
      setError(null);
      NotificationsManager.success("Connected successfully");
      onSuccessRef.current(token.access_token);
    } catch (err) {
      const msg = extractErrorMessage(err);
      setError(msg);
      setStatus("error");
      NotificationsManager.error(msg);
    } finally {
      clearStorage(FLOW_STATE_KEY);
      setTimeout(() => {
        processingRef.current = false;
      }, 1000);
    }
  }, [accessToken, serverId, userId]);

  useEffect(() => {
    resumeOAuthFlow();
  }, [resumeOAuthFlow]);

  return { startOAuthFlow, status, error };
};
