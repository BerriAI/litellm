"use client";

/**
 * Shared OAuth2 PKCE flow for MCP servers that already exist in the database.
 *
 * Both the Tools re-auth path (useToolsOAuthFlow) and the user connect path
 * (useUserMcpOAuthFlow) run the identical authorize -> redirect -> exchange
 * sequence; they diverge only in three places, injected via config:
 *   - the sessionStorage keys they read/write (so the flows don't collide),
 *   - how the post-redirect return URL is built,
 *   - what they do with the exchanged token (sessionStorage vs backend DB).
 *
 * The admin create-server flow (useMcpOAuthFlow) is deliberately NOT built on
 * this: it targets a not-yet-persisted server and needs an extra temp-session
 * cache step, so its logic is materially different.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { buildMcpOAuthAuthorizeUrl, exchangeMcpOAuthToken, registerMcpOAuthClient } from "@/components/networking";
import NotificationsManager from "@/components/molecules/notifications_manager";
import { extractErrorMessage } from "@/utils/errorUtils";
import { generateCodeChallenge, generateCodeVerifier } from "@/utils/pkce";
import { getSecureItem, setSecureItem } from "@/utils/secureStorage";
import { buildCallbackUrl, clearStorage } from "./mcpOAuthUtils";

export type McpOAuthPkceStatus = "idle" | "authorizing" | "exchanging" | "success" | "error";

export interface McpOAuthFlowState {
  state: string;
  codeVerifier: string;
  serverId: string;
  redirectUri: string;
  clientId?: string;
  clientSecret?: string;
  scopes?: string[];
}

export interface McpOAuthTokenResult {
  access_token: string;
  expires_in?: number;
  refresh_token?: string;
  token_type?: string;
}

export interface McpOAuthStorageKeys {
  flowState: string;
  result: string;
  returnUrl: string;
}

export interface UseMcpOAuthPkceFlowConfig {
  accessToken: string;
  serverId: string;
  serverAlias?: string | null;
  scopes?: string[];
  clientId?: string | null;
  storageKeys: McpOAuthStorageKeys;
  /** URL to return to after the OAuth redirect completes. */
  buildReturnUrl: () => string;
  /** Persist the exchanged token (sessionStorage, backend DB, etc.). Awaited; any resolved value is ignored. */
  persistToken: (token: McpOAuthTokenResult, flowState: McpOAuthFlowState) => Promise<unknown> | void;
  /** Invoked after the token is persisted so the caller can refresh UI state. */
  onSuccess: (accessToken: string) => void;
}

interface UseMcpOAuthPkceFlowResult {
  startOAuthFlow: () => Promise<void>;
  status: McpOAuthPkceStatus;
  error: string | null;
}

export const useMcpOAuthPkceFlow = ({
  accessToken,
  serverId,
  serverAlias,
  scopes,
  clientId: preClientId,
  storageKeys,
  buildReturnUrl,
  persistToken,
  onSuccess,
}: UseMcpOAuthPkceFlowConfig): UseMcpOAuthPkceFlowResult => {
  const [status, setStatus] = useState<McpOAuthPkceStatus>("idle");
  const [error, setError] = useState<string | null>(null);
  const processingRef = useRef(false);

  // Latest-ref pattern: reassigned every render so the memoized callbacks below
  // read the current buildReturnUrl/persistToken/onSuccess at call time without
  // listing them as deps (which would re-subscribe the on-mount resume effect).
  const callbacksRef = useRef({ buildReturnUrl, persistToken, onSuccess });
  callbacksRef.current = { buildReturnUrl, persistToken, onSuccess };

  const { flowState: FLOW_STATE_KEY, result: RESULT_KEY, returnUrl: RETURN_URL_KEY } = storageKeys;

  const startOAuthFlow = useCallback(async () => {
    if (typeof window === "undefined") return;
    try {
      setStatus("authorizing");
      setError(null);

      let clientId: string | undefined = preClientId ?? undefined;
      let clientSecret: string | undefined;

      if (!clientId) {
        try {
          const reg = await registerMcpOAuthClient(accessToken, serverId, {
            client_name: serverAlias || serverId,
            grant_types: ["authorization_code", "refresh_token"],
            response_types: ["code"],
            token_endpoint_auth_method: "none",
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
      const redirectUri = buildCallbackUrl();
      const scopeString = scopes?.filter((s) => s.trim()).join(" ");

      const authorizeUrl = buildMcpOAuthAuthorizeUrl({
        serverId,
        clientId,
        redirectUri,
        state,
        codeChallenge: challenge,
        scope: scopeString,
      });

      const flowState: McpOAuthFlowState = {
        state,
        codeVerifier: verifier,
        serverId,
        redirectUri,
        clientId,
        clientSecret,
        scopes,
      };

      setSecureItem(FLOW_STATE_KEY, JSON.stringify(flowState));
      setSecureItem(RETURN_URL_KEY, callbacksRef.current.buildReturnUrl());

      window.location.href = authorizeUrl;
    } catch (err) {
      const msg = extractErrorMessage(err);
      setError(msg);
      setStatus("error");
      NotificationsManager.error(msg);
    }
  }, [accessToken, serverId, serverAlias, scopes, preClientId, FLOW_STATE_KEY, RETURN_URL_KEY]);

  const resumeOAuthFlow = useCallback(async () => {
    if (typeof window === "undefined" || processingRef.current) return;

    const storedResult = getSecureItem(RESULT_KEY);
    if (!storedResult) return;

    // The callback page writes a result for every OAuth flow (Tools, user, admin).
    // Peek at this hook's flow state first and bail unless it exists and its
    // serverId matches: only the hook instance that initiated the flow should
    // consume the result. Without this, a stale result from another flow would
    // surface "OAuth session state was lost" on unrelated hook instances.
    const rawFlowState = getSecureItem(FLOW_STATE_KEY);
    if (!rawFlowState) return;

    let flowState: McpOAuthFlowState | null = null;
    try {
      flowState = JSON.parse(rawFlowState) as McpOAuthFlowState;
      if (flowState.serverId && flowState.serverId !== serverId) return;
    } catch (_) {}

    processingRef.current = true;
    clearStorage(RESULT_KEY);

    let payload: Record<string, unknown> | null = null;
    try {
      payload = JSON.parse(storedResult);
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
      const token: McpOAuthTokenResult = await exchangeMcpOAuthToken({
        serverId: flowState.serverId,
        code: payload.code as string,
        clientId: flowState.clientId,
        clientSecret: flowState.clientSecret,
        codeVerifier: flowState.codeVerifier,
        redirectUri: flowState.redirectUri,
        accessToken,
      });

      await callbacksRef.current.persistToken(token, flowState);

      setStatus("success");
      setError(null);
      NotificationsManager.success("Connected successfully");
      callbacksRef.current.onSuccess(token.access_token);
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
  }, [accessToken, serverId, FLOW_STATE_KEY, RESULT_KEY]);

  useEffect(() => {
    resumeOAuthFlow();
  }, [resumeOAuthFlow]);

  return { startOAuthFlow, status, error };
};
