"use client";

/**
 * OAuth2 PKCE flow for the *user* connect path.
 *
 * Unlike useMcpOAuthFlow (used in the admin create-server form), this hook
 * targets a server that already exists in the database.  It therefore skips
 * the temp-session cache step and calls /server/oauth/{serverId}/authorize
 * directly with the real server_id.
 *
 * On success it calls storeMCPOAuthUserCredential to persist the token for
 * the user and then invokes onSuccess so the caller can refresh UI state.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import {
  buildMcpOAuthAuthorizeUrl,
  exchangeMcpOAuthToken,
  getProxyBaseUrl,
  registerMcpOAuthClient,
  serverRootPath,
  storeMCPOAuthUserCredential,
} from "@/components/networking";
import NotificationsManager from "@/components/molecules/notifications_manager";
import { extractErrorMessage } from "@/utils/errorUtils";

export type UserMcpOAuthStatus = "idle" | "authorizing" | "exchanging" | "success" | "error";

interface UseUserMcpOAuthFlowOptions {
  accessToken: string;
  serverId: string;
  serverAlias?: string | null;
  /** Scopes to request, e.g. ["repo", "read:user"] */
  scopes?: string[];
  /** Pre-configured client_id if the MCP server record has one. */
  clientId?: string | null;
  onSuccess: () => void;
}

interface UseUserMcpOAuthFlowResult {
  startOAuthFlow: () => Promise<void>;
  status: UserMcpOAuthStatus;
  error: string | null;
}

const FLOW_STATE_KEY = "litellm-user-mcp-oauth-flow-state";
const RESULT_KEY = "litellm-mcp-oauth-result";
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

const b64url = (buf: ArrayBuffer) => {
  const bytes = new Uint8Array(buf);
  let s = "";
  bytes.forEach((b) => (s += String.fromCharCode(b)));
  return btoa(s).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
};

const genVerifier = () => {
  const arr = new Uint8Array(32);
  window.crypto.getRandomValues(arr);
  return b64url(arr.buffer);
};

const genChallenge = async (verifier: string) => {
  const data = new TextEncoder().encode(verifier);
  const digest = await window.crypto.subtle.digest("SHA-256", data);
  return b64url(digest);
};

const setStorage = (key: string, value: string) => {
  try {
    // Use sessionStorage only — do not write to localStorage.
    // The flow state may contain the LiteLLM access token; writing it to
    // localStorage would persist it across browser sessions and make it
    // readable by any injected script (XSS).
    window.sessionStorage.setItem(key, value);
  } catch (_) {}
};

const getStorage = (key: string): string | null => {
  try {
    return window.sessionStorage.getItem(key) || window.localStorage.getItem(key);
  } catch (_) {
    return null;
  }
};

const clearStorage = (...keys: string[]) => {
  keys.forEach((k) => {
    try {
      window.sessionStorage.removeItem(k);
      window.localStorage.removeItem(k);
    } catch (_) {}
  });
};

const buildCallbackUrl = (): string => {
  if (typeof window !== "undefined") {
    const path = window.location.pathname || "";
    const idx = path.indexOf("/ui");
    const prefix = idx >= 0 ? path.slice(0, idx + 3).replace(/\/+$/, "") : "";
    return `${window.location.origin}${prefix}/mcp/oauth/callback`;
  }
  const base = (getProxyBaseUrl() || "").replace(/\/+$/, "");
  const root = serverRootPath && serverRootPath !== "/" ? serverRootPath : "";
  return `${base}${root}/ui/mcp/oauth/callback`;
};

export const useUserMcpOAuthFlow = ({
  accessToken,
  serverId,
  serverAlias,
  scopes,
  clientId: preClientId,
  onSuccess,
}: UseUserMcpOAuthFlowOptions): UseUserMcpOAuthFlowResult => {
  const [status, setStatus] = useState<UserMcpOAuthStatus>("idle");
  const [error, setError] = useState<string | null>(null);
  const processingRef = useRef(false);

  const startOAuthFlow = useCallback(async () => {
    if (typeof window === "undefined") return;
    try {
      setStatus("authorizing");
      setError(null);

      let clientId: string | undefined = preClientId ?? undefined;
      let clientSecret: string | undefined;

      if (!clientId) {
        // Attempt dynamic client registration against the server's registration endpoint.
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

      const verifier = genVerifier();
      const challenge = await genChallenge(verifier);
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

      const flowState: StoredFlowState = {
        state,
        codeVerifier: verifier,
        serverId,
        redirectUri,
        clientId,
        clientSecret,
        scopes,
      };

      setStorage(FLOW_STATE_KEY, JSON.stringify(flowState));
      setStorage(RETURN_URL_KEY, window.location.href);

      window.location.href = authorizeUrl;
    } catch (err) {
      const msg = extractErrorMessage(err);
      setError(msg);
      setStatus("error");
      NotificationsManager.error(msg);
    }
  }, [accessToken, serverId, serverAlias, scopes, preClientId]);

  const resumeOAuthFlow = useCallback(async () => {
    if (typeof window === "undefined" || processingRef.current) return;

    const storedResult = getStorage(RESULT_KEY);
    if (!storedResult) return;

    // When multiple OAuth2ConnectButton components are mounted (one per server
    // card), each holds its own hook instance.  All run resumeOAuthFlow() on
    // mount and would compete for the same RESULT_KEY.  Peek at the stored
    // flow state first: only the hook instance whose serverId matches the one
    // that initiated the OAuth flow should consume the result.
    const rawFlowState = getStorage(FLOW_STATE_KEY);
    if (rawFlowState) {
      try {
        const peeked = JSON.parse(rawFlowState) as StoredFlowState;
        if (peeked.serverId && peeked.serverId !== serverId) return;
      } catch (_) {}
    }

    processingRef.current = true;
    clearStorage(RESULT_KEY);

    let payload: Record<string, unknown> | null = null;
    let flowState: StoredFlowState | null = null;

    try {
      payload = JSON.parse(storedResult);
      const raw = getStorage(FLOW_STATE_KEY);
      flowState = raw ? JSON.parse(raw) : null;
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
      });

      // Persist the token for this user via the backend.
      // accessToken comes from props — it is never stored in sessionStorage.
      await storeMCPOAuthUserCredential(accessToken, flowState.serverId, {
        access_token: token.access_token,
        refresh_token: token.refresh_token,
        expires_in: token.expires_in,
        scopes: flowState.scopes,
      });

      setStatus("success");
      setError(null);
      NotificationsManager.success("Connected successfully");
      onSuccess();
    } catch (err) {
      const msg = extractErrorMessage(err);
      setError(msg);
      setStatus("error");
      NotificationsManager.error(msg);
    } finally {
      clearStorage(FLOW_STATE_KEY);
      setTimeout(() => { processingRef.current = false; }, 1000);
    }
  }, [accessToken, serverId, onSuccess]);

  useEffect(() => {
    resumeOAuthFlow();
  }, [resumeOAuthFlow]);

  return { startOAuthFlow, status, error };
};
