"use client";

import { useCallback, useEffect, useState } from "react";
import NotificationsManager from "@/components/molecules/notifications_manager";
import {
  buildMcpOAuthAuthorizeUrl,
  cacheTemporaryMcpServer,
  exchangeMcpOAuthToken,
  getProxyBaseUrl,
  registerMcpOAuthClient,
  serverRootPath,
} from "@/components/networking";

export type McpOAuthStatus = "idle" | "authorizing" | "exchanging" | "success" | "error";

interface UseMcpOAuthFlowOptions {
  accessToken: string | null;
  getCredentials: () => {
    client_id?: string;
    client_secret?: string;
    scopes?: string[];
  } | undefined;
  getTemporaryPayload: () => Record<string, any> | null;
  onTokenReceived: (tokenResponse: Record<string, any>) => void;
  onBeforeRedirect?: () => void;
}

interface UseMcpOAuthFlowResult {
  startOAuthFlow: () => Promise<void>;
  status: McpOAuthStatus;
  error: string | null;
  tokenResponse: Record<string, any> | null;
}

const base64UrlEncode = (buffer: ArrayBuffer) => {
  const bytes = new Uint8Array(buffer);
  let binary = "";
  bytes.forEach((b) => (binary += String.fromCharCode(b)));
  return btoa(binary).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
};

const generateCodeVerifier = () => {
  const array = new Uint8Array(32);
  window.crypto.getRandomValues(array);
  return base64UrlEncode(array.buffer);
};

const generateCodeChallenge = async (verifier: string) => {
  const data = new TextEncoder().encode(verifier);
  const digest = await window.crypto.subtle.digest("SHA-256", data);
  return base64UrlEncode(digest);
};

export const useMcpOAuthFlow = ({
  accessToken,
  getCredentials,
  getTemporaryPayload,
  onTokenReceived,
  onBeforeRedirect,
}: UseMcpOAuthFlowOptions): UseMcpOAuthFlowResult => {
  const [status, setStatus] = useState<McpOAuthStatus>("idle");
  const [error, setError] = useState<string | null>(null);
  const [tokenResponse, setTokenResponse] = useState<Record<string, any> | null>(null);

  const FLOW_STATE_KEY = "litellm-mcp-oauth-flow-state";
  const RESULT_KEY = "litellm-mcp-oauth-result";
  const RETURN_URL_KEY = "litellm-mcp-oauth-return-url";

  type StoredFlowState = {
    state: string;
    codeVerifier: string;
    clientId?: string;
    clientSecret?: string;
    serverId: string;
    redirectUri: string;
  };

  const clearStoredFlow = () => {
    if (typeof window === "undefined") {
      return;
    }
    try {
      window.sessionStorage.removeItem(FLOW_STATE_KEY);
      window.sessionStorage.removeItem(RESULT_KEY);
      window.sessionStorage.removeItem(RETURN_URL_KEY);
    } catch (err) {
      console.warn("Failed to clear OAuth storage", err);
    }
  };

  const buildCallbackUrl = () => {
    if (typeof window !== "undefined") {
      const path = window.location.pathname || "";
      const uiIndex = path.indexOf("/ui");
      const uiPrefix = uiIndex >= 0 ? path.slice(0, uiIndex + 3) : "";
      const normalizedPrefix = uiPrefix.replace(/\/+$/, "");
      return `${window.location.origin}${normalizedPrefix}/mcp/oauth/callback`;
    }

    const base = (getProxyBaseUrl() || "").replace(/\/+$/, "");
    const rootPrefix = serverRootPath && serverRootPath !== "/" ? serverRootPath : "";
    return `${base}${rootPrefix}/ui/mcp/oauth/callback`;
  };

  const callbackUrl = () => buildCallbackUrl();

  const startOAuthFlow = useCallback(async () => {
    const credentials = getCredentials() || {};

    if (!accessToken) {
      setError("Missing admin token");
      NotificationsManager.error("Access token missing. Please re-authenticate and try again.");
      return;
    }

    const temporaryPayload = getTemporaryPayload();
    if (!temporaryPayload || !temporaryPayload.url || !temporaryPayload.transport) {
      const message = "Please complete server URL and transport before starting OAuth.";
      setError(message);
      NotificationsManager.error(message);
      return;
    }
    try {
      setStatus("authorizing");
      setError(null);

      const cachedServer = await cacheTemporaryMcpServer(accessToken, temporaryPayload);
      const serverId = cachedServer?.server_id?.trim();
      if (!serverId) {
        throw new Error("Temporary MCP server identifier missing. Please retry.");
      }

      let registeredClient: { clientId?: string; clientSecret?: string } = {};
      const hasPreconfiguredCredentials = Boolean(temporaryPayload.credentials?.client_id && temporaryPayload.credentials?.client_secret);

      if (!hasPreconfiguredCredentials) {
        const registration = await registerMcpOAuthClient(accessToken, serverId, {
          client_name: temporaryPayload.alias || temporaryPayload.server_name || serverId,
          grant_types: ["authorization_code", "refresh_token"],
          response_types: ["code"],
          token_endpoint_auth_method:
            temporaryPayload.credentials && temporaryPayload.credentials.client_secret ? "client_secret_post" : "none",
        });
        registeredClient = {
          clientId: registration?.client_id,
          clientSecret: registration?.client_secret,
        };
      }

      const verifier = generateCodeVerifier();
      const challenge = await generateCodeChallenge(verifier);
      const state = crypto.randomUUID();

      const clientId = registeredClient.clientId || credentials.client_id;
      const scopeString = Array.isArray(credentials.scopes)
        ? credentials.scopes.filter((s) => s && s.trim().length > 0).join(" ")
        : undefined;

      const authorizeUrl = buildMcpOAuthAuthorizeUrl({
        serverId,
        clientId: clientId,
        redirectUri: callbackUrl(),
        state,
        codeChallenge: challenge,
        scope: scopeString,
      });

      const flowState: StoredFlowState = {
        state,
        codeVerifier: verifier,
        clientId,
        clientSecret: registeredClient.clientSecret || credentials.client_secret,
        serverId,
        redirectUri: callbackUrl(),
      };

      if (typeof window === "undefined") {
        throw new Error("OAuth redirect is only supported in the browser.");
      }

      if (onBeforeRedirect) {
        try {
          onBeforeRedirect();
        } catch (prepErr) {
          console.error("Failed to prepare for OAuth redirect", prepErr);
        }
      }

      try {
        window.sessionStorage.setItem(FLOW_STATE_KEY, JSON.stringify(flowState));
        window.sessionStorage.setItem(RETURN_URL_KEY, window.location.href);
      } catch (storageErr) {
        console.error("Unable to persist OAuth state", storageErr);
        throw new Error("Unable to access browser storage for OAuth. Please enable storage and retry.");
      }

      window.location.href = authorizeUrl;
    } catch (err) {
      console.error("Failed to start OAuth flow", err);
      setStatus("error");
      const message = err instanceof Error ? err.message : String(err);
      setError(message);
      NotificationsManager.error(message);
    }
  }, [accessToken, getCredentials, getTemporaryPayload, onBeforeRedirect]);

  const resumeOAuthFlow = useCallback(async () => {
    if (typeof window === "undefined") {
      return;
    }

    let payload: Record<string, any> | null = null;
    let flowState: StoredFlowState | null = null;

    try {
      const storedPayload = window.sessionStorage.getItem(RESULT_KEY);
      if (!storedPayload) {
        return;
      }
      payload = JSON.parse(storedPayload);
      flowState = JSON.parse(window.sessionStorage.getItem(FLOW_STATE_KEY) || "null");
    } catch (err) {
      console.error("Failed to read OAuth session state", err);
      clearStoredFlow();
      setError("Failed to resume OAuth flow. Please retry.");
      setStatus("error");
      NotificationsManager.error("Failed to resume OAuth flow. Please retry.");
      return;
    }

    if (!payload) {
      return;
    }

    window.sessionStorage.removeItem(RESULT_KEY);

    try {
      if (!flowState || !flowState.state || !flowState.codeVerifier || !flowState.serverId) {
        throw new Error("Missing OAuth session state. Please retry.");
      }
      if (!payload.state || payload.state !== flowState.state) {
        throw new Error("OAuth state mismatch. Please retry.");
      }
      if (payload.error) {
        throw new Error(payload.error_description || payload.error);
      }
      if (!payload.code) {
        throw new Error("Authorization code missing in callback.");
      }

      setStatus("exchanging");
      const token = await exchangeMcpOAuthToken({
        serverId: flowState.serverId,
        code: payload.code,
        clientId: flowState.clientId,
        clientSecret: flowState.clientSecret,
        codeVerifier: flowState.codeVerifier,
        redirectUri: flowState.redirectUri,
      });

      onTokenReceived(token);
      setTokenResponse(token);
      setStatus("success");
      setError(null);
      NotificationsManager.success("OAuth token retrieved successfully");
    } catch (err) {
      console.error("OAuth flow failed", err);
      const message = err instanceof Error ? err.message : String(err);
      setError(message);
      setStatus("error");
      NotificationsManager.error(message);
    } finally {
      clearStoredFlow();
    }
  }, [onTokenReceived]);

  useEffect(() => {
    let cancelled = false;

    const maybeResume = async () => {
      if (cancelled) {
        return;
      }
      await resumeOAuthFlow();
    };

    maybeResume();

    return () => {
      cancelled = true;
    };
  }, [resumeOAuthFlow]);

  return {
    startOAuthFlow,
    status,
    error,
    tokenResponse,
  };
};
