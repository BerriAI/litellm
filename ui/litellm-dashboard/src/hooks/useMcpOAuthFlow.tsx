"use client";

import { useCallback, useEffect, useRef, useState } from "react";
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

  // crypto.subtle is only available in secure contexts (HTTPS or localhost).
  // When running over plain HTTP on a non-localhost address the API is undefined.
  if (typeof window !== "undefined" && window.crypto?.subtle?.digest) {
    const digest = await window.crypto.subtle.digest("SHA-256", data);
    return base64UrlEncode(digest);
  }

  // Fallback: compute SHA-256 using a pure-JS implementation so PKCE works
  // even when crypto.subtle is unavailable (e.g. HTTP on a LAN IP).
  const hash = jsSha256(data);
  return base64UrlEncode(hash.buffer);
};

/**
 * Minimal SHA-256 implementation (FIPS 180-4) used as a fallback when
 * the Web Crypto API is not available (non-secure HTTP contexts).
 *
 * Only used for PKCE code_challenge generation — not for security-critical
 * operations (the actual token exchange happens server-side over HTTPS).
 */
function jsSha256(message: Uint8Array): Uint8Array {
  const K: number[] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
  ];

  const rotr = (x: number, n: number) => (x >>> n) | (x << (32 - n));

  // Pre-processing: pad message
  const msgLen = message.length;
  const bitLen = msgLen * 8;
  // message + 0x80 + zeros + 8-byte length => multiple of 64 bytes
  const padLen = 64 - ((msgLen + 9) % 64 === 0 ? 64 : (msgLen + 9) % 64);
  const padded = new Uint8Array(msgLen + 1 + padLen + 8);
  padded.set(message);
  padded[msgLen] = 0x80;
  // big-endian 64-bit bit length (only lower 32 bits for messages < 512 MiB)
  const view = new DataView(padded.buffer);
  view.setUint32(padded.length - 4, bitLen, false);

  // Initial hash values
  let h0 = 0x6a09e667, h1 = 0xbb67ae85, h2 = 0x3c6ef372, h3 = 0xa54ff53a;
  let h4 = 0x510e527f, h5 = 0x9b05688c, h6 = 0x1f83d9ab, h7 = 0x5be0cd19;

  const W = new Int32Array(64);

  for (let offset = 0; offset < padded.length; offset += 64) {
    const block = new DataView(padded.buffer, offset, 64);
    for (let t = 0; t < 16; t++) W[t] = block.getInt32(t * 4, false);
    for (let t = 16; t < 64; t++) {
      const s0 = rotr(W[t - 15], 7) ^ rotr(W[t - 15], 18) ^ (W[t - 15] >>> 3);
      const s1 = rotr(W[t - 2], 17) ^ rotr(W[t - 2], 19) ^ (W[t - 2] >>> 10);
      W[t] = (W[t - 16] + s0 + W[t - 7] + s1) | 0;
    }

    let a = h0, b = h1, c = h2, d = h3, e = h4, f = h5, g = h6, h = h7;

    for (let t = 0; t < 64; t++) {
      const S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
      const ch = (e & f) ^ (~e & g);
      const temp1 = (h + S1 + ch + K[t] + W[t]) | 0;
      const S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
      const maj = (a & b) ^ (a & c) ^ (b & c);
      const temp2 = (S0 + maj) | 0;
      h = g; g = f; f = e; e = (d + temp1) | 0;
      d = c; c = b; b = a; a = (temp1 + temp2) | 0;
    }

    h0 = (h0 + a) | 0; h1 = (h1 + b) | 0; h2 = (h2 + c) | 0; h3 = (h3 + d) | 0;
    h4 = (h4 + e) | 0; h5 = (h5 + f) | 0; h6 = (h6 + g) | 0; h7 = (h7 + h) | 0;
  }

  const result = new Uint8Array(32);
  const rv = new DataView(result.buffer);
  rv.setUint32(0, h0, false); rv.setUint32(4, h1, false);
  rv.setUint32(8, h2, false); rv.setUint32(12, h3, false);
  rv.setUint32(16, h4, false); rv.setUint32(20, h5, false);
  rv.setUint32(24, h6, false); rv.setUint32(28, h7, false);
  return result;
}

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
  const processingRef = useRef(false);

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

  const setStorageItem = (key: string, value: string) => {
    if (typeof window === "undefined") return;
    try {
      // Store in both sessionStorage and localStorage for redundancy
      window.sessionStorage.setItem(key, value);
      window.localStorage.setItem(key, value);
    } catch (err) {
      console.warn(`Failed to set storage item ${key}`, err);
    }
  };

  const getStorageItem = (key: string): string | null => {
    if (typeof window === "undefined") return null;
    try {
      // Try sessionStorage first, fall back to localStorage
      return window.sessionStorage.getItem(key) || window.localStorage.getItem(key);
    } catch (err) {
      console.warn(`Failed to get storage item ${key}`, err);
      return null;
    }
  };

  const clearStoredFlow = () => {
    if (typeof window === "undefined") {
      return;
    }
    try {
      window.sessionStorage.removeItem(FLOW_STATE_KEY);
      window.sessionStorage.removeItem(RESULT_KEY);
      window.sessionStorage.removeItem(RETURN_URL_KEY);
      window.localStorage.removeItem(FLOW_STATE_KEY);
      window.localStorage.removeItem(RESULT_KEY);
      window.localStorage.removeItem(RETURN_URL_KEY);
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
        setStorageItem(FLOW_STATE_KEY, JSON.stringify(flowState));
        setStorageItem(RETURN_URL_KEY, window.location.href);
      } catch (storageErr) {
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

    // Prevent duplicate processing
    if (processingRef.current) {
      return;
    }

    let payload: Record<string, any> | null = null;
    let flowState: StoredFlowState | null = null;

    try {
      const storedPayload = getStorageItem(RESULT_KEY);
      if (!storedPayload) {
        return;
      }
      
      // Mark as processing
      processingRef.current = true;
      payload = JSON.parse(storedPayload);
      const storedFlowState = getStorageItem(FLOW_STATE_KEY);
      flowState = storedFlowState ? JSON.parse(storedFlowState) : null;
    } catch (err) {
      clearStoredFlow();
      processingRef.current = false;
      setError("Failed to resume OAuth flow. Please retry.");
      setStatus("error");
      NotificationsManager.error("Failed to resume OAuth flow. Please retry.");
      return;
    }

    if (!payload) {
      processingRef.current = false;
      return;
    }

    // Clear the result key after reading it
    if (typeof window !== "undefined") {
      try {
        window.sessionStorage.removeItem(RESULT_KEY);
        window.localStorage.removeItem(RESULT_KEY);
      } catch (err) {
        // Silently ignore storage errors
      }
    }

    try {
      if (!flowState || !flowState.state || !flowState.codeVerifier || !flowState.serverId) {
        throw new Error(
          "OAuth session state was lost. This can happen if you have strict browser privacy settings. " +
          "Please try again and ensure cookies/storage is enabled."
        );
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
      const message = err instanceof Error ? err.message : String(err);
      setError(message);
      setStatus("error");
      NotificationsManager.error(message);
    } finally {
      clearStoredFlow();
      // Reset processing flag after a delay to allow UI updates
      setTimeout(() => {
        processingRef.current = false;
      }, 1000);
    }
  }, [onTokenReceived]);

  useEffect(() => {
    resumeOAuthFlow();
  }, [resumeOAuthFlow]);

  return {
    startOAuthFlow,
    status,
    error,
    tokenResponse,
  };
};
