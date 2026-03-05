"use client";

import React, { useState, useEffect, useRef, useCallback } from "react";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface McpServer {
  server_id: string;
  server_name: string;
  description?: string;
  is_byok: boolean;
  has_user_credential: boolean;
  status?: string;
}

interface ChatMessage {
  role: "user" | "assistant" | "system";
  content: string;
}

type ConnectionState = "idle" | "connecting" | "connected" | "error";

interface ServerConnectionStatus {
  state: ConnectionState;
  errorMessage?: string;
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const DEMO_VIRTUAL_KEY = "sk-rJGGhNOSHLXi8OwdDAIX8Q";
const CLIENT_ID = "user-alice-123";
const PROXY_BASE_URL =
  process.env.NEXT_PUBLIC_LITELLM_PROXY_BASE_URL || "http://localhost:4000";

// ---------------------------------------------------------------------------
// PKCE helpers
// ---------------------------------------------------------------------------

async function generatePKCE(): Promise<{ verifier: string; challenge: string }> {
  const array = new Uint8Array(32);
  crypto.getRandomValues(array);
  const verifier = btoa(String.fromCharCode(...array))
    .replace(/\+/g, "-")
    .replace(/\//g, "_")
    .replace(/=/g, "");

  const encoder = new TextEncoder();
  const data = encoder.encode(verifier);
  const hash = await crypto.subtle.digest("SHA-256", data);
  const challenge = btoa(String.fromCharCode(...new Uint8Array(hash)))
    .replace(/\+/g, "-")
    .replace(/\//g, "_")
    .replace(/=/g, "");

  return { verifier, challenge };
}

function generateState(): string {
  const array = new Uint8Array(16);
  crypto.getRandomValues(array);
  return btoa(String.fromCharCode(...array))
    .replace(/\+/g, "-")
    .replace(/\//g, "_")
    .replace(/=/g, "");
}

// ---------------------------------------------------------------------------
// Icons (inline SVG — no icon library dependency)
// ---------------------------------------------------------------------------

function LockIcon({ className }: { className?: string }) {
  return (
    <svg
      className={className}
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={2}
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <rect x="3" y="11" width="18" height="11" rx="2" ry="2" />
      <path d="M7 11V7a5 5 0 0 1 10 0v4" />
    </svg>
  );
}

function CheckIcon({ className }: { className?: string }) {
  return (
    <svg
      className={className}
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={2.5}
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <polyline points="20 6 9 17 4 12" />
    </svg>
  );
}

function ServerIcon({ className }: { className?: string }) {
  return (
    <svg
      className={className}
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={2}
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <rect x="2" y="2" width="20" height="8" rx="2" ry="2" />
      <rect x="2" y="14" width="20" height="8" rx="2" ry="2" />
      <line x1="6" y1="6" x2="6.01" y2="6" />
      <line x1="6" y1="18" x2="6.01" y2="18" />
    </svg>
  );
}

function KeyIcon({ className }: { className?: string }) {
  return (
    <svg
      className={className}
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={2}
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <circle cx="7.5" cy="15.5" r="5.5" />
      <path d="M21 2L11 12" />
      <path d="M15 6l1 1" />
    </svg>
  );
}

function SpinnerIcon({ className }: { className?: string }) {
  return (
    <svg
      className={className}
      xmlns="http://www.w3.org/2000/svg"
      fill="none"
      viewBox="0 0 24 24"
    >
      <circle
        className="opacity-25"
        cx="12"
        cy="12"
        r="10"
        stroke="currentColor"
        strokeWidth="4"
      />
      <path
        className="opacity-75"
        fill="currentColor"
        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
      />
    </svg>
  );
}

// ---------------------------------------------------------------------------
// Main page component
// ---------------------------------------------------------------------------

export default function ByokDemoPage() {
  const [servers, setServers] = useState<McpServer[]>([]);
  const [loadingServers, setLoadingServers] = useState(true);
  const [fetchError, setFetchError] = useState<string | null>(null);
  const [connectionStatus, setConnectionStatus] = useState<
    Record<string, ServerConnectionStatus>
  >({});
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([
    {
      role: "system",
      content:
        "Welcome! This demo shows the LiteLLM MCP BYOK OAuth 2.1 flow. Connect a BYOK server on the left to get started.",
    },
  ]);

  // Ref to track active popup intervals so we can clear them on unmount
  const popupIntervalsRef = useRef<Record<string, ReturnType<typeof setInterval>>>({});

  // ---------------------------------------------------------------------------
  // Fetch MCP servers
  // ---------------------------------------------------------------------------

  const fetchServers = useCallback(async () => {
    setLoadingServers(true);
    setFetchError(null);
    try {
      const res = await fetch(`${PROXY_BASE_URL}/v1/mcp/server`, {
        headers: {
          Authorization: `Bearer ${DEMO_VIRTUAL_KEY}`,
          "Content-Type": "application/json",
        },
      });
      if (!res.ok) {
        throw new Error(`HTTP ${res.status}: ${res.statusText}`);
      }
      const data = await res.json();
      // The endpoint may return { data: McpServer[] } or McpServer[]
      const list: McpServer[] = Array.isArray(data)
        ? data
        : Array.isArray(data?.data)
        ? data.data
        : [];
      setServers(list);
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : String(err);
      setFetchError(message);
      setServers([]);
    } finally {
      setLoadingServers(false);
    }
  }, []);

  useEffect(() => {
    fetchServers();
  }, [fetchServers]);

  // Cleanup popup intervals on unmount
  useEffect(() => {
    const intervals = popupIntervalsRef.current;
    return () => {
      Object.values(intervals).forEach(clearInterval);
    };
  }, []);

  // ---------------------------------------------------------------------------
  // OAuth PKCE flow
  // ---------------------------------------------------------------------------

  const handleConnect = useCallback(
    async (server: McpServer) => {
      const { server_id, server_name } = server;

      setConnectionStatus((prev) => ({
        ...prev,
        [server_id]: { state: "connecting" },
      }));

      let verifier: string;
      let challenge: string;

      try {
        const pkce = await generatePKCE();
        verifier = pkce.verifier;
        challenge = pkce.challenge;
      } catch (err: unknown) {
        const message = err instanceof Error ? err.message : String(err);
        setConnectionStatus((prev) => ({
          ...prev,
          [server_id]: { state: "error", errorMessage: `PKCE generation failed: ${message}` },
        }));
        return;
      }

      const state = generateState();
      const redirectUri = window.location.href.split("?")[0];

      // Store PKCE data keyed by server_id for later retrieval
      sessionStorage.setItem(
        `byok_pkce_${server_id}`,
        JSON.stringify({ verifier, state, redirectUri })
      );

      const params = new URLSearchParams({
        server_id,
        client_id: CLIENT_ID,
        redirect_uri: redirectUri,
        code_challenge: challenge,
        code_challenge_method: "S256",
        state,
        response_type: "code",
      });

      const authorizeUrl = `${PROXY_BASE_URL}/v1/mcp/oauth/authorize?${params.toString()}`;

      const popup = window.open(authorizeUrl, "byok_auth", "width=600,height=700");
      if (!popup) {
        setConnectionStatus((prev) => ({
          ...prev,
          [server_id]: {
            state: "error",
            errorMessage:
              "Popup was blocked. Allow popups for this site and try again.",
          },
        }));
        return;
      }

      // Clear any existing interval for this server
      if (popupIntervalsRef.current[server_id]) {
        clearInterval(popupIntervalsRef.current[server_id]);
      }

      const intervalId = setInterval(async () => {
        try {
          if (popup.closed) {
            clearInterval(intervalId);
            delete popupIntervalsRef.current[server_id];
            // If we ended up here without connecting, revert to idle
            setConnectionStatus((prev) => {
              if (prev[server_id]?.state === "connecting") {
                return { ...prev, [server_id]: { state: "idle" } };
              }
              return prev;
            });
            return;
          }

          const currentUrl = popup.location.href;
          if (currentUrl.includes("code=")) {
            clearInterval(intervalId);
            delete popupIntervalsRef.current[server_id];
            popup.close();

            const urlObj = new URL(currentUrl);
            const code = urlObj.searchParams.get("code");
            const returnedState = urlObj.searchParams.get("state");

            if (!code) {
              setConnectionStatus((prev) => ({
                ...prev,
                [server_id]: { state: "error", errorMessage: "No code in redirect URL." },
              }));
              return;
            }

            // Retrieve stored PKCE data
            const stored = sessionStorage.getItem(`byok_pkce_${server_id}`);
            if (!stored) {
              setConnectionStatus((prev) => ({
                ...prev,
                [server_id]: {
                  state: "error",
                  errorMessage: "Session storage lost PKCE data.",
                },
              }));
              return;
            }
            const { verifier: storedVerifier, state: storedState, redirectUri: storedRedirectUri } =
              JSON.parse(stored) as { verifier: string; state: string; redirectUri: string };

            if (returnedState !== storedState) {
              setConnectionStatus((prev) => ({
                ...prev,
                [server_id]: {
                  state: "error",
                  errorMessage: "State mismatch — possible CSRF.",
                },
              }));
              return;
            }

            // Exchange code for token
            try {
              const tokenBody = new URLSearchParams({
                grant_type: "authorization_code",
                code,
                redirect_uri: storedRedirectUri,
                code_verifier: storedVerifier,
                client_id: CLIENT_ID,
              });

              const tokenRes = await fetch(`${PROXY_BASE_URL}/v1/mcp/oauth/token`, {
                method: "POST",
                headers: {
                  "Content-Type": "application/x-www-form-urlencoded",
                  Authorization: `Bearer ${DEMO_VIRTUAL_KEY}`,
                },
                body: tokenBody.toString(),
              });

              if (!tokenRes.ok) {
                const errText = await tokenRes.text();
                throw new Error(`Token exchange failed (${tokenRes.status}): ${errText}`);
              }

              sessionStorage.removeItem(`byok_pkce_${server_id}`);

              setConnectionStatus((prev) => ({
                ...prev,
                [server_id]: { state: "connected" },
              }));

              // Refresh server list to reflect has_user_credential: true
              await fetchServers();

              setChatMessages((prev) => [
                ...prev,
                {
                  role: "assistant",
                  content: `Connected to ${server_name}! OAuth 2.1 PKCE flow completed. Your API key is securely stored.`,
                },
              ]);
            } catch (tokenErr: unknown) {
              const message = tokenErr instanceof Error ? tokenErr.message : String(tokenErr);
              setConnectionStatus((prev) => ({
                ...prev,
                [server_id]: { state: "error", errorMessage: message },
              }));
            }
          }
        } catch {
          // Cross-origin access — popup is on a different origin, ignore
        }
      }, 500);

      popupIntervalsRef.current[server_id] = intervalId;
    },
    [fetchServers]
  );

  // ---------------------------------------------------------------------------
  // Derived state
  // ---------------------------------------------------------------------------

  const byokServers = servers.filter((s) => s.is_byok);
  const regularServers = servers.filter((s) => !s.is_byok);

  const truncatedKey = `${DEMO_VIRTUAL_KEY.slice(0, 8)}...${DEMO_VIRTUAL_KEY.slice(-4)}`;

  // ---------------------------------------------------------------------------
  // Render helpers
  // ---------------------------------------------------------------------------

  function ServerItem({ server }: { server: McpServer }) {
    const connStatus = connectionStatus[server.server_id];
    const isConnecting = connStatus?.state === "connecting";
    const isConnected =
      connStatus?.state === "connected" || server.has_user_credential;
    const hasError = connStatus?.state === "error";

    return (
      <div
        className={`rounded-lg p-3 mb-2 border transition-colors ${
          isConnected
            ? "border-emerald-500/40 bg-emerald-900/20"
            : hasError
            ? "border-red-500/40 bg-red-900/10"
            : "border-slate-700 bg-slate-800/50"
        }`}
      >
        <div className="flex items-start gap-2">
          <div className="mt-0.5 flex-shrink-0">
            {server.is_byok ? (
              isConnected ? (
                <CheckIcon className="w-4 h-4 text-emerald-400" />
              ) : (
                <LockIcon className="w-4 h-4 text-amber-400" />
              )
            ) : (
              <ServerIcon className="w-4 h-4 text-slate-400" />
            )}
          </div>
          <div className="flex-1 min-w-0">
            <div className="text-sm font-medium text-slate-200 truncate">
              {server.server_name}
            </div>
            {server.description && (
              <div className="text-xs text-slate-500 mt-0.5 truncate">
                {server.description}
              </div>
            )}
            <div className="flex items-center gap-2 mt-1.5">
              {server.is_byok && (
                <span className="inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-medium bg-amber-900/40 text-amber-300 border border-amber-700/50">
                  BYOK
                </span>
              )}
              {isConnected && (
                <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] font-medium bg-emerald-900/40 text-emerald-300 border border-emerald-700/50">
                  <CheckIcon className="w-2.5 h-2.5" />
                  Connected
                </span>
              )}
              {hasError && (
                <span className="inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-medium bg-red-900/40 text-red-300 border border-red-700/50">
                  Error
                </span>
              )}
            </div>
            {hasError && connStatus?.errorMessage && (
              <div className="text-[11px] text-red-400 mt-1.5 leading-snug">
                {connStatus.errorMessage}
              </div>
            )}
          </div>
        </div>
        {server.is_byok && !isConnected && (
          <button
            onClick={() => handleConnect(server)}
            disabled={isConnecting}
            className={`mt-2.5 w-full flex items-center justify-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${
              isConnecting
                ? "bg-slate-700 text-slate-400 cursor-not-allowed"
                : "bg-indigo-600 hover:bg-indigo-500 text-white"
            }`}
          >
            {isConnecting ? (
              <>
                <SpinnerIcon className="w-3 h-3 animate-spin" />
                Connecting…
              </>
            ) : (
              <>
                <KeyIcon className="w-3 h-3" />
                Connect
              </>
            )}
          </button>
        )}
        {server.is_byok && isConnected && !hasError && (
          <button
            onClick={() => {
              setConnectionStatus((prev) => ({
                ...prev,
                [server.server_id]: { state: "idle" },
              }));
              handleConnect(server);
            }}
            className="mt-2.5 w-full flex items-center justify-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium text-slate-400 hover:text-slate-200 border border-slate-700 hover:border-slate-500 transition-colors"
          >
            Reconnect
          </button>
        )}
      </div>
    );
  }

  function ChatBubble({ message }: { message: ChatMessage }) {
    const isUser = message.role === "user";
    const isSystem = message.role === "system";

    if (isSystem) {
      return (
        <div className="flex justify-center mb-4">
          <div className="bg-slate-800 border border-slate-700 rounded-lg px-4 py-3 max-w-lg text-sm text-slate-400 text-center">
            {message.content}
          </div>
        </div>
      );
    }

    const isSuccess = message.content.startsWith("Connected to ");

    return (
      <div className={`flex mb-4 ${isUser ? "justify-end" : "justify-start"}`}>
        {!isUser && (
          <div className="w-7 h-7 rounded-full bg-indigo-600 flex items-center justify-center flex-shrink-0 mr-2 mt-0.5">
            <span className="text-xs font-bold text-white">L</span>
          </div>
        )}
        <div
          className={`rounded-2xl px-4 py-2.5 max-w-md text-sm leading-relaxed ${
            isUser
              ? "bg-indigo-600 text-white rounded-br-sm"
              : isSuccess
              ? "bg-emerald-900/40 border border-emerald-700/50 text-emerald-200 rounded-bl-sm"
              : "bg-slate-800 border border-slate-700 text-slate-200 rounded-bl-sm"
          }`}
        >
          {isSuccess && (
            <div className="flex items-center gap-1.5 mb-1.5">
              <span className="text-emerald-400 text-base">✅</span>
              <span className="text-emerald-300 font-medium text-xs uppercase tracking-wide">
                Connected
              </span>
            </div>
          )}
          {message.content}
        </div>
        {isUser && (
          <div className="w-7 h-7 rounded-full bg-slate-600 flex items-center justify-center flex-shrink-0 ml-2 mt-0.5">
            <span className="text-xs font-bold text-white">A</span>
          </div>
        )}
      </div>
    );
  }

  // ---------------------------------------------------------------------------
  // Render
  // ---------------------------------------------------------------------------

  return (
    <div className="flex h-screen bg-[#0f172a] text-slate-100 overflow-hidden">
      {/* ------------------------------------------------------------------ */}
      {/* Left sidebar                                                         */}
      {/* ------------------------------------------------------------------ */}
      <aside className="w-72 flex-shrink-0 flex flex-col border-r border-slate-800 bg-[#0d1526]">
        {/* Header */}
        <div className="px-4 pt-5 pb-4 border-b border-slate-800">
          <h2 className="text-xs font-semibold uppercase tracking-widest text-slate-500 mb-1">
            MCP Tools
          </h2>
          <p className="text-[11px] text-slate-600">
            via {PROXY_BASE_URL.replace(/https?:\/\//, "")}
          </p>
        </div>

        {/* Server list */}
        <div className="flex-1 overflow-y-auto px-3 py-3">
          {loadingServers ? (
            <div className="flex flex-col items-center justify-center gap-2 py-10 text-slate-600">
              <SpinnerIcon className="w-6 h-6 animate-spin" />
              <span className="text-xs">Loading servers…</span>
            </div>
          ) : fetchError ? (
            <div className="rounded-lg border border-red-800/50 bg-red-900/10 p-3">
              <div className="text-xs font-medium text-red-400 mb-1">
                Could not fetch servers
              </div>
              <div className="text-[11px] text-red-500 leading-snug">{fetchError}</div>
              <button
                onClick={fetchServers}
                className="mt-2 text-[11px] text-indigo-400 hover:text-indigo-300 underline"
              >
                Retry
              </button>
            </div>
          ) : (
            <>
              {byokServers.length > 0 && (
                <div className="mb-4">
                  <div className="text-[10px] font-semibold uppercase tracking-widest text-amber-500/80 mb-2 px-1">
                    Requires your key
                  </div>
                  {byokServers.map((s) => (
                    <ServerItem key={s.server_id} server={s} />
                  ))}
                </div>
              )}

              {regularServers.length > 0 && (
                <div className="mb-4">
                  <div className="text-[10px] font-semibold uppercase tracking-widest text-slate-500 mb-2 px-1">
                    Available
                  </div>
                  {regularServers.map((s) => (
                    <ServerItem key={s.server_id} server={s} />
                  ))}
                </div>
              )}

              {servers.length === 0 && (
                <div className="text-center py-10 text-slate-600 text-xs">
                  No MCP servers found.
                </div>
              )}
            </>
          )}
        </div>

        {/* Footer: virtual key info */}
        <div className="px-4 py-3 border-t border-slate-800">
          <div className="flex items-center gap-2">
            <KeyIcon className="w-3.5 h-3.5 text-slate-500 flex-shrink-0" />
            <span className="text-[11px] text-slate-500 font-mono truncate">
              {truncatedKey}
            </span>
          </div>
          <div className="text-[10px] text-slate-700 mt-0.5">Demo virtual key</div>
        </div>
      </aside>

      {/* ------------------------------------------------------------------ */}
      {/* Main content area                                                    */}
      {/* ------------------------------------------------------------------ */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Top bar */}
        <header className="flex items-center justify-between px-6 py-3.5 border-b border-slate-800 bg-[#0d1526] flex-shrink-0">
          <div className="flex items-center gap-3">
            <div className="w-7 h-7 rounded-lg bg-indigo-600 flex items-center justify-center">
              <span className="text-sm font-bold text-white">L</span>
            </div>
            <div>
              <h1 className="text-sm font-semibold text-slate-100 leading-tight">
                LiteLLM MCP Demo
              </h1>
              <p className="text-[11px] text-slate-500 leading-tight">
                External chat UI
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <span className="inline-flex items-center px-2.5 py-1 rounded-full text-[11px] font-medium bg-indigo-900/50 text-indigo-300 border border-indigo-700/50">
              BYOK OAuth Flow
            </span>
            <span className="inline-flex items-center px-2.5 py-1 rounded-full text-[11px] font-mono bg-slate-800 text-slate-400 border border-slate-700">
              {truncatedKey}
            </span>
          </div>
        </header>

        {/* Chat messages */}
        <div className="flex-1 overflow-y-auto px-6 py-6">
          {/* Explainer card */}
          <div className="mb-6 rounded-xl border border-slate-700 bg-slate-800/50 p-5 max-w-2xl mx-auto">
            <h3 className="text-sm font-semibold text-slate-200 mb-2">
              How this demo works
            </h3>
            <ol className="text-xs text-slate-400 space-y-1.5 list-none">
              {[
                "This page calls GET /v1/mcp/server to list available MCP servers.",
                "BYOK servers require you to supply your own API key — they show a lock icon.",
                'Click "Connect" to start the OAuth 2.1 PKCE authorization flow.',
                "A popup opens the LiteLLM authorization page where you enter your key.",
                "LiteLLM redirects back with an authorization code.",
                "This page exchanges the code for an access token (PKCE verified).",
                "Your key is now securely stored — no plain-text transmission to this page.",
              ].map((step, i) => (
                <li key={i} className="flex gap-2">
                  <span className="flex-shrink-0 w-4 h-4 rounded-full bg-indigo-900/60 border border-indigo-700/50 text-indigo-400 text-[9px] font-bold flex items-center justify-center mt-0.5">
                    {i + 1}
                  </span>
                  <span>{step}</span>
                </li>
              ))}
            </ol>
          </div>

          {/* Chat messages */}
          <div className="max-w-2xl mx-auto">
            {chatMessages.map((msg, idx) => (
              <ChatBubble key={idx} message={msg} />
            ))}
          </div>
        </div>

        {/* Chat input (UI only — no LLM call in this demo) */}
        <div className="px-6 py-4 border-t border-slate-800 flex-shrink-0">
          <div className="max-w-2xl mx-auto">
            <div className="flex gap-2">
              <input
                type="text"
                placeholder="Type a message… (demo — not connected to an LLM)"
                className="flex-1 bg-slate-800 border border-slate-700 rounded-xl px-4 py-2.5 text-sm text-slate-300 placeholder-slate-600 focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500/50 transition-colors"
                onKeyDown={(e) => {
                  if (e.key === "Enter") {
                    const input = e.currentTarget;
                    const value = input.value.trim();
                    if (!value) return;
                    setChatMessages((prev) => [
                      ...prev,
                      { role: "user", content: value },
                      {
                        role: "assistant",
                        content:
                          "This is a demo UI. Connect a BYOK server from the sidebar to enable real MCP tool calls.",
                      },
                    ]);
                    input.value = "";
                  }
                }}
              />
              <button
                className="px-4 py-2.5 bg-indigo-600 hover:bg-indigo-500 text-white text-sm font-medium rounded-xl transition-colors flex-shrink-0"
                onClick={(e) => {
                  const input = (e.currentTarget.previousSibling as HTMLInputElement);
                  const value = input?.value?.trim();
                  if (!value) return;
                  setChatMessages((prev) => [
                    ...prev,
                    { role: "user", content: value },
                    {
                      role: "assistant",
                      content:
                        "This is a demo UI. Connect a BYOK server from the sidebar to enable real MCP tool calls.",
                    },
                  ]);
                  input.value = "";
                }}
              >
                Send
              </button>
            </div>
            <p className="text-[11px] text-slate-700 mt-2 text-center">
              Demo only — chat responses are simulated. MCP tool calls require a connected server.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
