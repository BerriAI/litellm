import React, { useCallback, useEffect, useState } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { ToolTestPanel } from "./ToolTestPanel";
import { resolveLogoSrc } from "@/lib/assetPaths";
import {
  isClientForwardedTokenMode,
  gatewayMintsClientFor,
  MCPTool,
  MCPToolsViewerProps,
  MCPContent,
  CallMCPToolResponse,
  getMcpOAuthMode,
} from "@/components/mcp_tools/types";
import { listMCPTools, callMCPTool, getMCPOAuthUserCredentialStatus } from "@/components/networking";
import { isTokenValid, getToken, removeToken } from "@/utils/mcpTokenStore";
import { sanitizeMcpAliasForHeader, buildMcpPassthroughAuthHeader } from "@/utils/mcpHeaderUtils";
import { useToolsOAuthFlow } from "@/hooks/useToolsOAuthFlow";
import { useUserMcpOAuthFlow } from "@/hooks/useUserMcpOAuthFlow";
import { TOOLS_OAUTH_UI_STATE_KEY } from "@/hooks/mcpOAuthUtils";
import { setSecureItem } from "@/utils/secureStorage";

import { Bot, Wrench, Search, Key, Lock } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { InputGroup, InputGroupAddon, InputGroupInput } from "@/components/ui/input-group";
import { UiLoadingSpinner } from "@/components/ui/ui-loading-spinner";
import { cn } from "@/lib/cva.config";

const MCPToolsViewer = ({
  serverId,
  accessToken,
  auth_type,
  oauth2_flow,
  delegate_auth_to_upstream,
  dcr_bridge,
  userRole,
  userID,
  serverAlias,
  extraHeaders,
}: MCPToolsViewerProps) => {
  const [selectedTool, setSelectedTool] = useState<MCPTool | null>(null);
  const [toolResult, setToolResult] = useState<MCPContent[] | null>(null);
  const [toolError, setToolError] = useState<Error | null>(null);
  const [toolSearchTerm, setToolSearchTerm] = useState("");

  // State for passthrough headers
  const [passthroughHeaders, setPassthroughHeaders] = useState<Record<string, string>>({});
  const [showHeaderInput, setShowHeaderInput] = useState(false);

  // PKCE passthrough holds a browser-side session token (sessionStorage) and
  // gates tool listing behind it. authorization_code uses a backend-stored
  // per-user token that the user must establish once via an interactive login;
  // we gate on whether that DB credential exists. M2M uses the backend's own
  // service token and needs no gate.
  const oauthMode = getMcpOAuthMode({ auth_type, oauth2_flow, delegate_auth_to_upstream });
  const isPassthrough = oauthMode === "passthrough";
  // The client-forwarded token modes gate the same way as PKCE passthrough: the
  // browser session token (established via the browser-only Authorize in the
  // create/edit forms, or right here) is the upstream credential.
  const usesBrowserHeldToken = isPassthrough || isClientForwardedTokenMode(auth_type);
  const isAuthorizationCode = oauthMode === "authorization_code";
  const [oauthToken, setOauthToken] = useState<string | null>(() =>
    usesBrowserHeldToken && isTokenValid(serverId, userID) ? getToken(serverId, userID)?.access_token ?? null : null,
  );

  // Re-sync token when serverId/userID changes (useState initializer only runs on mount).
  useEffect(() => {
    if (!usesBrowserHeldToken) {
      setOauthToken(null);
      return;
    }
    setOauthToken(isTokenValid(serverId, userID) ? getToken(serverId, userID)?.access_token ?? null : null);
  }, [serverId, userID, usesBrowserHeldToken]);

  const {
    startOAuthFlow,
    status: oauthStatus,
    error: oauthError,
  } = useToolsOAuthFlow({
    accessToken: accessToken ?? "",
    serverId,
    serverAlias,
    userId: userID,
    gatewayMintsClient: gatewayMintsClientFor({ auth_type, dcr_bridge }),
    onSuccess: setOauthToken,
  });

  // authorization_code servers list tools using a per-user token the backend stores in the DB;
  // check whether the current user has a valid one so we can prompt them to
  // authorize when they don't (otherwise the backend silently returns no tools).
  const {
    data: authorizationCodeCredStatus,
    isLoading: isLoadingAuthorizationCodeCred,
    isError: isAuthorizationCodeCredError,
    refetch: refetchAuthorizationCodeCred,
  } = useQuery({
    queryKey: ["mcpOauthUserCredStatus", serverId, userID],
    queryFn: () => getMCPOAuthUserCredentialStatus(accessToken ?? "", serverId),
    enabled: !!accessToken && isAuthorizationCode,
    staleTime: 30000,
  });

  // A stored credential is sufficient: the backend proactively refreshes an
  // expired or near-expiry token from the stored refresh_token on the next list
  // call, so the user only needs to authorize when no credential row exists. If
  // the status check itself fails we can't confirm a credential, so surface the
  // Authorize gate rather than a silent empty tool list; re-authorizing only
  // overwrites the user's own row, so it is safe when a credential did exist.
  const hasAuthorizationCodeCred = !!authorizationCodeCredStatus?.has_credential;
  const authorizationCodeNeedsAuth =
    isAuthorizationCode &&
    !isLoadingAuthorizationCodeCred &&
    (isAuthorizationCodeCredError || (!!authorizationCodeCredStatus && !hasAuthorizationCodeCred));
  const authorizationCodeStatusLoading = isAuthorizationCode && isLoadingAuthorizationCodeCred;

  // Check if this server has extra headers configured
  const hasExtraHeaders = extraHeaders && extraHeaders.length > 0;

  // Build custom headers for MCP server requests
  const buildCustomHeaders = () => {
    const customHeaders: Record<string, string> = {};

    // Include the session OAuth token using MCP-specific headers so it doesn't
    // conflict with the Authorization header used by the LiteLLM proxy itself.
    // The backend's _get_mcp_server_auth_headers_from_headers() picks up the
    // x-mcp-{alias}-{header} pattern and forwards it to the upstream MCP server.
    // When no alias is available, fall back to x-mcp-auth (legacy but still supported).
    // Passthrough only: authorization_code/token_exchange/M2M tokens are attached server-side, not from the browser.
    if (usesBrowserHeldToken && oauthToken) {
      Object.assign(customHeaders, buildMcpPassthroughAuthHeader(serverAlias, oauthToken));
    }

    // Add passthrough headers with server-specific prefix
    if (serverAlias && hasExtraHeaders) {
      const safeAlias = sanitizeMcpAliasForHeader(serverAlias);
      if (safeAlias) {
        Object.entries(passthroughHeaders).forEach(([headerName, headerValue]) => {
          if (headerValue && headerValue.trim()) {
            // Format: x-mcp-{alias}-{header_name}
            const mcpHeaderName = `x-mcp-${safeAlias}-${headerName.toLowerCase()}`;
            customHeaders[mcpHeaderName] = headerValue;
          }
        });
      }
    }

    return Object.keys(customHeaders).length > 0 ? customHeaders : undefined;
  };

  // Query to fetch MCP tools
  const {
    data: mcpToolsResponse,
    isLoading: isLoadingTools,
    error: mcpToolsError,
    refetch: refetchTools,
  } = useQuery({
    queryKey: ["mcpTools", serverId, passthroughHeaders, oauthToken],
    queryFn: async () => {
      if (!accessToken) throw new Error("Access Token required");
      const result = await listMCPTools(accessToken, serverId, buildCustomHeaders());
      // listMCPTools never throws — surface error responses as thrown errors
      // here so useQuery's retry/onError can react (e.g. clear the cached
      // OAuth token on 401).
      if (result?.error) {
        const status = (result as { status?: number }).status;
        if (status === 401) {
          removeToken(serverId, userID);
        }
        const enhancedError = new Error(result.message || result.error || "Failed to fetch MCP tools") as Error & {
          status?: number;
          statusText?: string;
          details?: any;
        };
        enhancedError.status = status;
        enhancedError.statusText = (result as any).statusText;
        enhancedError.details = (result as any).details;
        throw enhancedError;
      }
      return result;
    },
    // Passthrough blocks until a browser session token exists; authorization_code blocks until
    // the user has a valid DB credential (else the backend returns no tools).
    enabled:
      !!accessToken &&
      (usesBrowserHeldToken ? oauthToken !== null : isAuthorizationCode ? hasAuthorizationCodeCred : true),
    staleTime: 30000, // Consider data fresh for 30 seconds
    retry: (failureCount, error: any) => {
      // Don't retry on 401 — token is invalid, user must re-authenticate
      if (error?.status === 401 || error?.response?.status === 401) return false;
      return failureCount < 2;
    },
  });

  // authorization_code authorize: same redirect+exchange flow as the admin "Authorize & Fetch"
  // and the chat "Connect" button, but persists the token to the per-user DB.
  const onAuthorizationCodeAuthSuccess = useCallback(() => {
    refetchAuthorizationCodeCred();
    refetchTools();
  }, [refetchAuthorizationCodeCred, refetchTools]);

  const {
    startOAuthFlow: startDbOAuthFlow,
    status: dbOAuthStatus,
    error: dbOAuthError,
  } = useUserMcpOAuthFlow({
    accessToken: accessToken ?? "",
    serverId,
    serverAlias,
    onSuccess: onAuthorizationCodeAuthSuccess,
  });

  // Stash which server started the redirect so the MCP Servers page can reopen
  // this Tools tab on return and let the flow resume to persist the credential.
  const startAuthorizationCodeAuthorize = useCallback(() => {
    try {
      setSecureItem(TOOLS_OAUTH_UI_STATE_KEY, JSON.stringify({ serverId }));
    } catch (_) {}
    startDbOAuthFlow();
  }, [serverId, startDbOAuthFlow]);

  // If the tools query fails with 401, the cached OAuth token is invalid —
  // clear it so the auth gate is shown again and the user can re-authenticate.
  useEffect(() => {
    const err = mcpToolsError as (Error & { status?: number; response?: { status?: number } }) | null;
    const status = err?.status ?? err?.response?.status;
    if (status === 401) {
      removeToken(serverId, userID);
      setOauthToken(null);
    }
  }, [mcpToolsError, serverId, userID]);

  // Mutation for calling a tool
  const { mutate: executeTool, isPending: isCallingTool } = useMutation({
    mutationFn: async (args: { tool: MCPTool; arguments: Record<string, any> }) => {
      if (!accessToken) throw new Error("Access Token required");

      try {
        const result: CallMCPToolResponse = await callMCPTool(accessToken, serverId, args.tool.name, args.arguments, {
          customHeaders: buildCustomHeaders(),
        });
        return result;
      } catch (error) {
        throw error;
      }
    },
    onSuccess: (data) => {
      setToolResult(data.content);
      setToolError(null);
    },
    onError: (error: Error & { status?: number; response?: { status?: number } }) => {
      setToolError(error);
      setToolResult(null);
      // On 401, clear the cached token so the auth gate is shown again
      if (error?.status === 401 || (error as any)?.response?.status === 401) {
        removeToken(serverId, userID);
        setOauthToken(null);
      }
    },
  });

  const toolsData = mcpToolsResponse?.tools || [];

  const toolsError = mcpToolsError as (Error & { status?: number; response?: { status?: number } }) | null;
  // authorization_code only: a 401 from the list call means the stored credential is unusable and
  // the backend's refresh could not mint a token, so the user must re-authorize (the browser flow).
  // token_exchange has no gateway-side authorize step, so it is not gated here.
  const authorizationCodeTokenRejected =
    isAuthorizationCode && (toolsError?.status ?? toolsError?.response?.status) === 401;

  // An auth gate replaces the tool list when the user must authenticate first:
  // passthrough needs a browser token; authorization_code needs a stored DB credential or a
  // still-valid one — a 401 from the list call means the backend has none even
  // after attempting a refresh, so re-authorization is required.
  const authGateActive =
    (usesBrowserHeldToken && !oauthToken) || authorizationCodeNeedsAuth || authorizationCodeTokenRejected;
  // Treat authorization_code credential-status loading as "tools loading" so the empty state
  // doesn't flash before we know whether the user needs to authorize.
  const toolsAreaLoading = isLoadingTools || authorizationCodeStatusLoading;

  // Filter tools based on search term
  const filteredTools = toolsData.filter((tool: MCPTool) => {
    const searchLower = toolSearchTerm.toLowerCase();
    return (
      tool.name.toLowerCase().includes(searchLower) ||
      (tool.description && tool.description.toLowerCase().includes(searchLower)) ||
      (tool.mcp_info.server_name && tool.mcp_info.server_name.toLowerCase().includes(searchLower))
    );
  });

  return (
    <div className="w-full p-4">
      <Card className="w-full overflow-hidden rounded-xl shadow-md">
        <div className="grid h-auto w-full grid-cols-4 gap-4">
          {/* Left Sidebar with Controls */}
          <div className="col-span-1 flex flex-col bg-muted p-4">
            <h2 className="mt-2 mb-6 text-xl font-semibold">MCP Tools</h2>

            <div className="flex flex-col flex-1">
              {/* Extra Headers Input Section */}
              {hasExtraHeaders && (
                <div className="mb-4 rounded-lg border border-border bg-card p-3">
                  <div className="mb-2 flex items-center justify-between">
                    <div className="flex items-center">
                      <Key className="mr-2 size-4 text-muted-foreground" />
                      <p className="text-sm font-medium">Additional Headers</p>
                    </div>
                    <Button variant="link" size="sm" onClick={() => setShowHeaderInput(!showHeaderInput)}>
                      {showHeaderInput ? "Hide" : "Configure"}
                    </Button>
                  </div>

                  {!showHeaderInput && Object.keys(passthroughHeaders).length === 0 && (
                    <p className="text-xs text-muted-foreground">
                      This server requires additional headers. Click &quot;Configure&quot; to provide values.
                    </p>
                  )}

                  {showHeaderInput && (
                    <div className="mt-3 space-y-2">
                      {extraHeaders?.map((headerName) => (
                        <div key={headerName}>
                          <label className="mb-1 block text-xs font-medium">{headerName}</label>
                          <InputGroup className="w-full">
                            <InputGroupAddon>
                              <Key className="size-4 text-muted-foreground" />
                            </InputGroupAddon>
                            <InputGroupInput
                              placeholder={`Enter ${headerName}`}
                              value={passthroughHeaders[headerName] || ""}
                              onChange={(e) => {
                                setPassthroughHeaders({
                                  ...passthroughHeaders,
                                  [headerName]: e.target.value,
                                });
                              }}
                            />
                          </InputGroup>
                        </div>
                      ))}
                      <Button
                        size="sm"
                        onClick={() => {
                          refetchTools();
                          setShowHeaderInput(false);
                        }}
                        disabled={Object.values(passthroughHeaders).every((v) => !v || !v.trim())}
                        className="mt-2 w-full"
                      >
                        Load Tools
                      </Button>
                    </div>
                  )}

                  {!showHeaderInput && Object.keys(passthroughHeaders).length > 0 && (
                    <div className="mt-2">
                      <p className="flex items-center text-xs text-muted-foreground">
                        <span className="mr-2 inline-block size-2 rounded-full bg-green-500" />
                        {Object.keys(passthroughHeaders).length} header(s) configured
                      </p>
                    </div>
                  )}
                </div>
              )}

              {/* Tool Selection - Show tools first */}
              <div className="flex flex-col flex-1 min-h-0">
                <p className="mb-3 flex items-center text-sm font-medium">
                  <Wrench className="mr-2 size-4" /> Available Tools
                  {toolsData.length > 0 && (
                    <Badge variant="secondary" className="ml-2">
                      {toolsData.length}
                    </Badge>
                  )}
                </p>

                {/* Passthrough auth gate — browser session token absent */}
                {usesBrowserHeldToken && !oauthToken && (
                  <div className="rounded-lg border border-border bg-card p-4 text-center">
                    <Lock className="mx-auto mb-2 size-6 text-muted-foreground" />
                    <p className="mb-1 text-xs font-medium">Authentication required</p>
                    <p className="mb-3 text-xs text-muted-foreground">Authenticate to view available tools</p>
                    <Button
                      size="sm"
                      onClick={startOAuthFlow}
                      disabled={!accessToken || oauthStatus === "authorizing" || oauthStatus === "exchanging"}
                    >
                      Authorize
                    </Button>
                    {oauthError && <p className="mt-2 text-xs text-destructive">{oauthError}</p>}
                  </div>
                )}

                {/* Auth gate (authorization_code or token_exchange) — shown when there is no credential
                    row for this user, or when the list call returns 401 (no valid token and the
                    server-side refresh could not mint one, e.g. an expired token
                    with no usable refresh token). A refreshable token is refreshed
                    on the list call and never trips this gate. */}
                {(authorizationCodeNeedsAuth || authorizationCodeTokenRejected) && (
                  <div className="rounded-lg border border-border bg-card p-4 text-center">
                    <Lock className="mx-auto mb-2 size-6 text-muted-foreground" />
                    <p className="mb-1 text-xs font-medium">Authentication required</p>
                    <p className="mb-3 text-xs text-muted-foreground">
                      Authenticate with the upstream provider to view available tools
                    </p>
                    <Button
                      size="sm"
                      onClick={startAuthorizationCodeAuthorize}
                      disabled={!accessToken || dbOAuthStatus === "authorizing" || dbOAuthStatus === "exchanging"}
                    >
                      Authorize
                    </Button>
                    {dbOAuthError && <p className="mt-2 text-xs text-destructive">{dbOAuthError}</p>}
                  </div>
                )}

                {/* Search Bar — only shown when tools are loaded */}
                {!authGateActive ? (
                  <>
                    {toolsData.length > 0 && (
                      <div className="mb-3">
                        <InputGroup className="w-full">
                          <InputGroupAddon>
                            <Search className="size-4 text-muted-foreground" />
                          </InputGroupAddon>
                          <InputGroupInput
                            placeholder="Search tools..."
                            value={toolSearchTerm}
                            onChange={(e) => setToolSearchTerm(e.target.value)}
                          />
                        </InputGroup>
                      </div>
                    )}

                    {/* Loading State */}
                    {toolsAreaLoading && (
                      <div className="flex flex-col items-center justify-center rounded-lg border border-border bg-card py-8">
                        <UiLoadingSpinner className="mb-3 size-6 text-muted-foreground" />
                        <p className="text-xs font-medium">Loading tools...</p>
                      </div>
                    )}

                    {/* Error State */}
                    {(mcpToolsResponse?.error || mcpToolsError) && !toolsAreaLoading && !toolsData.length && (
                      <div className="rounded-lg border border-destructive/40 bg-destructive/5 p-3 text-xs text-destructive">
                        <p className="font-medium">
                          Error: {mcpToolsResponse?.message || (mcpToolsError as Error)?.message}
                        </p>
                      </div>
                    )}

                    {/* No Tools State */}
                    {!toolsAreaLoading &&
                      !mcpToolsResponse?.error &&
                      !mcpToolsError &&
                      (!toolsData || toolsData.length === 0) && (
                        <div className="rounded-lg border border-border bg-card p-4 text-center">
                          <div className="mx-auto mb-2 flex size-8 items-center justify-center rounded-full bg-muted">
                            <svg
                              className="size-4 text-muted-foreground"
                              fill="none"
                              stroke="currentColor"
                              viewBox="0 0 24 24"
                            >
                              <path
                                strokeLinecap="round"
                                strokeLinejoin="round"
                                strokeWidth={2}
                                d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 8.172V5L8 4z"
                              />
                            </svg>
                          </div>
                          <p className="mb-1 text-xs font-medium">No tools available</p>
                          <p className="text-xs text-muted-foreground">No tools found for this server</p>
                        </div>
                      )}

                    {/* Tools List */}
                    {!toolsAreaLoading && !mcpToolsResponse?.error && toolsData.length > 0 && (
                      <>
                        {filteredTools.length === 0 ? (
                          <div className="rounded-lg border border-border bg-card p-4 text-center">
                            <Search className="mx-auto mb-2 size-6 text-muted-foreground" />
                            <p className="mb-1 text-xs font-medium">No tools found</p>
                            <p className="text-xs text-muted-foreground">No tools match &quot;{toolSearchTerm}&quot;</p>
                          </div>
                        ) : (
                          <div className="mcp-tools-scrollable max-h-100 min-h-0 flex-1 space-y-2 overflow-y-auto">
                            {filteredTools.map((tool: MCPTool) => (
                              <div
                                key={tool.name}
                                className={cn(
                                  "cursor-pointer rounded-lg border p-3 transition-all hover:shadow-xs",
                                  selectedTool?.name === tool.name
                                    ? "border-primary bg-accent ring-1 ring-ring"
                                    : "border-border bg-card",
                                )}
                                onClick={() => {
                                  setSelectedTool(tool);
                                  setToolResult(null);
                                  setToolError(null);
                                }}
                              >
                                <div className="flex items-start space-x-2">
                                  {tool.mcp_info.logo_url && (
                                    <img
                                      src={resolveLogoSrc(tool.mcp_info.logo_url)}
                                      alt={`${tool.mcp_info.server_name} logo`}
                                      className="w-4 h-4 object-contain shrink-0 mt-0.5"
                                    />
                                  )}
                                  <div className="flex-1 min-w-0">
                                    <h4 className="truncate font-mono text-xs font-medium">{tool.name}</h4>
                                    <p className="truncate text-xs text-muted-foreground">
                                      {tool.mcp_info.server_name}
                                    </p>
                                    <p className="mt-1 line-clamp-2 text-xs leading-relaxed text-muted-foreground">
                                      {tool.description}
                                    </p>
                                  </div>
                                </div>
                                {selectedTool?.name === tool.name && (
                                  <div className="mt-2 border-t border-border pt-2">
                                    <div className="flex items-center text-xs font-medium text-primary">
                                      <svg className="w-3 h-3 mr-1" fill="currentColor" viewBox="0 0 20 20">
                                        <path
                                          fillRule="evenodd"
                                          d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
                                          clipRule="evenodd"
                                        />
                                      </svg>
                                      Selected
                                    </div>
                                  </div>
                                )}
                              </div>
                            ))}
                          </div>
                        )}
                      </>
                    )}
                  </>
                ) : null}
              </div>
            </div>
          </div>

          {/* Main Testing Area */}
          <div className="col-span-3 flex flex-col">
            <div className="flex items-center justify-between border-b border-border p-4">
              <h2 className="mb-0 text-xl font-semibold">Tool Testing Playground</h2>
            </div>

            <div className="flex-1 overflow-auto p-4">
              {!selectedTool ? (
                /* Empty State */
                <div className="flex h-full flex-col items-center justify-center text-muted-foreground">
                  <Bot className="mb-4 size-12" />
                  <p className="mb-2 text-lg font-medium">Select a Tool to Test</p>
                  <p className="max-w-md text-center text-sm">
                    Choose a tool from the left sidebar to start testing its functionality with custom inputs.
                  </p>
                </div>
              ) : (
                /* Tool Test Panel */
                <div className="h-full">
                  <ToolTestPanel
                    tool={selectedTool}
                    onSubmit={(args) => {
                      executeTool({ tool: selectedTool, arguments: args });
                    }}
                    result={toolResult}
                    error={toolError}
                    isLoading={isCallingTool}
                    onClose={() => setSelectedTool(null)}
                  />
                </div>
              )}
            </div>
          </div>
        </div>
      </Card>
    </div>
  );
};

export default MCPToolsViewer;
