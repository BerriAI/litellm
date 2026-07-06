import React, { useCallback, useEffect, useState } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { ToolTestPanel } from "./ToolTestPanel";
import { resolveLogoSrc } from "@/lib/assetPaths";
import { MCPTool, MCPToolsViewerProps, MCPContent, CallMCPToolResponse, getMcpOAuthMode } from "./types";
import { listMCPTools, callMCPTool, getMCPOAuthUserCredentialStatus } from "../networking";
import { isTokenValid, getToken, removeToken } from "@/utils/mcpTokenStore";
import { sanitizeMcpAliasForHeader, buildMcpPassthroughAuthHeader } from "@/utils/mcpHeaderUtils";
import { useToolsOAuthFlow } from "@/hooks/useToolsOAuthFlow";
import { useUserMcpOAuthFlow } from "@/hooks/useUserMcpOAuthFlow";
import { TOOLS_OAUTH_UI_STATE_KEY } from "@/hooks/mcpOAuthUtils";
import { setSecureItem } from "@/utils/secureStorage";

import { Card, Title, Text } from "@tremor/react";
import { RobotOutlined, ToolOutlined, SearchOutlined, KeyOutlined, LockOutlined } from "@ant-design/icons";
import { Input, Button as AntdButton } from "antd";

const MCPToolsViewer = ({
  serverId,
  accessToken,
  auth_type,
  oauth2_flow,
  delegate_auth_to_upstream,
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
  const isAuthorizationCode = oauthMode === "authorization_code";
  const [oauthToken, setOauthToken] = useState<string | null>(() =>
    isPassthrough && isTokenValid(serverId, userID) ? getToken(serverId, userID)?.access_token ?? null : null,
  );

  // Re-sync token when serverId/userID changes (useState initializer only runs on mount).
  useEffect(() => {
    if (!isPassthrough) {
      setOauthToken(null);
      return;
    }
    setOauthToken(isTokenValid(serverId, userID) ? getToken(serverId, userID)?.access_token ?? null : null);
  }, [serverId, userID, isPassthrough]);

  const {
    startOAuthFlow,
    status: oauthStatus,
    error: oauthError,
  } = useToolsOAuthFlow({
    accessToken: accessToken ?? "",
    serverId,
    serverAlias,
    userId: userID,
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
    if (isPassthrough && oauthToken) {
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
      !!accessToken && (isPassthrough ? oauthToken !== null : isAuthorizationCode ? hasAuthorizationCodeCred : true),
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
  const authGateActive = (isPassthrough && !oauthToken) || authorizationCodeNeedsAuth || authorizationCodeTokenRejected;
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
    <div className="w-full h-screen p-4 bg-white">
      <Card className="w-full rounded-xl shadow-md overflow-hidden">
        <div className="flex h-auto w-full gap-4">
          {/* Left Sidebar with Controls */}
          <div className="w-1/4 p-4 bg-gray-50 flex flex-col">
            <Title className="text-xl font-semibold mb-6 mt-2">MCP Tools</Title>

            <div className="flex flex-col flex-1">
              {/* Extra Headers Input Section */}
              {hasExtraHeaders && (
                <div className="mb-4 p-3 bg-blue-50 border border-blue-200 rounded-lg">
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center">
                      <KeyOutlined className="text-blue-600 mr-2" />
                      <Text className="text-sm font-medium text-blue-800">Additional Headers</Text>
                    </div>
                    <AntdButton
                      size="small"
                      type="link"
                      onClick={() => setShowHeaderInput(!showHeaderInput)}
                      className="text-blue-700 p-0 h-auto"
                    >
                      {showHeaderInput ? "Hide" : "Configure"}
                    </AntdButton>
                  </div>

                  {!showHeaderInput && Object.keys(passthroughHeaders).length === 0 && (
                    <Text className="text-xs text-blue-700">
                      This server requires additional headers. Click &quot;Configure&quot; to provide values.
                    </Text>
                  )}

                  {showHeaderInput && (
                    <div className="mt-3 space-y-2">
                      {extraHeaders?.map((headerName) => (
                        <div key={headerName}>
                          <label className="block text-xs font-medium text-gray-700 mb-1">{headerName}</label>
                          <Input
                            size="small"
                            placeholder={`Enter ${headerName}`}
                            value={passthroughHeaders[headerName] || ""}
                            onChange={(e) => {
                              setPassthroughHeaders({
                                ...passthroughHeaders,
                                [headerName]: e.target.value,
                              });
                            }}
                            prefix={<KeyOutlined className="text-gray-400" />}
                            className="rounded-sm"
                          />
                        </div>
                      ))}
                      <AntdButton
                        size="small"
                        type="primary"
                        onClick={() => {
                          refetchTools();
                          setShowHeaderInput(false);
                        }}
                        disabled={Object.values(passthroughHeaders).every((v) => !v || !v.trim())}
                        className="w-full mt-2"
                      >
                        Load Tools
                      </AntdButton>
                    </div>
                  )}

                  {!showHeaderInput && Object.keys(passthroughHeaders).length > 0 && (
                    <div className="mt-2">
                      <Text className="text-xs text-green-700 flex items-center">
                        <span className="inline-block w-2 h-2 bg-green-500 rounded-full mr-2"></span>
                        {Object.keys(passthroughHeaders).length} header(s) configured
                      </Text>
                    </div>
                  )}
                </div>
              )}

              {/* Tool Selection - Show tools first */}
              <div className="flex flex-col flex-1 min-h-0">
                <Text className="font-medium block mb-3 text-gray-700 flex items-center">
                  <ToolOutlined className="mr-2" /> Available Tools
                  {toolsData.length > 0 && (
                    <span className="ml-2 bg-blue-100 text-blue-800 text-xs font-medium px-2 py-0.5 rounded-full">
                      {toolsData.length}
                    </span>
                  )}
                </Text>

                {/* Passthrough auth gate — browser session token absent */}
                {isPassthrough && !oauthToken && (
                  <div className="p-4 text-center bg-white border border-gray-200 rounded-lg">
                    <LockOutlined className="text-2xl text-gray-400 mb-2" />
                    <p className="text-xs font-medium text-gray-700 mb-1">Authentication required</p>
                    <p className="text-xs text-gray-500 mb-3">Authenticate to view available tools</p>
                    <AntdButton
                      size="small"
                      type="primary"
                      loading={oauthStatus === "authorizing" || oauthStatus === "exchanging"}
                      onClick={startOAuthFlow}
                      disabled={!accessToken}
                    >
                      Authorize
                    </AntdButton>
                    {oauthError && <p className="text-xs text-red-500 mt-2">{oauthError}</p>}
                  </div>
                )}

                {/* Auth gate (authorization_code or token_exchange) — shown when there is no credential
                    row for this user, or when the list call returns 401 (no valid token and the
                    server-side refresh could not mint one, e.g. an expired token
                    with no usable refresh token). A refreshable token is refreshed
                    on the list call and never trips this gate. */}
                {(authorizationCodeNeedsAuth || authorizationCodeTokenRejected) && (
                  <div className="p-4 text-center bg-white border border-gray-200 rounded-lg">
                    <LockOutlined className="text-2xl text-gray-400 mb-2" />
                    <p className="text-xs font-medium text-gray-700 mb-1">Authentication required</p>
                    <p className="text-xs text-gray-500 mb-3">
                      Authenticate with the upstream provider to view available tools
                    </p>
                    <AntdButton
                      size="small"
                      type="primary"
                      loading={dbOAuthStatus === "authorizing" || dbOAuthStatus === "exchanging"}
                      onClick={startAuthorizationCodeAuthorize}
                      disabled={!accessToken}
                    >
                      Authorize
                    </AntdButton>
                    {dbOAuthError && <p className="text-xs text-red-500 mt-2">{dbOAuthError}</p>}
                  </div>
                )}

                {/* Search Bar — only shown when tools are loaded */}
                {!authGateActive ? (
                  <>
                    {toolsData.length > 0 && (
                      <div className="mb-3">
                        <Input
                          placeholder="Search tools..."
                          prefix={<SearchOutlined className="text-gray-400" />}
                          value={toolSearchTerm}
                          onChange={(e) => setToolSearchTerm(e.target.value)}
                          allowClear
                          className="rounded-lg"
                          size="middle"
                        />
                      </div>
                    )}

                    {/* Loading State */}
                    {toolsAreaLoading && (
                      <div className="flex flex-col items-center justify-center py-8 bg-white border border-gray-200 rounded-lg">
                        <div className="relative mb-3">
                          <div className="animate-spin rounded-full h-6 w-6 border-2 border-gray-200"></div>
                          <div className="animate-spin rounded-full h-6 w-6 border-2 border-blue-600 border-t-transparent absolute top-0"></div>
                        </div>
                        <p className="text-xs font-medium text-gray-700">Loading tools...</p>
                      </div>
                    )}

                    {/* Error State */}
                    {(mcpToolsResponse?.error || mcpToolsError) && !toolsAreaLoading && !toolsData.length && (
                      <div className="p-3 text-xs text-red-800 rounded-lg bg-red-50 border border-red-200">
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
                        <div className="p-4 text-center bg-white border border-gray-200 rounded-lg">
                          <div className="mx-auto w-8 h-8 bg-gray-200 rounded-full flex items-center justify-center mb-2">
                            <svg
                              className="w-4 h-4 text-gray-400"
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
                          <p className="text-xs font-medium text-gray-700 mb-1">No tools available</p>
                          <p className="text-xs text-gray-500">No tools found for this server</p>
                        </div>
                      )}

                    {/* Tools List */}
                    {!toolsAreaLoading && !mcpToolsResponse?.error && toolsData.length > 0 && (
                      <>
                        {filteredTools.length === 0 ? (
                          <div className="p-4 text-center bg-white border border-gray-200 rounded-lg">
                            <SearchOutlined className="text-2xl text-gray-400 mb-2" />
                            <p className="text-xs font-medium text-gray-700 mb-1">No tools found</p>
                            <p className="text-xs text-gray-500">No tools match &quot;{toolSearchTerm}&quot;</p>
                          </div>
                        ) : (
                          <div
                            className="space-y-2 flex-1 overflow-y-auto min-h-0 mcp-tools-scrollable"
                            style={{
                              maxHeight: "400px",
                              scrollbarWidth: "auto",
                              scrollbarColor: "#cbd5e0 #f7fafc",
                            }}
                          >
                            {filteredTools.map((tool: MCPTool) => (
                              <div
                                key={tool.name}
                                className={`border rounded-lg p-3 cursor-pointer transition-all hover:shadow-xs ${
                                  selectedTool?.name === tool.name
                                    ? "border-blue-500 bg-blue-50 ring-1 ring-blue-200"
                                    : "border-gray-200 bg-white hover:border-gray-300"
                                }`}
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
                                    <h4 className="font-mono text-xs font-medium text-gray-900 truncate">
                                      {tool.name}
                                    </h4>
                                    <p className="text-xs text-gray-500 truncate">{tool.mcp_info.server_name}</p>
                                    <p className="text-xs text-gray-600 mt-1 line-clamp-2 leading-relaxed">
                                      {tool.description}
                                    </p>
                                  </div>
                                </div>
                                {selectedTool?.name === tool.name && (
                                  <div className="mt-2 pt-2 border-t border-blue-200">
                                    <div className="flex items-center text-xs font-medium text-blue-700">
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
          <div className="w-3/4 flex flex-col bg-white">
            <div className="p-4 border-b border-gray-200 flex justify-between items-center">
              <Title className="text-xl font-semibold mb-0">Tool Testing Playground</Title>
            </div>

            <div className="flex-1 overflow-auto p-4">
              {!selectedTool ? (
                /* Empty State */
                <div className="h-full flex flex-col items-center justify-center text-gray-400">
                  <RobotOutlined style={{ fontSize: "48px", marginBottom: "16px" }} />
                  <Text className="text-lg font-medium text-gray-600 mb-2">Select a Tool to Test</Text>
                  <Text className="text-center text-gray-500 max-w-md">
                    Choose a tool from the left sidebar to start testing its functionality with custom inputs.
                  </Text>
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
