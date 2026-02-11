import { useState, useEffect } from "react";
import { testMCPToolsListRequest } from "../components/networking";
import { AUTH_TYPE, OAUTH_FLOW } from "@/components/mcp_tools/types";

interface MCPServerConfig {
  server_id?: string;
  server_name?: string;
  url?: string;
  transport?: string;
  auth_type?: string;
  authorization_url?: string;
  token_url?: string;
  registration_url?: string;
  mcp_info?: any;
  static_headers?: Record<string, string>;
  credentials?: {
    auth_value?: string;
    client_id?: string;
    client_secret?: string;
    scopes?: string[];
  };
}

interface UseTestMCPConnectionProps {
  accessToken: string | null;
  oauthAccessToken?: string | null;
  formValues: Record<string, any>;
  enabled?: boolean; // Optional flag to enable/disable auto-fetching
}

interface UseTestMCPConnectionReturn {
  tools: any[];
  isLoadingTools: boolean;
  toolsError: string | null;
  toolsErrorStackTrace: string | null;
  hasShownSuccessMessage: boolean;
  canFetchTools: boolean;
  fetchTools: () => Promise<void>;
  clearTools: () => void;
}

export const useTestMCPConnection = ({
  accessToken,
  oauthAccessToken,
  formValues,
  enabled = true,
}: UseTestMCPConnectionProps): UseTestMCPConnectionReturn => {
  const [tools, setTools] = useState<any[]>([]);
  const [isLoadingTools, setIsLoadingTools] = useState(false);
  const [toolsError, setToolsError] = useState<string | null>(null);
  const [toolsErrorStackTrace, setToolsErrorStackTrace] = useState<string | null>(null);
  const [hasShownSuccessMessage, setHasShownSuccessMessage] = useState(false);

  // Check if we have the minimum required fields to fetch tools
  const isM2MOAuth = formValues.auth_type === AUTH_TYPE.OAUTH2
    && formValues.oauth_flow_type === OAUTH_FLOW.M2M;
  const requiresOAuthToken = formValues.auth_type === AUTH_TYPE.OAUTH2 && !isM2MOAuth;
  const canFetchTools = !!(
    formValues.url &&
    formValues.transport &&
    formValues.auth_type &&
    accessToken &&
    (!requiresOAuthToken || oauthAccessToken)
  );

  const staticHeadersKey = JSON.stringify(formValues.static_headers ?? {});
  const credentialsKey = JSON.stringify(formValues.credentials ?? {});

  const fetchTools = async () => {
    if (!accessToken || !formValues.url) {
      return;
    }

    if (requiresOAuthToken && !oauthAccessToken) {
      return;
    }

    setIsLoadingTools(true);
    setToolsError(null);

    try {
      // Prepare the MCP server config from form values
      const staticHeaders = Array.isArray(formValues.static_headers)
        ? formValues.static_headers.reduce((acc: Record<string, string>, entry: Record<string, string>) => {
            const header = entry?.header?.trim();
            if (!header) {
              return acc;
            }
            acc[header] = entry?.value != null ? String(entry.value) : "";
            return acc;
          }, {})
        : !Array.isArray(formValues.static_headers) && formValues.static_headers && typeof formValues.static_headers === "object"
          ? Object.entries(formValues.static_headers).reduce(
              (acc: Record<string, string>, [header, value]) => {
                if (!header) {
                  return acc;
                }
                acc[header] = value != null ? String(value) : "";
                return acc;
              },
              {},
            )
          : {} as Record<string, string>;

      const credentials =
        formValues.credentials && typeof formValues.credentials === "object"
          ? Object.entries(formValues.credentials).reduce(
              (acc: Record<string, any>, [key, value]) => {
                if (value === undefined || value === null || value === "") {
                  return acc;
                }
                if (key === "scopes") {
                  if (Array.isArray(value)) {
                    const normalizedScopes = value.filter((scope) => scope != null && scope !== "");
                    if (normalizedScopes.length > 0) {
                      acc[key] = normalizedScopes;
                    }
                  }
                } else {
                  acc[key] = value;
                }
                return acc;
              },
              {},
            )
          : undefined;

      const mcpServerConfig: MCPServerConfig = {
        server_id: formValues.server_id || "",
        server_name: formValues.server_name || "",
        url: formValues.url,
        transport: formValues.transport,
        auth_type: formValues.auth_type,
        authorization_url: formValues.authorization_url,
        token_url: formValues.token_url,
        registration_url: formValues.registration_url,
        mcp_info: formValues.mcp_info,
        static_headers: staticHeaders,
      };

      if (credentials && Object.keys(credentials).length > 0) {
        mcpServerConfig.credentials = credentials;
      }

      const toolsResponse = await testMCPToolsListRequest(accessToken, mcpServerConfig, oauthAccessToken);

      if (toolsResponse.tools && !toolsResponse.error) {
        setTools(toolsResponse.tools);
        setToolsError(null);
        setToolsErrorStackTrace(null);
        if (toolsResponse.tools.length > 0 && !hasShownSuccessMessage) {
          setHasShownSuccessMessage(true);
        }
      } else {
        const errorMessage = toolsResponse.message || "Failed to retrieve tools list";
        setToolsError(errorMessage);
        setToolsErrorStackTrace(toolsResponse.stack_trace || null);
        setTools([]);
        setHasShownSuccessMessage(false);
      }
    } catch (error) {
      console.error("Tools fetch error:", error);
      setToolsError(error instanceof Error ? error.message : String(error));
      setToolsErrorStackTrace(null);
      setTools([]);
      setHasShownSuccessMessage(false);
    } finally {
      setIsLoadingTools(false);
    }
  };

  const clearTools = () => {
    setTools([]);
    setToolsError(null);
    setToolsErrorStackTrace(null);
    setHasShownSuccessMessage(false);
  };

  // Auto-fetch tools when form values change and required fields are available
  useEffect(() => {
    if (!enabled) {
      return;
    }

    if (canFetchTools) {
      fetchTools();
    } else {
      clearTools();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    formValues.url,
    formValues.transport,
    formValues.auth_type,
    accessToken,
    enabled,
    oauthAccessToken,
    canFetchTools,
    staticHeadersKey,
    credentialsKey,
  ]);

  return {
    tools,
    isLoadingTools,
    toolsError,
    toolsErrorStackTrace,
    hasShownSuccessMessage,
    canFetchTools,
    fetchTools,
    clearTools,
  };
};
