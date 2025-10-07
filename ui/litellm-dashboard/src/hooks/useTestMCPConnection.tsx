import { useState, useEffect } from "react";
import { testMCPToolsListRequest } from "../components/networking";

interface MCPServerConfig {
  server_id?: string;
  server_name?: string;
  url?: string;
  transport?: string;
  auth_type?: string;
  mcp_info?: any;
}

interface UseTestMCPConnectionProps {
  accessToken: string | null;
  formValues: Record<string, any>;
  enabled?: boolean; // Optional flag to enable/disable auto-fetching
}

interface UseTestMCPConnectionReturn {
  tools: any[];
  isLoadingTools: boolean;
  toolsError: string | null;
  hasShownSuccessMessage: boolean;
  canFetchTools: boolean;
  fetchTools: () => Promise<void>;
  clearTools: () => void;
}

export const useTestMCPConnection = ({
  accessToken,
  formValues,
  enabled = true,
}: UseTestMCPConnectionProps): UseTestMCPConnectionReturn => {
  const [tools, setTools] = useState<any[]>([]);
  const [isLoadingTools, setIsLoadingTools] = useState(false);
  const [toolsError, setToolsError] = useState<string | null>(null);
  const [hasShownSuccessMessage, setHasShownSuccessMessage] = useState(false);

  // Check if we have the minimum required fields to fetch tools
  const canFetchTools = !!(formValues.url && formValues.transport && formValues.auth_type && accessToken);

  const fetchTools = async () => {
    if (!accessToken || !formValues.url) {
      return;
    }

    setIsLoadingTools(true);
    setToolsError(null);

    try {
      // Prepare the MCP server config from form values
      const mcpServerConfig: MCPServerConfig = {
        server_id: formValues.server_id || "",
        server_name: formValues.server_name || "",
        url: formValues.url,
        transport: formValues.transport,
        auth_type: formValues.auth_type,
        mcp_info: formValues.mcp_info,
      };

      const toolsResponse = await testMCPToolsListRequest(accessToken, mcpServerConfig);

      if (toolsResponse.tools && !toolsResponse.error) {
        setTools(toolsResponse.tools);
        setToolsError(null);
        if (toolsResponse.tools.length > 0 && !hasShownSuccessMessage) {
          setHasShownSuccessMessage(true);
        }
      } else {
        const errorMessage = toolsResponse.message || "Failed to retrieve tools list";
        setToolsError(errorMessage);
        setTools([]);
        setHasShownSuccessMessage(false);
      }
    } catch (error) {
      console.error("Tools fetch error:", error);
      setToolsError(error instanceof Error ? error.message : String(error));
      setTools([]);
      setHasShownSuccessMessage(false);
    } finally {
      setIsLoadingTools(false);
    }
  };

  const clearTools = () => {
    setTools([]);
    setToolsError(null);
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
  }, [formValues.url, formValues.transport, formValues.auth_type, accessToken, enabled, canFetchTools]);

  return {
    tools,
    isLoadingTools,
    toolsError,
    hasShownSuccessMessage,
    canFetchTools,
    fetchTools,
    clearTools,
  };
};
