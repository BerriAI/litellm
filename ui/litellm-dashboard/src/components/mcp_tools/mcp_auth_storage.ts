// Utility functions for managing MCP server authentication tokens in localStorage

const MCP_AUTH_STORAGE_KEY = "litellm_mcp_auth_tokens";

export interface MCPAuthToken {
  serverId: string;
  serverAlias?: string;
  authValue: string;
  authType: string;
  timestamp: number;
}

export interface MCPAuthStorage {
  [serverId: string]: MCPAuthToken;
}

/**
 * Get all stored MCP authentication tokens
 */
export const getMCPAuthTokens = (): MCPAuthStorage => {
  try {
    const stored = localStorage.getItem(MCP_AUTH_STORAGE_KEY);
    return stored ? JSON.parse(stored) : {};
  } catch (error) {
    console.error("Error reading MCP auth tokens from localStorage:", error);
    return {};
  }
};

/**
 * Get authentication token for a specific MCP server
 */
export const getMCPAuthToken = (serverId: string, serverAlias?: string): string | null => {
  try {
    const tokens = getMCPAuthTokens();
    const token = tokens[serverId];

    // If token exists, check if serverAlias matches (both can be undefined)
    if (token && token.serverAlias === serverAlias) {
      return token.authValue;
    }

    // If no serverAlias was provided and token exists without serverAlias, return it
    if (token && !serverAlias && !token.serverAlias) {
      return token.authValue;
    }

    return null;
  } catch (error) {
    console.error("Error getting MCP auth token:", error);
    return null;
  }
};

/**
 * Store authentication token for an MCP server
 */
export const setMCPAuthToken = (serverId: string, authValue: string, authType: string, serverAlias?: string): void => {
  try {
    const tokens = getMCPAuthTokens();

    tokens[serverId] = {
      serverId,
      serverAlias,
      authValue,
      authType,
      timestamp: Date.now(),
    };

    localStorage.setItem(MCP_AUTH_STORAGE_KEY, JSON.stringify(tokens));
  } catch (error) {
    console.error("Error storing MCP auth token:", error);
  }
};

/**
 * Remove authentication token for an MCP server
 */
export const removeMCPAuthToken = (serverId: string): void => {
  try {
    const tokens = getMCPAuthTokens();
    delete tokens[serverId];
    localStorage.setItem(MCP_AUTH_STORAGE_KEY, JSON.stringify(tokens));
  } catch (error) {
    console.error("Error removing MCP auth token:", error);
  }
};

/**
 * Clear all MCP authentication tokens (useful for logout)
 */
export const clearMCPAuthTokens = (): void => {
  try {
    localStorage.removeItem(MCP_AUTH_STORAGE_KEY);
  } catch (error) {
    console.error("Error clearing MCP auth tokens:", error);
  }
};

/**
 * Check if a token exists for a server
 */
export const hasMCPAuthToken = (serverId: string, serverAlias?: string): boolean => {
  const token = getMCPAuthToken(serverId, serverAlias);
  return token !== null;
};
