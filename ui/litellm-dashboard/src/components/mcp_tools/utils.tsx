export const extractMCPToken = (url: string): { token: string | null, baseUrl: string } => {
  try {
    const mcpIndex = url.indexOf('/mcp/');
    if (mcpIndex === -1) return { token: null, baseUrl: url };

    const parts = url.split('/mcp/');
    if (parts.length !== 2) return { token: null, baseUrl: url };

    const baseUrl = parts[0] + '/mcp/';
    const afterMcp = parts[1];

    // If there's no content after /mcp/, return null token
    if (!afterMcp) return { token: null, baseUrl: url };

    return {
      token: afterMcp,
      baseUrl: baseUrl
    };
  } catch (error) {
    console.error('Error parsing MCP URL:', error);
    return { token: null, baseUrl: url };
  }
};

export const maskUrl = (url: string): string => {
  const { token, baseUrl } = extractMCPToken(url);
  if (!token) return url;
  return baseUrl + '...';
};

export const getMaskedAndFullUrl = (url: string): { maskedUrl: string, hasToken: boolean } => {
  const { token } = extractMCPToken(url);
  return {
    maskedUrl: maskUrl(url),
    hasToken: !!token
  };
}; 