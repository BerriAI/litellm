import React, { useEffect, useState } from "react"
import { Text } from "@tremor/react"
import { fetchMCPServers } from "../networking"
import { MCPServer } from "../mcp_tools/types"

interface MCPServerConfigurationSectionProps {
  accessToken: string
  onChange?: (config: any) => void
  value?: any
}

const MCPServerConfigurationSection: React.FC<MCPServerConfigurationSectionProps> = ({
  accessToken,
  onChange,
  value,
}) => {
  const [mcpServers, setMCPServers] = useState<MCPServer[]>([])
  const [loading, setLoading] = useState(false)

  // Fetch MCP servers with their allowed_tools on mount
  useEffect(() => {
    const fetchServers = async () => {
      if (!accessToken) return
      setLoading(true)
      try {
        const serversRes = await fetchMCPServers(accessToken)
        const servers = Array.isArray(serversRes) ? serversRes : serversRes.data || []
        setMCPServers(servers)
      } catch (error) {
        console.error("Error fetching MCP servers:", error)
      } finally {
        setLoading(false)
      }
    }
    fetchServers()
  }, [accessToken])

  return (
    <div className="mt-4">
      <Text className="text-sm text-gray-600 mb-4">
        Configure fine-grained tool access for MCP servers. Select specific tools from each server that this team can
        use.
      </Text>
      {/* Placeholder for MCP tool configuration UI */}
      <div className="p-4 border border-dashed border-gray-300 rounded-md bg-gray-50">
        <Text className="text-sm text-gray-500 text-center">
          {loading
            ? "Loading MCP servers..."
            : `${mcpServers.length} MCP server(s) available. UI configuration coming soon...`}
        </Text>
        {/* Debug info - will be replaced with actual UI */}
        {mcpServers.length > 0 && (
          <div className="mt-2 text-xs text-gray-400">
            <details>
              <summary className="cursor-pointer">Debug: View loaded servers and their allowed_tools</summary>
              <ul className="mt-2 space-y-1">
                {mcpServers.map((server) => (
                  <li key={server.server_id}>
                    {server.server_name || server.server_id} - {server.allowed_tools?.length || 0} allowed tools
                    {server.allowed_tools && server.allowed_tools.length > 0 && (
                      <span className="ml-2 text-gray-500">
                        [{server.allowed_tools.slice(0, 3).join(", ")}
                        {server.allowed_tools.length > 3 ? ", ..." : ""}]
                      </span>
                    )}
                  </li>
                ))}
              </ul>
            </details>
          </div>
        )}
      </div>
    </div>
  )
}

export default MCPServerConfigurationSection
