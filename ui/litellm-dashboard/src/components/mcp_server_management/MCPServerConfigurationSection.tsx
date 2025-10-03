import React, { useEffect, useState } from "react"
import { Text } from "@tremor/react"
import { Select, Checkbox } from "antd"
import { fetchMCPServers } from "../networking"
import { MCPServer } from "../mcp_tools/types"
import { InfoCircleOutlined } from "@ant-design/icons"

interface MCPServerConfigurationSectionProps {
  accessToken: string
  onChange?: (config: ServerToolsConfig) => void
  value?: ServerToolsConfig
}

type ServerToolsConfig = Record<string, { allowed_tools: string[] }>

const MCPServerConfigurationSection: React.FC<MCPServerConfigurationSectionProps> = ({
  accessToken,
  onChange,
  value = {},
}) => {
  const [mcpServers, setMCPServers] = useState<MCPServer[]>([])
  const [loading, setLoading] = useState(false)
  const [selectedServerIds, setSelectedServerIds] = useState<string[]>([])
  const [expanded, setExpanded] = useState(true)

  // Initialize selected servers from value prop on mount
  useEffect(() => {
    if (value && Object.keys(value).length > 0) {
      setSelectedServerIds(Object.keys(value))
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

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

  // Handle server selection change
  const handleServerSelectionChange = (serverIds: string[]) => {
    setSelectedServerIds(serverIds)

    // Create new config with only selected servers
    const newConfig: ServerToolsConfig = {}
    serverIds.forEach((serverId) => {
      // If server already exists in config, preserve its selections
      if (value?.[serverId]) {
        newConfig[serverId] = value[serverId]
      } else {
        // New server - auto-select all its allowed_tools
        const server = mcpServers.find((s) => s.server_id === serverId)
        newConfig[serverId] = {
          allowed_tools: server?.allowed_tools || [],
        }
      }
    })

    onChange?.(newConfig)
  }

  // Handle tool selection for a specific server
  const handleToolSelection = (serverId: string, toolName: string, checked: boolean) => {
    const currentTools = value?.[serverId]?.allowed_tools || []
    const newTools = checked ? [...currentTools, toolName] : currentTools.filter((t) => t !== toolName)

    const newConfig = {
      ...value,
      [serverId]: { allowed_tools: newTools },
    }

    onChange?.(newConfig)
  }

  // Handle select all tools for a server
  const handleSelectAllTools = (serverId: string) => {
    const server = mcpServers.find((s) => s.server_id === serverId)
    if (!server?.allowed_tools) return

    const newConfig = {
      ...value,
      [serverId]: { allowed_tools: [...server.allowed_tools] },
    }

    onChange?.(newConfig)
  }

  // Handle deselect all tools for a server
  const handleDeselectAllTools = (serverId: string) => {
    const newConfig = {
      ...value,
      [serverId]: { allowed_tools: [] },
    }

    onChange?.(newConfig)
  }

  // Handle removing a server
  const handleRemoveServer = (serverId: string) => {
    const newServerIds = selectedServerIds.filter((id) => id !== serverId)
    handleServerSelectionChange(newServerIds)
  }

  // Get server by ID
  const getServerById = (serverId: string) => mcpServers.find((s) => s.server_id === serverId)

  return (
    <div className="mt-4">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <Text className="text-lg font-semibold">Manage MCPs</Text>
          <InfoCircleOutlined className="text-gray-400" />
        </div>
        <button onClick={() => setExpanded(!expanded)} className="text-gray-500 hover:text-gray-700">
          {expanded ? "▼" : "▶"}
        </button>
      </div>

      {expanded && (
        <>
          {/* Server Selection Dropdown */}
          <div className="mb-4">
            <Text className="text-sm font-medium mb-2">Selected MCP Servers</Text>
            <Select
              mode="multiple"
              placeholder="Select MCP servers"
              value={selectedServerIds}
              onChange={handleServerSelectionChange}
              style={{ width: "100%" }}
              loading={loading}
              showSearch
              filterOption={(input, option) => {
                const server = mcpServers.find((s) => s.server_id === option?.value)
                return (server?.server_name || server?.server_id || "").toLowerCase().includes(input.toLowerCase())
              }}
            >
              {mcpServers.map((server) => (
                <Select.Option key={server.server_id} value={server.server_id}>
                  {server.server_name || server.server_id}
                  {server.description && <span className="text-gray-500 ml-2">- {server.description}</span>}
                </Select.Option>
              ))}
            </Select>
            <Text className="text-xs text-gray-500 mt-1">{selectedServerIds.length} MCP server(s) selected</Text>
          </div>

          {/* Selected Servers with Tool Configuration */}
          {selectedServerIds.length > 0 && (
            <div className="space-y-4">
              {selectedServerIds.map((serverId) => {
                const server = getServerById(serverId)
                if (!server) return null

                const serverConfig = value?.[serverId]
                const selectedTools = serverConfig?.allowed_tools || []
                const availableTools = server.allowed_tools || []

                return (
                  <div key={serverId} className="border rounded-lg p-4 bg-white">
                    {/* Server Header */}
                    <div className="flex items-start justify-between mb-3">
                      <div>
                        <Text className="font-semibold">{server.server_name || server.server_id}</Text>
                        {server.description && <Text className="text-sm text-gray-500">{server.description}</Text>}
                      </div>
                      <div className="flex items-center gap-2">
                        <button
                          onClick={() => handleSelectAllTools(serverId)}
                          className="text-sm text-blue-500 hover:text-blue-700"
                        >
                          Select All
                        </button>
                        <button
                          onClick={() => handleDeselectAllTools(serverId)}
                          className="text-sm text-blue-500 hover:text-blue-700"
                        >
                          Deselect All
                        </button>
                        <button
                          onClick={() => handleRemoveServer(serverId)}
                          className="text-gray-400 hover:text-gray-600"
                        >
                          ✕
                        </button>
                      </div>
                    </div>

                    {/* Tools List */}
                    <div>
                      <Text className="text-sm font-medium mb-2">Available Tools</Text>
                      {availableTools.length > 0 ? (
                        <div className="space-y-2">
                          {availableTools.map((tool) => (
                            <div key={tool} className="flex items-start gap-2">
                              <Checkbox
                                checked={selectedTools.includes(tool)}
                                onChange={(e) => handleToolSelection(serverId, tool, e.target.checked)}
                              />
                              <div>
                                <Text className="text-sm font-medium">{tool}</Text>
                              </div>
                            </div>
                          ))}
                        </div>
                      ) : (
                        <Text className="text-sm text-gray-500">No tools available for this server</Text>
                      )}
                    </div>
                  </div>
                )
              })}
            </div>
          )}

          {/* Help Text */}
          <Text className="text-sm text-gray-500 mt-4">
            Select MCP servers and their specific tools to configure fine-grained access control.
          </Text>
        </>
      )}
    </div>
  )
}

export default MCPServerConfigurationSection
