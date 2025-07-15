import React, { useState } from "react"
import { Modal, Tooltip, Form, Select, message, Button as AntdButton, Input } from "antd"
import { InfoCircleOutlined } from "@ant-design/icons"
import { Button, TextInput } from "@tremor/react"
import { createMCPServer } from "../networking"
import { MCPServer, MCPServerCostInfo } from "./types"
import MCPServerCostConfig from "./mcp_server_cost_config"
import MCPConnectionStatus from "./mcp_connection_status"
import StdioConfiguration from "./StdioConfiguration"
import { isAdminRole } from "@/utils/roles"

const asset_logos_folder = "../ui/assets/logos/"
export const mcpLogoImg = `${asset_logos_folder}mcp_logo.png`

interface CreateMCPServerProps {
  userRole: string
  accessToken: string | null
  onCreateSuccess: (newMcpServer: MCPServer) => void
  isModalVisible: boolean
  setModalVisible: (visible: boolean) => void
  availableAccessGroups: string[]
}

const CreateMCPServer: React.FC<CreateMCPServerProps> = ({
  userRole,
  accessToken,
  onCreateSuccess,
  isModalVisible,
  setModalVisible,
  availableAccessGroups,
}) => {
  const [form] = Form.useForm()
  const [isLoading, setIsLoading] = useState(false)
  const [costConfig, setCostConfig] = useState<MCPServerCostInfo>({})
  const [formValues, setFormValues] = useState<Record<string, any>>({})
  const [tools, setTools] = useState<any[]>([])
  const [transportType, setTransportType] = useState<string>("sse")
  const [searchValue, setSearchValue] = useState<string>("")

  const handleCreate = async (formValues: Record<string, any>) => {
    setIsLoading(true)
    try {
      // Transform access groups into objects with name property

      const accessGroups = formValues.mcp_access_groups

      // Process stdio configuration if present
      let stdioFields = {}
      if (formValues.stdio_config && transportType === "stdio") {
        try {
          const stdioConfig = JSON.parse(formValues.stdio_config)

          // Handle both formats:
          // 1. Full mcpServers structure: {"mcpServers": {"server-name": {...}}}
          // 2. Direct config: {"command": "...", "args": [...], "env": {...}}

          let actualConfig = stdioConfig

          // If it's the full mcpServers structure, extract the first server config
          if (stdioConfig.mcpServers && typeof stdioConfig.mcpServers === "object") {
            const serverNames = Object.keys(stdioConfig.mcpServers)
            if (serverNames.length > 0) {
              const firstServerName = serverNames[0]
              actualConfig = stdioConfig.mcpServers[firstServerName]

              // If no alias is provided, use the server name from the JSON
              if (!formValues.alias) {
                formValues.alias = firstServerName.replace(/-/g, "_") // Replace hyphens with underscores
              }
            }
          }

          stdioFields = {
            command: actualConfig.command,
            args: actualConfig.args,
            env: actualConfig.env,
          }

          console.log("Parsed stdio config:", stdioFields)
        } catch (error) {
          message.error("Invalid JSON in stdio configuration")
          return
        }
      }

      // Prepare the payload with cost configuration
      const payload = {
        ...formValues,
        ...stdioFields,
        // Remove the raw stdio_config field as we've extracted its components
        stdio_config: undefined,
        mcp_info: {
          server_name: formValues.alias || formValues.url,
          description: formValues.description,
          mcp_server_cost_info: Object.keys(costConfig).length > 0 ? costConfig : null,
        },
        mcp_access_groups: accessGroups,
      }

      console.log(`Payload: ${JSON.stringify(payload)}`)

      if (accessToken != null) {
        const response = await createMCPServer(accessToken, payload)

        message.success("MCP Server created successfully")
        form.resetFields()
        setCostConfig({})
        setTools([])
        setModalVisible(false)
        onCreateSuccess(response)
      }
    } catch (error) {
      message.error("Error creating MCP Server: " + error, 20)
    } finally {
      setIsLoading(false)
    }
  }

  // state
  const handleCancel = () => {
    form.resetFields()
    setCostConfig({})
    setTools([])
    setModalVisible(false)
  }

  const handleTransportChange = (value: string) => {
    setTransportType(value)
    // Clear fields that are not relevant for the selected transport
    if (value === "stdio") {
      form.setFieldsValue({ url: undefined, auth_type: undefined })
    } else {
      form.setFieldsValue({ command: undefined, args: undefined, env: undefined })
    }
  }

  // Generate options with existing groups and potential new group
  const getAccessGroupOptions = () => {
    const existingOptions = availableAccessGroups.map((group: string) => ({
      value: group,
      label: (
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 bg-green-500 rounded-full"></div>
          <span className="font-medium">{group}</span>
        </div>
      ),
    }))

    // If search value doesn't match any existing group and is not empty, add "create new group" option
    if (searchValue && !availableAccessGroups.some(group => group.toLowerCase().includes(searchValue.toLowerCase()))) {
      existingOptions.push({
        value: searchValue,
        label: (
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
            <span className="font-medium">{searchValue}</span>
            <span className="text-gray-400 text-xs ml-1">create new group</span>
          </div>
        ),
      })
    }

    return existingOptions
  }

  // rendering
  if (!isAdminRole(userRole)) {
    return null
  }

  return (
    <Modal
      title={
        <div className="flex items-center space-x-3 pb-4 border-b border-gray-100">
          <img
            src={mcpLogoImg}
            alt="MCP Logo"
            className="w-8 h-8 object-contain"
            style={{
              height: "20px",
              width: "20px",
              marginRight: "8px",
              objectFit: "contain",
            }}
          />
          <h2 className="text-xl font-semibold text-gray-900">Add New MCP Server</h2>
        </div>
      }
      open={isModalVisible}
      width={1000}
      onCancel={handleCancel}
      footer={null}
      className="top-8"
      styles={{
        body: { padding: "24px" },
        header: { padding: "24px 24px 0 24px", border: "none" },
      }}
    >
      <div className="mt-6">
        <Form
          form={form}
          onFinish={handleCreate}
          onValuesChange={(_, allValues) => setFormValues(allValues)}
          layout="vertical"
          className="space-y-6"
        >
          <div className="grid grid-cols-1 gap-6">
            <Form.Item
              label={
                <span className="text-sm font-medium text-gray-700 flex items-center">
                  MCP Server Name
                  <Tooltip title="Best practice: Use a descriptive name that indicates the server's purpose (e.g., 'GitHub_MCP', 'Email_Service'). Hyphens '-' are not allowed; use underscores '_' instead.">
                    <InfoCircleOutlined className="ml-2 text-blue-400 hover:text-blue-600 cursor-help" />
                  </Tooltip>
                </span>
              }
              name="alias"
              rules={[
                { required: false, message: "Please enter a server name" },
                {
                  validator: (_, value) =>
                    value && value.includes("-")
                      ? Promise.reject("Server name cannot contain '-' (hyphen). Please use '_' (underscore) instead.")
                      : Promise.resolve(),
                },
              ]}
            >
              <TextInput
                placeholder="e.g., GitHub_MCP, Zapier_MCP, etc."
                className="rounded-lg border-gray-300 focus:border-blue-500 focus:ring-blue-500"
              />
            </Form.Item>

            <Form.Item
              label={<span className="text-sm font-medium text-gray-700">Description</span>}
              name="description"
              rules={[
                {
                  required: false,
                  message: "Please enter a server description",
                },
              ]}
            >
              <TextInput
                placeholder="Brief description of what this server does"
                className="rounded-lg border-gray-300 focus:border-blue-500 focus:ring-blue-500"
              />
            </Form.Item>

            <Form.Item
              label={<span className="text-sm font-medium text-gray-700">Transport Type</span>}
              name="transport"
              rules={[{ required: true, message: "Please select a transport type" }]}
            >
              <Select
                placeholder="Select transport"
                className="rounded-lg"
                size="large"
                onChange={handleTransportChange}
                value={transportType}
              >
                <Select.Option value="http">HTTP</Select.Option>
                <Select.Option value="sse">Server-Sent Events (SSE)</Select.Option>
                <Select.Option value="stdio">Standard Input/Output (stdio)</Select.Option>
              </Select>
            </Form.Item>

            {/* URL field - only show for HTTP and SSE */}
            {transportType !== "stdio" && (
              <Form.Item
                label={<span className="text-sm font-medium text-gray-700">MCP Server URL</span>}
                name="url"
                rules={[
                  { required: true, message: "Please enter a server URL" },
                  { type: "url", message: "Please enter a valid URL" },
                ]}
              >
                <TextInput
                  placeholder="https://your-mcp-server.com"
                  className="rounded-lg border-gray-300 focus:border-blue-500 focus:ring-blue-500"
                />
              </Form.Item>
            )}

            {/* Authentication - only show for HTTP and SSE */}
            {transportType !== "stdio" && (
              <Form.Item
                label={<span className="text-sm font-medium text-gray-700">Authentication</span>}
                name="auth_type"
                rules={[{ required: true, message: "Please select an auth type" }]}
              >
                <Select placeholder="Select auth type" className="rounded-lg" size="large">
                  <Select.Option value="none">None</Select.Option>
                  <Select.Option value="api_key">API Key</Select.Option>
                  <Select.Option value="bearer_token">Bearer Token</Select.Option>
                  <Select.Option value="basic">Basic Auth</Select.Option>
                </Select>
              </Form.Item>
            )}

            {/* Stdio Configuration - only show for stdio transport */}
            <StdioConfiguration isVisible={transportType === "stdio"} />

            <Form.Item
              label={
                <span className="text-sm font-medium text-gray-700 flex items-center">
                  MCP Version
                  <Tooltip title="Select the MCP specification version your server supports">
                    <InfoCircleOutlined className="ml-2 text-gray-400 hover:text-gray-600" />
                  </Tooltip>
                </span>
              }
              name="spec_version"
              rules={[{ required: true, message: "Please select a spec version" }]}
            >
              <Select placeholder="Select MCP version" className="rounded-lg" size="large">
                <Select.Option value="2025-03-26">2025-03-26 (Latest)</Select.Option>
                <Select.Option value="2024-11-05">2024-11-05</Select.Option>
              </Select>
            </Form.Item>

            <Form.Item
              label={
                <span className="text-sm font-medium text-gray-700 flex items-center">
                  MCP Access Groups
                  <Tooltip title="Specify access groups for this MCP server. Users must be in at least one of these groups to access the server.">
                    <InfoCircleOutlined className="ml-2 text-blue-400 hover:text-blue-600 cursor-help" />
                  </Tooltip>
                </span>
              }
              name="mcp_access_groups"
              className="mb-4"
            >
              <Select
                mode="tags"
                showSearch
                placeholder="Select existing groups or type to create new ones"
                optionFilterProp="value"
                filterOption={(input, option) =>
                  (option?.value ?? '').toLowerCase().includes(input.toLowerCase())
                }
                onSearch={(value) => setSearchValue(value)}
                tokenSeparators={[","]}
                options={getAccessGroupOptions()}
                maxTagCount="responsive"
                allowClear
              />
            </Form.Item>
          </div>

          {/* Connection Status Section */}
          <div className="mt-8 pt-6 border-t border-gray-200">
            <MCPConnectionStatus accessToken={accessToken} formValues={formValues} onToolsLoaded={setTools} />
          </div>

          {/* Cost Configuration Section */}
          <div className="mt-6">
            <MCPServerCostConfig value={costConfig} onChange={setCostConfig} tools={tools} disabled={false} />
          </div>

          <div className="flex items-center justify-end space-x-3 pt-6 border-t border-gray-100">
            <Button variant="secondary" onClick={handleCancel}>
              Cancel
            </Button>
            <Button variant="primary" loading={isLoading}>
              {isLoading ? "Creating..." : "Add MCP Server"}
            </Button>
          </div>
        </Form>
      </div>
    </Modal>
  )
}

export default CreateMCPServer
