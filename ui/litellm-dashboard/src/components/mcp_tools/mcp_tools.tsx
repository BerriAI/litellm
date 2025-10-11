import React, { useState, useEffect } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { ToolTestPanel } from "./ToolTestPanel";
import { MCPTool, MCPToolsViewerProps, CallMCPToolResponse, mcpServerHasAuth } from "./types";
import { listMCPTools, callMCPTool } from "../networking";
import { getMCPAuthToken, setMCPAuthToken, removeMCPAuthToken } from "./mcp_auth_storage";

import { Modal, Input, Form } from "antd";
import { Button, Card, Title, Text } from "@tremor/react";
import { RobotOutlined, SafetyOutlined, ToolOutlined } from "@ant-design/icons";

import { AUTH_TYPE } from "./types";
import NotificationsManager from "../molecules/notifications_manager";

type AuthModalProps = {
  visible: boolean;
  onOk: (values: any) => void;
  onCancel: () => void;
  authType?: string | null;
};

export const AuthModal = ({ visible, onOk, onCancel, authType }: AuthModalProps) => {
  const [form] = Form.useForm();

  // Handler for modal OK
  const handleOk = () => {
    form.validateFields().then((values) => {
      if (authType === AUTH_TYPE.BASIC) {
        onOk(`${values.username.trim()}:${values.password.trim()}`);
      } else {
        onOk(values.authValue.trim());
      }
    });
  };

  let content;
  if (authType === AUTH_TYPE.API_KEY || authType === AUTH_TYPE.BEARER_TOKEN) {
    const label = authType === AUTH_TYPE.API_KEY ? "API Key" : "Bearer Token";
    content = (
      <Form.Item name="authValue" label={label} rules={[{ required: true, message: `Please input your ${label}` }]}>
        <Input.Password />
      </Form.Item>
    );
  } else if (authType === AUTH_TYPE.BASIC) {
    content = (
      <>
        <Form.Item name="username" label="Username" rules={[{ required: true, message: "Please input your username" }]}>
          <Input />
        </Form.Item>
        <Form.Item name="password" label="Password" rules={[{ required: true, message: "Please input your password" }]}>
          <Input.Password />
        </Form.Item>
      </>
    );
  }

  return (
    <Modal open={visible} title="Authentication" onOk={handleOk} onCancel={onCancel} destroyOnClose>
      <Form form={form} layout="vertical">
        {content}
      </Form>
    </Modal>
  );
};

const AuthSection = ({
  authType,
  onAuthSubmit,
  onClearAuth,
  hasAuth,
}: {
  authType: string | null | undefined;
  onAuthSubmit: (value: string) => void;
  onClearAuth: () => void;
  hasAuth: boolean;
}) => {
  const [modalVisible, setModalVisible] = useState(false);

  const handleAddAuth = () => setModalVisible(true);

  const handleModalOk = (authValue: string) => {
    onAuthSubmit(authValue);
    setModalVisible(false);
  };

  const handleModalCancel = () => setModalVisible(false);

  const handleClearAuth = () => {
    onClearAuth();
  };

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <Text className="text-sm font-medium text-gray-700">Authentication {hasAuth ? "✓" : ""}</Text>
        <div className="flex gap-2">
          {hasAuth && (
            <Button
              onClick={handleClearAuth}
              size="sm"
              variant="secondary"
              className="text-xs text-red-600 hover:text-red-700"
            >
              Clear
            </Button>
          )}
          <Button onClick={handleAddAuth} size="sm" variant="secondary" className="text-xs">
            {hasAuth ? "Update" : "Add Auth"}
          </Button>
        </div>
      </div>
      <Text className="text-xs text-gray-500">
        {hasAuth ? "Authentication configured and saved locally" : "Some tools may require authentication"}
      </Text>
      <AuthModal visible={modalVisible} onOk={handleModalOk} onCancel={handleModalCancel} authType={authType} />
    </div>
  );
};

const MCPToolsViewer = ({
  serverId,
  accessToken,
  auth_type,
  userRole,
  userID,
  serverAlias, // Add serverAlias prop
}: MCPToolsViewerProps) => {
  const [mcpAuthValue, setMcpAuthValue] = useState("");
  const [selectedTool, setSelectedTool] = useState<MCPTool | null>(null);
  const [toolResult, setToolResult] = useState<CallMCPToolResponse | null>(null);
  const [toolError, setToolError] = useState<Error | null>(null);

  // Load stored auth token on component mount
  useEffect(() => {
    if (mcpServerHasAuth(auth_type)) {
      const storedAuthValue = getMCPAuthToken(serverId, serverAlias || undefined);
      if (storedAuthValue) {
        setMcpAuthValue(storedAuthValue);
      }
    }
  }, [serverId, serverAlias, auth_type]);

  // Function to handle auth submission with localStorage persistence
  const handleAuthSubmit = (authValue: string) => {
    setMcpAuthValue(authValue);
    if (authValue && mcpServerHasAuth(auth_type)) {
      setMCPAuthToken(serverId, authValue, auth_type || "none", serverAlias || undefined);
      NotificationsManager.success("Authentication token saved locally");
    }
  };

  // Function to clear auth token
  const handleClearAuth = () => {
    setMcpAuthValue("");
    removeMCPAuthToken(serverId);
    NotificationsManager.info("Authentication token cleared");
  };

  // Query to fetch MCP tools
  const {
    data: mcpToolsResponse,
    isLoading: isLoadingTools,
    error: mcpToolsError,
  } = useQuery({
    queryKey: ["mcpTools", serverId, mcpAuthValue, serverAlias],
    queryFn: () => {
      if (!accessToken) throw new Error("Access Token required");
      return listMCPTools(accessToken, serverId, mcpAuthValue, serverAlias || undefined);
    },
    enabled: !!accessToken,
    staleTime: 30000, // Consider data fresh for 30 seconds
  });

  // Mutation for calling a tool
  const { mutate: executeTool, isPending: isCallingTool } = useMutation({
    mutationFn: async (args: { tool: MCPTool; arguments: Record<string, any>; authValue: string }) => {
      if (!accessToken) throw new Error("Access Token required");

      try {
        const result = await callMCPTool(
          accessToken,
          args.tool.name,
          args.arguments,
          args.authValue,
          serverAlias || undefined,
        );
        return result;
      } catch (error) {
        throw error;
      }
    },
    onSuccess: (data) => {
      setToolResult(data);
      setToolError(null);
    },
    onError: (error: Error) => {
      setToolError(error);
      setToolResult(null);
    },
  });

  const toolsData = mcpToolsResponse?.tools || [];
  const hasAuth = mcpAuthValue !== "";

  return (
    <div className="w-full h-screen p-4 bg-white">
      <Card className="w-full rounded-xl shadow-md overflow-hidden">
        <div className="flex h-auto w-full gap-4">
          {/* Left Sidebar with Controls */}
          <div className="w-1/4 p-4 bg-gray-50 flex flex-col">
            <Title className="text-xl font-semibold mb-6 mt-2">MCP Tools</Title>

            <div className="flex flex-col flex-1">
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

                {/* Loading State */}
                {isLoadingTools && (
                  <div className="flex flex-col items-center justify-center py-8 bg-white border border-gray-200 rounded-lg">
                    <div className="relative mb-3">
                      <div className="animate-spin rounded-full h-6 w-6 border-2 border-gray-200"></div>
                      <div className="animate-spin rounded-full h-6 w-6 border-2 border-blue-600 border-t-transparent absolute top-0"></div>
                    </div>
                    <p className="text-xs font-medium text-gray-700">Loading tools...</p>
                  </div>
                )}

                {/* Error State */}
                {mcpToolsResponse?.error && !isLoadingTools && !toolsData.length && (
                  <div className="p-3 text-xs text-red-800 rounded-lg bg-red-50 border border-red-200">
                    <p className="font-medium">Error: {mcpToolsResponse.message}</p>
                  </div>
                )}

                {/* No Tools State */}
                {!isLoadingTools && !mcpToolsResponse?.error && (!toolsData || toolsData.length === 0) && (
                  <div className="p-4 text-center bg-white border border-gray-200 rounded-lg">
                    <div className="mx-auto w-8 h-8 bg-gray-200 rounded-full flex items-center justify-center mb-2">
                      <svg className="w-4 h-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
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
                {!isLoadingTools && !mcpToolsResponse?.error && toolsData.length > 0 && (
                  <div
                    className="space-y-2 flex-1 overflow-y-auto min-h-0 mcp-tools-scrollable"
                    style={{
                      maxHeight: "400px",
                      scrollbarWidth: "auto",
                      scrollbarColor: "#cbd5e0 #f7fafc",
                    }}
                  >
                    {toolsData.map((tool: MCPTool) => (
                      <div
                        key={tool.name}
                        className={`border rounded-lg p-3 cursor-pointer transition-all hover:shadow-sm ${
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
                              src={tool.mcp_info.logo_url}
                              alt={`${tool.mcp_info.server_name} logo`}
                              className="w-4 h-4 object-contain flex-shrink-0 mt-0.5"
                            />
                          )}
                          <div className="flex-1 min-w-0">
                            <h4 className="font-mono text-xs font-medium text-gray-900 truncate">{tool.name}</h4>
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
              </div>

              {/* Authentication Section - Below tools list */}
              {mcpServerHasAuth(auth_type) && (
                <div className="pt-4 border-t border-gray-200 flex-shrink-0 mt-6">
                  {!hasAuth ? (
                    /* Prominent display when auth required but not provided */
                    <div className="p-4 bg-gradient-to-r from-orange-50 to-red-50 border border-orange-200 rounded-lg">
                      <div className="flex items-center mb-3">
                        <SafetyOutlined className="mr-2 text-orange-600 text-lg" />
                        <Text className="font-semibold text-orange-800">Authentication Required</Text>
                      </div>
                      <Text className="text-sm text-orange-700 mb-4">
                        This MCP server requires authentication. You must add your credentials below to access the
                        tools.
                      </Text>
                      <AuthSection
                        authType={auth_type}
                        onAuthSubmit={handleAuthSubmit}
                        onClearAuth={handleClearAuth}
                        hasAuth={hasAuth}
                      />
                    </div>
                  ) : (
                    /* Subtle display when already authenticated */
                    <>
                      <Text className="font-medium block mb-3 text-gray-700 flex items-center">
                        <SafetyOutlined className="mr-2" /> Authentication
                      </Text>
                      <AuthSection
                        authType={auth_type}
                        onAuthSubmit={handleAuthSubmit}
                        onClearAuth={handleClearAuth}
                        hasAuth={hasAuth}
                      />
                    </>
                  )}
                </div>
              )}
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
                    needsAuth={mcpServerHasAuth(auth_type)}
                    authValue={mcpAuthValue}
                    onSubmit={(args) => {
                      executeTool({ tool: selectedTool, arguments: args, authValue: mcpAuthValue });
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
