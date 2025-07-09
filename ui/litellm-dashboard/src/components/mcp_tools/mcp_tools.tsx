import React, { useState } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { DataTable } from "../view_logs/table";
import { columns, ToolTestPanel } from "./columns";
import {
  MCPTool,
  MCPToolsViewerProps,
  CallMCPToolResponse,
  mcpServerHasAuth,
} from "./types";
import { listMCPTools, callMCPTool } from "../networking";

import { Modal, Input, Form } from "antd";
import { Button } from "@tremor/react";

import { AUTH_TYPE } from "./types";

type AuthModalProps = {
  visible: boolean;
  onOk: (values: any) => void;
  onCancel: () => void;
  authType?: string | null;
};

export const AuthModal = ({
  visible,
  onOk,
  onCancel,
  authType,
}: AuthModalProps) => {
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
      <Form.Item
        name="authValue"
        label={label}
        rules={[{ required: true, message: `Please input your ${label}` }]}
      >
        <Input.Password />
      </Form.Item>
    );
  } else if (authType === AUTH_TYPE.BASIC) {
    content = (
      <>
        <Form.Item
          name="username"
          label="Username"
          rules={[{ required: true, message: "Please input your username" }]}
        >
          <Input />
        </Form.Item>
        <Form.Item
          name="password"
          label="Password"
          rules={[{ required: true, message: "Please input your password" }]}
        >
          <Input.Password />
        </Form.Item>
      </>
    );
  }

  return (
    <Modal
      open={visible}
      title="Authentication"
      onOk={handleOk}
      onCancel={onCancel}
      destroyOnClose
    >
      <Form form={form} layout="vertical">
        {content}
      </Form>
    </Modal>
  );
};

const AuthSection = ({ 
  authType, 
  onAuthSubmit 
}: {
  authType: string | null | undefined;
  onAuthSubmit: (value: string) => void;
}) => {
  const [modalVisible, setModalVisible] = useState(false);

  const handleAddAuth = () => setModalVisible(true);

  const handleModalOk = (authValue: string) => {
    onAuthSubmit(authValue);
    setModalVisible(false);
  };

  const handleModalCancel = () => setModalVisible(false);

  return (
    <>
      <Button onClick={handleAddAuth}>
        Add Auth
      </Button>
      <AuthModal
        visible={modalVisible}
        onOk={handleModalOk}
        onCancel={handleModalCancel}
        authType={authType}
      />
    </>
  );
};

// Wrapper to handle the type mismatch between MCPTool and DataTable's expected type
function DataTableWrapper({
  columns,
  data,
  isLoading,
}: {
  columns: any;
  data: MCPTool[];
  isLoading: boolean;
}) {
  // Create a dummy renderSubComponent and getRowCanExpand function
  const renderSubComponent = () => <div />;
  const getRowCanExpand = () => false;

  return (
    <DataTable
      columns={columns as any}
      data={data as any}
      isLoading={isLoading}
      renderSubComponent={renderSubComponent}
      getRowCanExpand={getRowCanExpand}
      loadingMessage="🚅 Loading tools..."
      noDataMessage="No tools found"
    />
  );
}

const MCPToolsViewer = ({
  serverId,
  accessToken,
  auth_type,
  userRole,
  userID,
}: MCPToolsViewerProps) => {
  const [searchTerm, setSearchTerm] = useState("");
  const [mcpAuthValue, setMcpAuthValue] = useState("");
  const [selectedTool, setSelectedTool] = useState<MCPTool | null>(null);
  const [toolResult, setToolResult] = useState<CallMCPToolResponse | null>(null);
  const [toolError, setToolError] = useState<Error | null>(null);

  // Query to fetch MCP tools
  const { data: mcpToolsResponse, isLoading: isLoadingTools, error: mcpToolsError } = useQuery({
    queryKey: ["mcpTools"],
    queryFn: () => {
      if (!accessToken) throw new Error("Access Token required");
      return listMCPTools(accessToken, serverId);
    },
    enabled: !!accessToken,
  });

  // Mutation for calling a tool
  const { mutate: executeTool, isPending: isCallingTool } = useMutation({
    mutationFn: (args: { tool: MCPTool; arguments: Record<string, any>, authValue: string }) => {
      if (!accessToken) throw new Error("Access Token required");
      return callMCPTool(accessToken, args.tool.name, args.arguments, args.authValue);
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

  // Add onToolSelect handler to each tool
  const toolsData = React.useMemo(() => {
    if (!mcpToolsResponse) return [];

    return (mcpToolsResponse.tools || []).map((tool: MCPTool) => ({
      ...tool,
      onToolSelect: (tool: MCPTool) => {
        setSelectedTool(tool);
        setToolResult(null);
        setToolError(null);
      },
    }));
  }, [mcpToolsResponse]);

  // Error message display
  const errorMessage = mcpToolsResponse?.error ? (
    <div className="p-4 mb-4 text-sm text-red-800 rounded-lg bg-red-50">
      <p className="font-medium">Error: {mcpToolsResponse.message}</p>
      {mcpToolsResponse.error === "server_not_found" && (
        <p className="mt-2">The specified server could not be found. Please check the server ID and try again.</p>
      )}
      {mcpToolsResponse.error === "server_error" && (
        <p className="mt-2">There was an error connecting to the server. Please try again later or contact support if the issue persists.</p>
      )}
      {mcpToolsResponse.error === "unexpected_error" && (
        <p className="mt-2">An unexpected error occurred. Please try again later or contact support if the issue persists.</p>
      )}
    </div>
  ) : null;

  // No tools message
  const noToolsMessage = !isLoadingTools && !mcpToolsResponse?.error && (!toolsData || toolsData.length === 0) ? (
    <div className="p-4 mb-4 text-sm text-gray-800 rounded-lg bg-gray-50">
      <p className="font-medium">No tools available</p>
      <p className="mt-2">No tools were found for this server. This could be because:</p>
      <ul className="list-disc list-inside mt-2">
        <li>The server has not registered any tools</li>
        <li>There was an error connecting to the server</li>
        <li>The server is still initializing</li>
      </ul>
    </div>
  ) : null;

  return (
    <div className="space-y-4">
      {errorMessage}
      {noToolsMessage}
      <DataTableWrapper
        columns={columns}
        data={toolsData}
        isLoading={isLoadingTools}
      />
      {selectedTool && (
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
      )}
      {mcpServerHasAuth(auth_type) && (
        <AuthSection
          authType={auth_type}
          onAuthSubmit={(value) => setMcpAuthValue(value)}
        />
      )}
    </div>
  );
};

export default MCPToolsViewer;
