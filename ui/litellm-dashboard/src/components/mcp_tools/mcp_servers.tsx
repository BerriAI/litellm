import React, { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";

import { PencilAltIcon, TrashIcon } from "@heroicons/react/outline";

import {
  Modal,
  Tooltip,
  Form,
  Select,
  message,
  Button as AntdButton,
} from "antd";
import { InfoCircleOutlined } from "@ant-design/icons";

import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeaderCell,
  TableRow,
  Icon,
  Button,
  Grid,
  Col,
  Title,
  TextInput,
} from "@tremor/react";

import {
  createMCPServer,
  deleteMCPServer,
  fetchMCPServers,
} from "../networking";
import {
  MCPServer,
  MCPServerProps,
  handleAuth,
  handleTransport,
} from "./types";
import { isAdminRole } from "@/utils/roles";
import { MCPServerView } from "./mcp_server_view";

const displayFriendlyId = (id: string) => {
  return `${id.slice(0, 7)}...`;
};

interface CreateMCPServerProps {
  userRole: string;
  accessToken: string | null;
  onCreateSuccess: (newMcpServer: MCPServer) => void;
}

const CreateMCPServer: React.FC<CreateMCPServerProps> = ({
  userRole,
  accessToken,
  onCreateSuccess,
}) => {
  const [form] = Form.useForm();
  const [isLoading, setIsLoading] = useState(false);

  const handleCreate = async (formValues: Record<string, any>) => {
    setIsLoading(true);
    try {
      console.log(`formValues: ${JSON.stringify(formValues)}`);

      if (accessToken != null) {
        const response: MCPServer = await createMCPServer(
          accessToken,
          formValues
        );

        message.success("MCP Server created successfully");
        form.resetFields();
        setModalVisible(false);
        onCreateSuccess(response);
      }
    } catch (error) {
      message.error("Error creating MCP Server: " + error, 20);
    } finally {
      setIsLoading(false);
    }
  };

  // state
  const [isModalVisible, setModalVisible] = useState(false);

  const handleCancel = () => {
    form.resetFields();
    setModalVisible(false);
  };

  // rendering
  if (!isAdminRole(userRole)) {
    return null;
  }

  return (
    <div>
      <Button 
        className="mx-auto" 
        onClick={() => setModalVisible(true)}
      >
        + Create New MCP Server
      </Button>

      <Modal
        title={
          <div className="flex items-center space-x-3 pb-4 border-b border-gray-100">
            <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-blue-600 rounded-lg flex items-center justify-center">
              <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 6v6m0 0v6m0-6h6m-6 0H6"/>
              </svg>
            </div>
            <div>
              <h2 className="text-xl font-semibold text-gray-900">Create New MCP Server</h2>
              <p className="text-sm text-gray-500 mt-1">Configure your MCP server connection</p>
            </div>
          </div>
        }
        open={isModalVisible}
        width={1000}
        onCancel={handleCancel}
        footer={null}
        className="top-8"
        styles={{
          body: { padding: '24px' },
          header: { padding: '24px 24px 0 24px', border: 'none' },
        }}
      >
        <div className="mt-6">
          <Form
            form={form}
            onFinish={handleCreate}
            layout="vertical"
            className="space-y-6"
          >
            <div className="grid grid-cols-1 gap-6">
              <Form.Item
                label={<span className="text-sm font-medium text-gray-700">MCP Server Name</span>}
                name="alias"
                rules={[
                  { required: false, message: "Please enter a server name" },
                ]}
              >
                <TextInput 
                  placeholder="Enter a descriptive name for your server" 
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
                label={
                  <span className="text-sm font-medium text-gray-700">
                    MCP Server URL <span className="text-red-500">*</span>
                  </span>
                }
                name="url"
                rules={[
                  { required: true, message: "Please enter a server URL" },
                  { type: 'url', message: "Please enter a valid URL" }
                ]}
              >
                <TextInput 
                  placeholder="https://your-mcp-server.com" 
                  className="rounded-lg border-gray-300 focus:border-blue-500 focus:ring-blue-500"
                />
              </Form.Item>

              <div className="grid grid-cols-2 gap-4">
                <Form.Item
                  label={
                    <span className="text-sm font-medium text-gray-700">
                      Transport Type <span className="text-red-500">*</span>
                    </span>
                  }
                  name="transport"
                  rules={[{ required: true, message: "Please select a transport type" }]}
                >
                  <Select 
                    placeholder="Select transport"
                    className="rounded-lg"
                    size="large"
                  >
                    <Select.Option value="sse">Server-Sent Events (SSE)</Select.Option>
                    <Select.Option value="http">HTTP</Select.Option>
                  </Select>
                </Form.Item>

                <Form.Item
                  label={
                    <span className="text-sm font-medium text-gray-700">
                      Authentication <span className="text-red-500">*</span>
                    </span>
                  }
                  name="auth_type"
                  rules={[{ required: true, message: "Please select an auth type" }]}
                >
                  <Select 
                    placeholder="Select auth type"
                    className="rounded-lg"
                    size="large"
                  >
                    <Select.Option value="none">None</Select.Option>
                    <Select.Option value="api_key">API Key</Select.Option>
                    <Select.Option value="bearer_token">Bearer Token</Select.Option>
                    <Select.Option value="basic">Basic Auth</Select.Option>
                  </Select>
                </Form.Item>
              </div>

              <Form.Item
                label={
                  <span className="text-sm font-medium text-gray-700 flex items-center">
                    MCP Version <span className="text-red-500 ml-1">*</span>
                    <Tooltip title="Select the MCP specification version your server supports">
                      <InfoCircleOutlined className="ml-2 text-gray-400 hover:text-gray-600" />
                    </Tooltip>
                  </span>
                }
                name="spec_version"
                rules={[
                  { required: true, message: "Please select a spec version" },
                ]}
              >
                <Select 
                  placeholder="Select MCP version"
                  className="rounded-lg"
                  size="large"
                >
                  <Select.Option value="2025-03-26">2025-03-26 (Latest)</Select.Option>
                  <Select.Option value="2024-11-05">2024-11-05</Select.Option>
                </Select>
              </Form.Item>
            </div>

            <div className="flex items-center justify-end space-x-3 pt-6 border-t border-gray-100">
              <Button 
                variant="secondary"
                onClick={handleCancel}
              >
                Cancel
              </Button>
              <Button 
                variant="primary"
                loading={isLoading}
              >
                {isLoading ? 'Creating...' : 'Create MCP Server'}
              </Button>
            </div>
          </Form>
        </div>
      </Modal>
    </div>
  );
};

interface DeleteModalProps {
  isModalOpen: boolean;
  title: string;
  confirmDelete: () => void;
  cancelDelete: () => void;
}

const DeleteModal: React.FC<DeleteModalProps> = ({
  isModalOpen,
  title,
  confirmDelete,
  cancelDelete,
}) => {
  if (!isModalOpen) return null;

  return (
    <Modal
      open={isModalOpen}
      onOk={confirmDelete}
      okType="danger"
      onCancel={cancelDelete}
    >
      <Grid numItems={1} className="gap-2 w-full">
        <Title>{title}</Title>
        <Col numColSpan={1}>
          <p>Are you sure you want to delete this MCP Server?</p>
        </Col>
      </Grid>
    </Modal>
  );
};

const MCPServers: React.FC<MCPServerProps> = ({
  accessToken,
  userRole,
  userID,
}) => {
  // Query to fetch MCP tools
  const {
    data: mcpServers,
    isLoading: isLoadingServers,
    refetch,
  } = useQuery({
    queryKey: ["mcpServers"],
    queryFn: () => {
      if (!accessToken) throw new Error("Access Token required");
      return fetchMCPServers(accessToken);
    },
    enabled: !!accessToken,
  });

  const createMCPServer = (newMcpServer: MCPServer) => {
    refetch();
  };

  // state
  const [serverIdToDelete, setServerToDelete] = useState<string | null>(null);
  const [isDeleteModalOpen, setIsDeleteModalOpen] = useState(false);
  const [selectedServerId, setSelectedServerId] = useState<string | null>(null);
  const [editServer, setEditServer] = useState(false);

  const handleDelete = (server_id: string) => {
    // Set the team to delete and open the confirmation modal
    setServerToDelete(server_id);
    setIsDeleteModalOpen(true);
  };

  const confirmDelete = async () => {
    if (serverIdToDelete == null || accessToken == null) {
      return;
    }

    try {
      await deleteMCPServer(accessToken, serverIdToDelete);
      // Successfully completed the deletion. Update the state to trigger a rerender.
      message.success("Deleted MCP Server successfully");
      refetch();
    } catch (error) {
      console.error("Error deleting the mcp server:", error);
      // Handle any error situations, such as displaying an error message to the user.
    }

    // Close the confirmation modal and reset the serverToDelete
    setIsDeleteModalOpen(false);
    setServerToDelete(null);
  };

  const cancelDelete = () => {
    // Close the confirmation modal and reset the serverToDelete
    setIsDeleteModalOpen(false);
    setServerToDelete(null);
  };

  if (!accessToken || !userRole || !userID) {
    return (
      <div className="p-6 text-center text-gray-500">
        Missing required authentication parameters.
      </div>
    );
  }

  return (
    <div className="w-full mx-4 h-[75vh]">
      {selectedServerId ? (
        <MCPServerView
          mcpServer={
            mcpServers.find(
              (server: MCPServer) => server.server_id === selectedServerId
            ) || {}
          }
          onBack={() => setSelectedServerId(null)}
          isProxyAdmin={isAdminRole(userRole)}
          isEditing={editServer}
          accessToken={accessToken}
          userID={userID}
          userRole={userRole}
        />
      ) : (
        <div className="w-full p-6">
          <div className="flex items-center justify-between mb-4">
            <h1 className="text-xl font-semibold">MCP Servers</h1>
          </div>
          <Table>
            <TableHead>
              <TableRow>
                <TableHeaderCell>Server ID</TableHeaderCell>
                <TableHeaderCell>Server Name</TableHeaderCell>
                <TableHeaderCell>Description</TableHeaderCell>
                <TableHeaderCell>Transport</TableHeaderCell>
                <TableHeaderCell>Auth Type</TableHeaderCell>
                <TableHeaderCell>Url</TableHeaderCell>
                <TableHeaderCell>Created</TableHeaderCell>
                <TableHeaderCell>Info</TableHeaderCell>
              </TableRow>
            </TableHead>

            <TableBody>
              {!mcpServers || mcpServers.length == 0
                ? []
                : mcpServers.map((mcpServer: MCPServer) => (
                    <TableRow key={mcpServer.server_id}>
                      <TableCell>
                        <div className="overflow-hidden">
                          <Tooltip title={mcpServer.server_id}>
                            <Button
                              size="xs"
                              variant="light"
                              className="font-mono text-blue-500 bg-blue-50 hover:bg-blue-100 text-xs font-normal px-2 py-0.5 text-left overflow-hidden truncate max-w-[200px]"
                              onClick={() => {
                                // Add click handler
                                setSelectedServerId(mcpServer.server_id);
                              }}
                            >
                              {displayFriendlyId(mcpServer.server_id)}
                            </Button>
                          </Tooltip>
                        </div>
                      </TableCell>
                      <TableCell
                        style={{
                          maxWidth: "4px",
                          whiteSpace: "pre-wrap",
                          overflow: "hidden",
                        }}
                      >
                        {mcpServer.alias}
                      </TableCell>
                      <TableCell
                        style={{
                          maxWidth: "4px",
                          whiteSpace: "pre-wrap",
                          overflow: "hidden",
                        }}
                      >
                        {mcpServer.description}
                      </TableCell>
                      <TableCell
                        style={{
                          maxWidth: "4px",
                          whiteSpace: "pre-wrap",
                          overflow: "hidden",
                        }}
                      >
                        {handleTransport(mcpServer.transport)}
                      </TableCell>
                      <TableCell
                        style={{
                          maxWidth: "4px",
                          whiteSpace: "pre-wrap",
                          overflow: "hidden",
                        }}
                      >
                        {handleAuth(mcpServer.auth_type)}
                      </TableCell>
                      <TableCell>
                        <div className="overflow-hidden">
                          <Tooltip title={mcpServer.url}>
                            {mcpServer.url}
                          </Tooltip>
                        </div>
                      </TableCell>
                      <TableCell
                        style={{
                          maxWidth: "4px",
                          whiteSpace: "pre-wrap",
                          overflow: "hidden",
                        }}
                      >
                        {mcpServer.created_at
                          ? new Date(mcpServer.created_at).toLocaleDateString()
                          : "N/A"}
                      </TableCell>
                      <TableCell>
                        {isAdminRole(userRole) ? (
                          <>
                            <Icon
                              icon={PencilAltIcon}
                              size="sm"
                              onClick={() => {
                                setSelectedServerId(mcpServer.server_id);
                                setEditServer(true);
                              }}
                            />
                            <Icon
                              onClick={() => handleDelete(mcpServer.server_id)}
                              icon={TrashIcon}
                              size="sm"
                            />
                          </>
                        ) : null}
                      </TableCell>
                    </TableRow>
                  ))}
            </TableBody>
          </Table>
          <DeleteModal
            isModalOpen={isDeleteModalOpen}
            title="Delete MCP Server"
            confirmDelete={confirmDelete}
            cancelDelete={cancelDelete}
          />
          <CreateMCPServer
            userRole={userRole}
            accessToken={accessToken}
            onCreateSuccess={createMCPServer}
          />
        </div>
      )}
    </div>
  );
};

export default MCPServers;
