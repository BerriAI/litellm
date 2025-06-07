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

  const handleCreate = async (formValues: Record<string, any>) => {
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
      message.error("Error creating the team: " + error, 20);
    }
  };

  // state
  const [isModalVisible, setModalVisible] = useState(false);

  // rendering
  if (!isAdminRole(userRole)) {
    return null;
  }

  return (
    <div>
      <Button className="mx-auto" onClick={() => setModalVisible(true)}>
        + Create New MCP Server
      </Button>

      <Modal
        title="Create New MCP Server"
        open={isModalVisible}
        okType="primary"
        width={800}
        onCancel={() => setModalVisible(false)}
        okButtonProps={{ style: { display: "none" } }}
      >
        <Form
          form={form}
          onFinish={handleCreate}
          labelCol={{ span: 8 }}
          wrapperCol={{ span: 16 }}
          labelAlign="left"
        >
          <>
            <Form.Item
              label="MCP Server Name"
              name="alias"
              rules={[
                { required: false, message: "Please enter a server name" },
              ]}
            >
              <TextInput placeholder="" />
            </Form.Item>
            <Form.Item
              label="MCP Description"
              name="description"
              rules={[
                {
                  required: false,
                  message: "Please enter a server description",
                },
              ]}
            >
              <TextInput placeholder="" />
            </Form.Item>
            <Form.Item
              label="MCP Server URL"
              name="url"
              rules={[{ required: true, message: "Please enter a server url" }]}
            >
              <TextInput placeholder="https://" />
            </Form.Item>
            <Form.Item
              label="MCP Server Transport"
              name="transport"
              rules={[{ required: true, message: "Please enter a server url" }]}
            >
              <Select placeholder="MCP Transport Type">
                <Select.Option value="sse">sse</Select.Option>
                <Select.Option value="http">http</Select.Option>
              </Select>
            </Form.Item>
            <Form.Item
              label="MCP Server Auth Type"
              name="auth_type"
              rules={[{ required: true, message: "Please enter an auth type" }]}
            >
              <Select placeholder="MCP Auth Type">
                <Select.Option value="none">None</Select.Option>
                <Select.Option value="api_key">api_key</Select.Option>
                <Select.Option value="bearer_token">bearer_token</Select.Option>
                <Select.Option value="basic">basic</Select.Option>
              </Select>
            </Form.Item>
            <Form.Item
              label={
                <span>
                  MCP Version{" "}
                  <Tooltip title="Supported MCP Specification Version">
                    <InfoCircleOutlined style={{ marginLeft: "4px" }} />
                  </Tooltip>
                </span>
              }
              name="spec_version"
              rules={[
                { required: true, message: "Please enter a spec version" },
              ]}
            >
              <Select placeholder="MCP Version">
                <Select.Option value="2025-03-26">2025-03-26</Select.Option>
                <Select.Option value="2024-11-05">2024-11-05</Select.Option>
              </Select>
            </Form.Item>
          </>
          <div
            style={{
              textAlign: "right",
              marginTop: "10px",
              paddingBottom: "10px",
            }}
          >
            <AntdButton htmlType="submit">Create MCP Server</AntdButton>
          </div>
        </Form>
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
