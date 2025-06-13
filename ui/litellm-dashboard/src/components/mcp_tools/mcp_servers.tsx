import React, { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";

import { PencilAltIcon, TrashIcon } from "@heroicons/react/outline";

import {
  Modal,
  Tooltip,
  message,
} from "antd";

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
import CreateMCPServer from "./create_mcp_server";

const displayFriendlyId = (id: string) => {
  return `${id.slice(0, 7)}...`;
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
          <div className="bg-white rounded-lg shadow">
            <div className="rounded-lg custom-border">
              <Table className="[&_td]:py-0.5 [&_th]:py-1">
                <TableHead>
                  <TableRow>
                    <TableHeaderCell className="py-1 h-8">Server ID</TableHeaderCell>
                    <TableHeaderCell className="py-1 h-8">Server Name</TableHeaderCell>
                    <TableHeaderCell className="py-1 h-8">Description</TableHeaderCell>
                    <TableHeaderCell className="py-1 h-8">Transport</TableHeaderCell>
                    <TableHeaderCell className="py-1 h-8">Auth Type</TableHeaderCell>
                    <TableHeaderCell className="py-1 h-8">Url</TableHeaderCell>
                    <TableHeaderCell className="py-1 h-8">Created</TableHeaderCell>
                    <TableHeaderCell className="py-1 h-8">Info</TableHeaderCell>
                  </TableRow>
                </TableHead>

                <TableBody>
                  {isLoadingServers ? (
                    <TableRow>
                      <TableCell colSpan={8} className="h-8 text-center">
                        <div className="text-center text-gray-500">
                          <p>🚅 Loading MCP servers...</p>
                        </div>
                      </TableCell>
                    </TableRow>
                  ) : !mcpServers || mcpServers.length == 0 ? (
                    <TableRow>
                      <TableCell colSpan={8} className="h-8 text-center">
                        <div className="text-center text-gray-500">
                          <p>No MCP servers found</p>
                        </div>
                      </TableCell>
                    </TableRow>
                  ) : (
                    mcpServers.map((mcpServer: MCPServer) => (
                      <TableRow key={mcpServer.server_id} className="h-8">
                        <TableCell className="py-0.5 max-h-8 overflow-hidden text-ellipsis whitespace-nowrap">
                          <Tooltip title={mcpServer.server_id}>
                            <Button
                              size="xs"
                              variant="light"
                              className="font-mono text-blue-500 bg-blue-50 hover:bg-blue-100 text-xs font-normal px-2 py-0.5 text-left overflow-hidden truncate max-w-[15ch]"
                              onClick={() => {
                                setSelectedServerId(mcpServer.server_id);
                              }}
                            >
                              {displayFriendlyId(mcpServer.server_id)}
                            </Button>
                          </Tooltip>
                        </TableCell>
                        <TableCell className="py-0.5 max-h-8 overflow-hidden text-ellipsis whitespace-nowrap">
                          <Tooltip title={mcpServer.alias || "-"}>
                            <span className="max-w-[15ch] truncate block">{mcpServer.alias || "-"}</span>
                          </Tooltip>
                        </TableCell>
                        <TableCell className="py-0.5 max-h-8 overflow-hidden text-ellipsis whitespace-nowrap">
                          <Tooltip title={mcpServer.description || "-"}>
                            <span className="max-w-[15ch] truncate block">{mcpServer.description || "-"}</span>
                          </Tooltip>
                        </TableCell>
                        <TableCell className="py-0.5 max-h-8 overflow-hidden text-ellipsis whitespace-nowrap">
                          <span>{handleTransport(mcpServer.transport)}</span>
                        </TableCell>
                        <TableCell className="py-0.5 max-h-8 overflow-hidden text-ellipsis whitespace-nowrap">
                          <span>{handleAuth(mcpServer.auth_type)}</span>
                        </TableCell>
                        <TableCell className="py-0.5 max-h-8 overflow-hidden text-ellipsis whitespace-nowrap">
                          <Tooltip title={mcpServer.url || "-"}>
                            <span className="max-w-[15ch] truncate block">{mcpServer.url || "-"}</span>
                          </Tooltip>
                        </TableCell>
                        <TableCell className="py-0.5 max-h-8 overflow-hidden text-ellipsis whitespace-nowrap">
                          <span>
                            {mcpServer.created_at
                              ? new Date(mcpServer.created_at).toLocaleDateString()
                              : "N/A"}
                          </span>
                        </TableCell>
                        <TableCell className="py-0.5 max-h-8">
                          {isAdminRole(userRole) ? (
                            <div className="flex items-center space-x-1">
                              <Icon
                                icon={PencilAltIcon}
                                size="sm"
                                className="cursor-pointer hover:text-blue-600"
                                onClick={() => {
                                  setSelectedServerId(mcpServer.server_id);
                                  setEditServer(true);
                                }}
                              />
                              <Icon
                                onClick={() => handleDelete(mcpServer.server_id)}
                                icon={TrashIcon}
                                size="sm"
                                className="cursor-pointer hover:text-red-600"
                              />
                            </div>
                          ) : (
                            <span>-</span>
                          )}
                        </TableCell>
                      </TableRow>
                    ))
                  )}
                </TableBody>
              </Table>
            </div>
          </div>
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
