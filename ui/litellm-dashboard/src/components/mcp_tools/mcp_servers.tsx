import React, { useState, useEffect } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";

import {
  InformationCircleIcon,
  PencilAltIcon,
  PencilIcon,
  RefreshIcon,
  StatusOnlineIcon,
  TrashIcon,
} from "@heroicons/react/outline";

import {
  Button as Button2,
  Modal,
  Form,
  Input,
  Select as Select2,
  message,
  Tooltip,
} from "antd";

import {
  Title,
  Button as TremorButton,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeaderCell,
  TableRow,
  TextInput,
  Card,
  Icon,
  Button,
  Badge,
  Col,
  Text,
  Grid,
  Accordion,
  AccordionHeader,
  AccordionBody,
  TabGroup,
  TabList,
  TabPanel,
  TabPanels,
  Tab,
} from "@tremor/react";

import { deleteMCPServer, fetchMCPServers } from "../networking";
import {
  MCPServer,
  MCPServerProps,
  MCPTool,
  MCPToolsViewerProps,
  CallMCPToolResponse,
} from "./types";
import { isAdminRole } from "@/utils/roles";

const TRANSPORT = {
  SSE: "sse",
  HTTP: "http",
};

const handleTransport = (transport?: string) => {
  if (transport === null || transport === undefined) {
    return TRANSPORT.SSE;
  }

  return transport;
};

const displayFriendlyId = (id: string) => {
  return `${id.slice(0, 7)}...`;
};

const createMCPServer = (server: MCPServer) => {
  return {
    server_id: server.server_id,
    alias: server.alias,
    transport: server.transport,
    sort_by: "created_at",
    sort_order: "desc",
  };
};

const handleEdit = (server_id: string) => {};

interface CreateMCPServerProps {
  userRole: string;
  setServerCreate: (value: boolean) => void;
}

const CreateMCPServer: React.FC<CreateMCPServerProps> = ({
  userRole,
  setServerCreate,
}) => {
  const [isModalAddVisible, setModalAddVisible] = useState(false);

  if (!isAdminRole(userRole)) {
    return null;
  }

  return (
    <div>
      <Button className="mx-auto" onClick={() => setServerCreate(true)}>
        + Add New MCP Server
      </Button>
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
    <div className="fixed z-10 inset-0 overflow-y-auto">
      <div className="flex items-end justify-center min-h-screen pt-4 px-4 pb-20 text-center sm:block sm:p-0">
        <div className="fixed inset-0 transition-opacity" aria-hidden="true">
          <div className="absolute inset-0 bg-gray-500 opacity-75"></div>
        </div>

        {/* Modal Panel */}
        <span
          className="hidden sm:inline-block sm:align-middle sm:h-screen"
          aria-hidden="true"
        >
          &#8203;
        </span>

        {/* Confirmation Modal Content */}
        <div className="inline-block align-bottom bg-white rounded-lg text-left overflow-hidden shadow-xl transform transition-all sm:my-8 sm:align-middle sm:max-w-lg sm:w-full">
          <div className="bg-white px-4 pt-5 pb-4 sm:p-6 sm:pb-4">
            <div className="sm:flex sm:items-start">
              <div className="mt-3 text-center sm:mt-0 sm:ml-4 sm:text-left">
                <h3 className="text-lg leading-6 font-medium text-gray-900">
                  {title}
                </h3>
                <div className="mt-2">
                  <p className="text-sm text-gray-500">
                    Are you sure you want to delete this MCP Server?
                  </p>
                </div>
              </div>
            </div>
          </div>
          <div className="bg-gray-50 px-4 py-3 sm:px-6 sm:flex sm:flex-row-reverse">
            <Button onClick={confirmDelete} color="red" className="ml-2">
              Delete
            </Button>
            <Button onClick={cancelDelete}>Cancel</Button>
          </div>
        </div>
      </div>
    </div>
  );
};

interface MCPServerViewProps {
  mcpServer: MCPServer;
  onBack: () => void;
  isProxyAdmin: boolean;
  isEditing: boolean;
}

const MCPServerView: React.FC<MCPServerViewProps> = ({
  mcpServer,
  onBack,
  isEditing,
  isProxyAdmin,
}) => {
  return (
    <div className="p-4">
      <div className="flex justify-between items-center mb-6">
        <div>
          <Button onClick={onBack} className="mb-4">
            ← Back
          </Button>
          <Title>{mcpServer.alias}</Title>
          <Text className="text-gray-500 font-mono">{mcpServer.server_id}</Text>
        </div>
      </div>
    </div>
  );
};

interface FilterState {
  server_id: string;
  alias: string;
  transport: string;
  sort_by: string;
  sort_order: "asc" | "desc";
}

const MCPServers: React.FC<MCPServerProps> = ({
  accessToken,
  userRole,
  userID,
  mcp_servers,
}) => {
  const [filters, setFilters] = useState<FilterState>({
    server_id: "",
    alias: "",
    transport: "",
    sort_by: "created_at",
    sort_order: "desc",
  });
  const [lastRefreshed, setLastRefreshed] = useState("");
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
      fetchMCPServers(accessToken);
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

  // Query to fetch MCP tools
  const { data: mcpServers, isLoading: isLoadingServers } = useQuery({
    queryKey: ["mcpServers"],
    queryFn: () => {
      if (!accessToken) throw new Error("Access Token required");
      return fetchMCPServers(accessToken);
    },
    enabled: !!accessToken,
  });

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
                <TableHeaderCell>Url</TableHeaderCell>
                <TableHeaderCell>Created</TableHeaderCell>
                <TableHeaderCell>Info</TableHeaderCell>
              </TableRow>
            </TableHead>

            <TableBody>
              {!mcpServers || mcpServers.length == 0
                ? []
                : mcpServers.map((mcpServer: MCPServer) => (
                    <TableRow key={mcpServer.alias}>
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
                        {mcpServer.transport}
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
            setServerCreate={(value) => {
              setEditServer(value);
            }}
          />
        </div>
      )}
    </div>
  );
};

export default MCPServers;
