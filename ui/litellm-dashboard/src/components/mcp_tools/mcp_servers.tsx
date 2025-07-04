import React, { useState } from "react";
import { useQuery } from "@tanstack/react-query";

import {
  Modal,
  Tooltip,
  message,
} from "antd";
import {
  TabPanel,
  TabPanels,
  TabGroup,
  TabList,
  Tab,
} from "@tremor/react";
import { LinkIcon } from "lucide-react";

import {
  Grid,
  Col,
  Title,
  Text,
} from "@tremor/react";
import { DataTable } from "../view_logs/table";
import { mcpServerColumns } from "./mcp_server_columns";

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
import MCPConnect from "./mcp_connect";

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

  const columns = React.useMemo(
    () =>
      mcpServerColumns(
        userRole,
        (serverId: string) => setSelectedServerId(serverId),
        (serverId: string) => {
          setSelectedServerId(serverId);
          setEditServer(true);
        },
        handleDelete
      ),
    [userRole, handleDelete]
  );

  function handleDelete(server_id: string) {
    // Set the team to delete and open the confirmation modal
    setServerToDelete(server_id);
    setIsDeleteModalOpen(true);
  }

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

  const ServersTab = () => (
    selectedServerId ? (
      <MCPServerView
        mcpServer={
          mcpServers.find(
            (server: MCPServer) => server.server_id === selectedServerId
          ) || {}
        }
        onBack={() => {
          setEditServer(false);
          setSelectedServerId(null);
          refetch();
        }}
        isProxyAdmin={isAdminRole(userRole)}
        isEditing={editServer}
        accessToken={accessToken}
        userID={userID}
        userRole={userRole}
      />
    ) : (
      <div>
        <div className="mb-2 mt-4">
          <div>
            <Title>MCP Servers</Title>
            <Text className="text-tremor-content">
              Configure and manage your MCP servers
            </Text>
          </div>
        </div>
        <CreateMCPServer
          userRole={userRole}
          accessToken={accessToken}
          onCreateSuccess={createMCPServer}
        />
        <DataTable
          columns={columns}
          data={mcpServers || []}
          renderSubComponent={() => <></>}
          getRowCanExpand={() => false}
          isLoading={isLoadingServers}
          noDataMessage="No MCP Servers found"
        />
        <DeleteModal
          isModalOpen={isDeleteModalOpen}
          title="Delete MCP Server"
          confirmDelete={confirmDelete}
          cancelDelete={cancelDelete}
        />
      </div>
    )
  );

  return (
    <div className="w-full h-[75vh] mt-8 p-6">
      <TabGroup className="h-full">
        <TabList>
          <div className="flex">
            <Tab>Servers</Tab>
            <Tab>
              <span className="flex items-center gap-2">
                <LinkIcon size={16} />
                Connect
              </span>
            </Tab>
          </div>
        </TabList>
        <TabPanels className="h-full">
          <TabPanel className="h-full">
            <ServersTab />
          </TabPanel>
          <TabPanel className="h-full">
            <MCPConnect />
          </TabPanel>
        </TabPanels>
      </TabGroup>
    </div>
  );
};

export default MCPServers;
