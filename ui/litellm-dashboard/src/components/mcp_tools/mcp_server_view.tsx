import React, { useState } from "react";

import {
  Title,
  Card,
  Button,
  Text,
  Grid,
  TabGroup,
  TabList,
  TabPanel,
  TabPanels,
  Tab,
} from "@tremor/react";

import { MCPServer, handleTransport, handleAuth } from "./types";
// TODO: Move Tools viewer from index file
import { MCPToolsViewer } from ".";
import MCPServerEdit from "./mcp_server_edit";

interface MCPServerViewProps {
  mcpServer: MCPServer;
  onBack: () => void;
  isProxyAdmin: boolean;
  isEditing: boolean;
  accessToken: string | null;
  userRole: string | null;
  userID: string | null;
}

export const MCPServerView: React.FC<MCPServerViewProps> = ({
  mcpServer,
  onBack,
  isEditing,
  isProxyAdmin,
  accessToken,
  userRole,
  userID,
}) => {
  const [editing, setEditing] = useState(isEditing);

  const handleSuccess = (updated: MCPServer) => {
    setEditing(false);
    onBack();
  };

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

      {/* TODO: magic number for index */}
      <TabGroup defaultIndex={editing ? 2 : 0}>
        <TabList className="mb-4">
          {[
            <Tab key="overview">Overview</Tab>,
            <Tab key="tools">MCP Tools</Tab>,
            ...(isProxyAdmin ? [<Tab key="settings">Settings</Tab>] : []),
          ]}
        </TabList>

        <TabPanels>
          {/* Overview Panel */}
          <TabPanel>
            <Grid numItems={1} numItemsSm={2} numItemsLg={3} className="gap-6">
              <Card>
                <Text>Transport</Text>
                <div className="mt-2">
                  <Title>{handleTransport(mcpServer.transport ?? undefined)}</Title>
                </div>
              </Card>

              <Card>
                <Text>Auth Type</Text>
                <div className="mt-2">
                  <Text>{handleAuth(mcpServer.auth_type ?? undefined)}</Text>
                </div>
              </Card>

              <Card>
                <Text>Host Url</Text>
                <div className="mt-2 flex flex-wrap gap-2">{mcpServer.url}</div>
              </Card>
            </Grid>
          </TabPanel>

          {/* Tool Panel */}
          <TabPanel>
            <MCPToolsViewer
              serverId={mcpServer.server_id}
              accessToken={accessToken}
              userRole={userRole}
              userID={userID}
            />
          </TabPanel>

          {/* Settings Panel */}
          <TabPanel>
            <Card>
              <div className="flex justify-between items-center mb-4">
                <Title>MCP Server Settings</Title>
                {editing ? null : (
                  <Button variant="light" onClick={() => setEditing(true)}>
                    Edit Settings
                  </Button>
                )}
              </div>
              {editing ? (
                <MCPServerEdit
                  mcpServer={mcpServer}
                  accessToken={accessToken}
                  onCancel={() => setEditing(false)}
                  onSuccess={handleSuccess}
                />
              ) : (
                <div className="space-y-4">
                  <div>
                    <Text className="font-medium">Server Name</Text>
                    <div>{mcpServer.alias}</div>
                  </div>
                  <div>
                    <Text className="font-medium">Description</Text>
                    <div>{mcpServer.description}</div>
                  </div>
                  <div>
                    <Text className="font-medium">URL</Text>
                    <div className="font-mono">{mcpServer.url}</div>
                  </div>
                  <div>
                    <Text className="font-medium">Transport</Text>
                    <div>{handleTransport(mcpServer.transport)}</div>
                  </div>
                  <div>
                    <Text className="font-medium">Auth Type</Text>
                    <div>{handleAuth(mcpServer.auth_type)}</div>
                  </div>
                  <div>
                    <Text className="font-medium">Spec Version</Text>
                    <div>{mcpServer.spec_version}</div>
                  </div>
                </div>
              )}
            </Card>
          </TabPanel>
        </TabPanels>
      </TabGroup>
    </div>
  );
};
