import React from "react";

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
      <TabGroup defaultIndex={isEditing ? 2 : 0}>
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
                <Title>Editing MCP Servers coming soon!</Title>
              </div>
            </Card>
          </TabPanel>
        </TabPanels>
      </TabGroup>
    </div>
  );
};
