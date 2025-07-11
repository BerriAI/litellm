import React, { useState } from "react";
import { EyeIcon, EyeOffIcon } from "@heroicons/react/outline";
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
  Icon,
} from "@tremor/react";

import { MCPServer, handleTransport, handleAuth } from "./types";
// TODO: Move Tools viewer from index file
import { MCPToolsViewer } from ".";
import MCPServerEdit from "./mcp_server_edit";
import MCPServerCostDisplay from "./mcp_server_cost_display";
import { getMaskedAndFullUrl } from "./utils";

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
  const [showFullUrl, setShowFullUrl] = useState(false);

  const handleSuccess = (updated: MCPServer) => {
    setEditing(false);
    onBack();
  };

  const { maskedUrl, hasToken } = getMaskedAndFullUrl(mcpServer.url);

  const renderUrlWithToggle = (url: string, showFull: boolean) => {
    if (!hasToken) return url;
    return showFull ? url : maskedUrl;
  };

  return (
    <div className="p-4 max-w-full">
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
                <div className="mt-2 flex items-center gap-2">
                  <Text className="break-all overflow-wrap-anywhere">
                    {renderUrlWithToggle(mcpServer.url, showFullUrl)}
                  </Text>
                  {hasToken && (
                    <button
                      onClick={() => setShowFullUrl(!showFullUrl)}
                      className="p-1 hover:bg-gray-100 rounded"
                    >
                      <Icon
                        icon={showFullUrl ? EyeOffIcon : EyeIcon}
                        size="sm"
                        className="text-gray-500"
                      />
                    </button>
                  )}
                </div>
              </Card>
            </Grid>
            <Card className="mt-2">
              <Title>Cost Configuration</Title>
              <MCPServerCostDisplay costConfig={mcpServer.mcp_info?.mcp_server_cost_info} />
            </Card>

          </TabPanel>

          {/* Tool Panel */}
          <TabPanel>
            <MCPToolsViewer
              serverId={mcpServer.server_id}
              accessToken={accessToken}
              auth_type={mcpServer.auth_type}
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
                    <div className="font-mono break-all overflow-wrap-anywhere max-w-full flex items-center gap-2">
                      {renderUrlWithToggle(mcpServer.url, showFullUrl)}
                      {hasToken && (
                        <button
                          onClick={() => setShowFullUrl(!showFullUrl)}
                          className="p-1 hover:bg-gray-100 rounded"
                        >
                          <Icon
                            icon={showFullUrl ? EyeOffIcon : EyeIcon}
                            size="sm"
                            className="text-gray-500"
                          />
                        </button>
                      )}
                    </div>
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
                  <div>
                    <Text className="font-medium">Access Groups</Text>
                    <div>
                      {mcpServer.mcp_access_groups && mcpServer.mcp_access_groups.length > 0 ? (
                        <div className="flex flex-wrap gap-2">
                          {mcpServer.mcp_access_groups.map((group: any, index: number) => (
                            <span 
                              key={index}
                              className="px-2 py-1 bg-gray-100 rounded-md text-sm"
                            >
                              {typeof group === 'string' ? group : group?.name ?? ''}
                            </span>
                          ))}
                        </div>
                      ) : (
                        <Text className="text-gray-500">No access groups defined</Text>
                      )}
                    </div>
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
