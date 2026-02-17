import React, { useState } from "react";
import { ArrowLeftIcon, EyeIcon, EyeOffIcon } from "@heroicons/react/outline";
import { Title, Card, Button, Text, Grid, TabGroup, TabList, TabPanel, TabPanels, Tab, Icon } from "@tremor/react";

import { MCPServer, handleTransport, handleAuth } from "./types";
// TODO: Move Tools viewer from index file
import { MCPToolsViewer } from ".";
import MCPServerEdit from "./mcp_server_edit";
import MCPServerCostDisplay from "./mcp_server_cost_display";
import { getMaskedAndFullUrl } from "./utils";
import { copyToClipboard as utilCopyToClipboard } from "@/utils/dataUtils";
import { CheckIcon, CopyIcon } from "lucide-react";
import { Button as AntdButton } from "antd";

interface MCPServerViewProps {
  mcpServer: MCPServer;
  onBack: () => void;
  isProxyAdmin: boolean;
  isEditing: boolean;
  accessToken: string | null;
  userRole: string | null;
  userID: string | null;
  availableAccessGroups: string[];
}

export const MCPServerView: React.FC<MCPServerViewProps> = ({
  mcpServer,
  onBack,
  isEditing,
  isProxyAdmin,
  accessToken,
  userRole,
  userID,
  availableAccessGroups,
}) => {
  const [editing, setEditing] = useState(isEditing);
  const [showFullUrl, setShowFullUrl] = useState(false);
  const [copiedStates, setCopiedStates] = useState<Record<string, boolean>>({});
  const [selectedTabIndex, setSelectedTabIndex] = useState(0);

  const handleSuccess = (updated: MCPServer) => {
    setEditing(false);
    onBack();
  };

  const urlValue = mcpServer.url ?? "";
  const { maskedUrl, hasToken } = urlValue ? getMaskedAndFullUrl(urlValue) : { maskedUrl: "—", hasToken: false };

  const renderUrlWithToggle = (url: string | null | undefined, showFull: boolean) => {
    if (!url) return "—";
    if (!hasToken) return url;
    return showFull ? url : maskedUrl;
  };

  const copyToClipboard = async (text: string | null | undefined, key: string) => {
    const success = await utilCopyToClipboard(text);
    if (success) {
      setCopiedStates((prev) => ({ ...prev, [key]: true }));
      setTimeout(() => {
        setCopiedStates((prev) => ({ ...prev, [key]: false }));
      }, 2000);
    }
  };

  return (
    <div className="p-4 max-w-full">
      <div className="flex justify-between items-center mb-6">
        <div>
          <Button icon={ArrowLeftIcon} variant="light" className="mb-4" onClick={onBack}>
            Back to All Servers
          </Button>
          <div className="flex items-center cursor-pointer">
            <Title>{mcpServer.server_name}</Title>
            <AntdButton
              type="text"
              size="small"
              icon={copiedStates["mcp-server_name"] ? <CheckIcon size={12} /> : <CopyIcon size={12} />}
              onClick={() => copyToClipboard(mcpServer.server_name, "mcp-server_name")}
              className={`left-2 z-10 transition-all duration-200 ${copiedStates["mcp-server_name"]
                ? "text-green-600 bg-green-50 border-green-200"
                : "text-gray-500 hover:text-gray-700 hover:bg-gray-100"
                }`}
            />
            {mcpServer.alias && (
              <>
                <span className="ml-4 text-gray-500">Alias:</span>
                <span className="ml-1 font-mono text-blue-600">{mcpServer.alias}</span>
                <AntdButton
                  type="text"
                  size="small"
                  icon={copiedStates["mcp-alias"] ? <CheckIcon size={12} /> : <CopyIcon size={12} />}
                  onClick={() => copyToClipboard(mcpServer.alias, "mcp-alias")}
                  className={`left-2 z-10 transition-all duration-200 ${copiedStates["mcp-alias"]
                    ? "text-green-600 bg-green-50 border-green-200"
                    : "text-gray-500 hover:text-gray-700 hover:bg-gray-100"
                    }`}
                />
              </>
            )}
          </div>
          <div className="flex items-center cursor-pointer">
            <Text className="text-gray-500 font-mono">{mcpServer.server_id}</Text>
            <AntdButton
              type="text"
              size="small"
              icon={copiedStates["mcp-server-id"] ? <CheckIcon size={12} /> : <CopyIcon size={12} />}
              onClick={() => copyToClipboard(mcpServer.server_id, "mcp-server-id")}
              className={`left-2 z-10 transition-all duration-200 ${copiedStates["mcp-server-id"]
                ? "text-green-600 bg-green-50 border-green-200"
                : "text-gray-500 hover:text-gray-700 hover:bg-gray-100"
                }`}
            />
          </div>
        </div>
      </div>

      {/* TODO: magic number for index */}
      <TabGroup index={selectedTabIndex} onIndexChange={setSelectedTabIndex}>
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
                    <button onClick={() => setShowFullUrl(!showFullUrl)} className="p-1 hover:bg-gray-100 rounded">
                      <Icon icon={showFullUrl ? EyeOffIcon : EyeIcon} size="sm" className="text-gray-500" />
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
              serverAlias={mcpServer.alias}
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
                  availableAccessGroups={availableAccessGroups}
                />
              ) : (
                <div className="space-y-4">
                  <div>
                    <Text className="font-medium">Server Name</Text>
                    <div>{mcpServer.server_name}</div>
                  </div>
                  <div>
                    <Text className="font-medium">Alias</Text>
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
                        <button onClick={() => setShowFullUrl(!showFullUrl)} className="p-1 hover:bg-gray-100 rounded">
                          <Icon icon={showFullUrl ? EyeOffIcon : EyeIcon} size="sm" className="text-gray-500" />
                        </button>
                      )}
                    </div>
                  </div>
                  <div>
                    <Text className="font-medium">Transport</Text>
                    <div>{handleTransport(mcpServer.transport)}</div>
                  </div>
                  <div>
                    <Text className="font-medium">Extra Headers</Text>
                    <div>{mcpServer.extra_headers?.join(", ")}</div>
                  </div>
                  <div>
                    <Text className="font-medium">Auth Type</Text>
                    <div>{handleAuth(mcpServer.auth_type)}</div>
                  </div>
                  <div>
                    <Text className="font-medium">Allow All LiteLLM Keys</Text>
                    <div className="flex items-center gap-2">
                      {mcpServer.allow_all_keys ? (
                        <span className="px-2 py-1 bg-green-50 text-green-700 rounded-md text-sm">
                          Enabled
                        </span>
                      ) : (
                        <span className="px-2 py-1 bg-gray-100 text-gray-600 rounded-md text-sm">
                          Disabled
                        </span>
                      )}
                      {mcpServer.allow_all_keys && (
                        <Text className="text-xs text-gray-500">
                          All keys can access this MCP server
                        </Text>
                      )}
                    </div>
                  </div>
                  <div>
                    <Text className="font-medium">Available on Public Internet</Text>
                    <div className="flex items-center gap-2">
                      {mcpServer.available_on_public_internet ? (
                        <span className="px-2 py-1 bg-green-50 text-green-700 rounded-md text-sm">
                          Public
                        </span>
                      ) : (
                        <span className="px-2 py-1 bg-gray-100 text-gray-600 rounded-md text-sm">
                          Internal
                        </span>
                      )}
                      {mcpServer.available_on_public_internet && (
                        <Text className="text-xs text-gray-500">
                          Accessible from external/public IPs
                        </Text>
                      )}
                    </div>
                  </div>
                  <div>
                    <Text className="font-medium">Access Groups</Text>
                    <div>
                      {mcpServer.mcp_access_groups && mcpServer.mcp_access_groups.length > 0 ? (
                        <div className="flex flex-wrap gap-2">
                          {mcpServer.mcp_access_groups.map((group: any, index: number) => (
                            <span key={index} className="px-2 py-1 bg-gray-100 rounded-md text-sm">
                              {typeof group === "string" ? group : group?.name ?? ""}
                            </span>
                          ))}
                        </div>
                      ) : (
                        <Text className="text-gray-500">No access groups defined</Text>
                      )}
                    </div>
                  </div>
                  <div>
                    <Text className="font-medium">Allowed Tools</Text>
                    <div>
                      {mcpServer.allowed_tools && mcpServer.allowed_tools.length > 0 ? (
                        <div className="flex flex-wrap gap-2">
                          {mcpServer.allowed_tools.map((tool: string, index: number) => (
                            <span
                              key={index}
                              className="px-2 py-1 bg-blue-50 border border-blue-200 rounded-md text-sm"
                            >
                              {tool}
                            </span>
                          ))}
                        </div>
                      ) : (
                        <Text className="text-gray-500">All tools enabled</Text>
                      )}
                    </div>
                  </div>
                  <div>
                    <Text className="font-medium">Cost Configuration</Text>
                    <MCPServerCostDisplay costConfig={mcpServer.mcp_info?.mcp_server_cost_info} />
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
