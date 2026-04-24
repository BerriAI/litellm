import React, { useState } from "react";
import { ArrowLeft, Check, Copy, Eye, EyeOff } from "lucide-react";
import {
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
} from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { cn } from "@/lib/utils";

import { MCPServer, handleTransport, handleAuth } from "./types";
// TODO: Move Tools viewer from index file
import { MCPToolsViewer } from ".";
import MCPServerEdit from "./mcp_server_edit";
import MCPServerCostDisplay from "./mcp_server_cost_display";
import { getMaskedAndFullUrl } from "./utils";
import { copyToClipboard as utilCopyToClipboard } from "@/utils/dataUtils";

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
  const [selectedTab, setSelectedTab] = useState<string>("overview");

  const handleSuccess = () => {
    setEditing(false);
    onBack();
  };

  const urlValue = mcpServer.url ?? "";
  const { maskedUrl, hasToken } = urlValue
    ? getMaskedAndFullUrl(urlValue)
    : { maskedUrl: "—", hasToken: false };

  const renderUrlWithToggle = (
    url: string | null | undefined,
    showFull: boolean,
  ) => {
    if (!url) return "—";
    if (!hasToken) return url;
    return showFull ? url : maskedUrl;
  };

  const copyToClipboard = async (
    text: string | null | undefined,
    key: string,
  ) => {
    const success = await utilCopyToClipboard(text);
    if (success) {
      setCopiedStates((prev) => ({ ...prev, [key]: true }));
      setTimeout(() => {
        setCopiedStates((prev) => ({ ...prev, [key]: false }));
      }, 2000);
    }
  };

  const getTransportBadge = (transport: string) => {
    const label = transport.toUpperCase();
    return (
      <span className="inline-flex items-center text-sm font-medium px-2.5 py-0.5 rounded border bg-muted text-foreground/90 border-border">
        {label}
      </span>
    );
  };

  const getAuthBadge = (authType: string) => {
    return (
      <span className="inline-flex items-center text-sm font-medium px-2.5 py-0.5 rounded border bg-muted text-foreground/90 border-border">
        {authType}
      </span>
    );
  };

  return (
    <div className="p-4 max-w-full">
      <div className="mb-6">
        <Button variant="ghost" className="mb-4" onClick={onBack}>
          <ArrowLeft className="h-4 w-4" />
          Back to All Servers
        </Button>
        <div className="flex items-center gap-2">
          <h1 className="text-2xl font-semibold">
            {mcpServer.server_name || mcpServer.alias || "Unnamed Server"}
          </h1>
          <Button
            variant="ghost"
            size="icon"
            className={cn(
              "h-7 w-7 transition-all duration-200",
              copiedStates["mcp-server_name"]
                ? "text-emerald-600 bg-emerald-50 dark:bg-emerald-950/30"
                : "text-muted-foreground hover:text-foreground",
            )}
            onClick={() =>
              copyToClipboard(
                mcpServer.server_name || mcpServer.alias,
                "mcp-server_name",
              )
            }
            aria-label="Copy server name"
          >
            {copiedStates["mcp-server_name"] ? (
              <Check className="h-3 w-3" />
            ) : (
              <Copy className="h-3 w-3" />
            )}
          </Button>
          {mcpServer.alias &&
            mcpServer.server_name &&
            mcpServer.alias !== mcpServer.server_name && (
              <span className="ml-2 inline-flex items-center text-xs font-medium px-2 py-0.5 rounded bg-muted text-muted-foreground border border-border font-mono">
                {mcpServer.alias}
              </span>
            )}
        </div>
        <div className="flex items-center gap-1.5 mt-1">
          <span className="text-muted-foreground font-mono text-xs">
            {mcpServer.server_id}
          </span>
          <Button
            variant="ghost"
            size="icon"
            className={cn(
              "h-5 w-5 transition-all duration-200",
              copiedStates["mcp-server-id"]
                ? "text-emerald-600 bg-emerald-50 dark:bg-emerald-950/30"
                : "text-muted-foreground hover:text-foreground",
            )}
            onClick={() => copyToClipboard(mcpServer.server_id, "mcp-server-id")}
            aria-label="Copy server ID"
          >
            {copiedStates["mcp-server-id"] ? (
              <Check className="h-3 w-3" />
            ) : (
              <Copy className="h-3 w-3" />
            )}
          </Button>
        </div>
        {mcpServer.description && (
          <p className="text-muted-foreground mt-2">{mcpServer.description}</p>
        )}
      </div>

      <Tabs value={selectedTab} onValueChange={setSelectedTab}>
        <TabsList className="mb-4 bg-transparent p-0 h-auto">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="tools">MCP Tools</TabsTrigger>
          {isProxyAdmin && <TabsTrigger value="settings">Settings</TabsTrigger>}
        </TabsList>

        {/* Overview Panel */}
        <TabsContent value="overview">
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
              <Card className="p-4">
                <span className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
                  Transport
                </span>
                <div className="mt-3">
                  {getTransportBadge(
                    handleTransport(
                      mcpServer.transport ?? undefined,
                      mcpServer.spec_path ?? undefined,
                    ),
                  )}
                </div>
              </Card>

              <Card className="p-4">
                <span className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
                  Authentication
                </span>
                <div className="mt-3">
                  {getAuthBadge(handleAuth(mcpServer.auth_type ?? undefined))}
                </div>
              </Card>

              <Card className="p-4">
                <span className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
                  Host URL
                </span>
                <div className="mt-3 flex items-center gap-2">
                  <span className="break-all overflow-wrap-anywhere font-mono text-sm">
                    {renderUrlWithToggle(mcpServer.url, showFullUrl)}
                  </span>
                  {hasToken && (
                    <button
                      type="button"
                      onClick={() => setShowFullUrl(!showFullUrl)}
                      className="p-1 hover:bg-muted rounded flex-shrink-0"
                    >
                      {showFullUrl ? (
                        <EyeOff className="h-4 w-4 text-muted-foreground" />
                      ) : (
                        <Eye className="h-4 w-4 text-muted-foreground" />
                      )}
                    </button>
                  )}
                </div>
              </Card>
            </div>
            <Card className="mt-4 p-4">
              <span className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
                Cost Configuration
              </span>
              <div className="mt-3">
                <MCPServerCostDisplay
                  costConfig={mcpServer.mcp_info?.mcp_server_cost_info}
                />
              </div>
            </Card>
        </TabsContent>

        {/* Tool Panel */}
        <TabsContent value="tools">
          <MCPToolsViewer
            serverId={mcpServer.server_id}
            accessToken={accessToken}
            auth_type={mcpServer.auth_type}
            userRole={userRole}
            userID={userID}
            serverAlias={mcpServer.alias}
            extraHeaders={mcpServer.extra_headers}
          />
        </TabsContent>

        {/* Settings Panel */}
        <TabsContent value="settings">
            <Card className="p-4">
              <div className="flex justify-between items-center mb-4">
                <h2 className="text-lg font-semibold">MCP Server Settings</h2>
                {editing ? null : (
                  <Button variant="ghost" onClick={() => setEditing(true)}>
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
                <div className="divide-y divide-border">
                  <div className="py-3 grid grid-cols-3 gap-4">
                    <span className="text-sm font-medium text-muted-foreground">
                      Server Name
                    </span>
                    <div className="col-span-2 text-sm text-foreground">
                      {mcpServer.server_name || (
                        <span className="text-muted-foreground">—</span>
                      )}
                    </div>
                  </div>
                  <div className="py-3 grid grid-cols-3 gap-4">
                    <span className="text-sm font-medium text-muted-foreground">
                      Alias
                    </span>
                    <div className="col-span-2 text-sm font-mono text-foreground">
                      {mcpServer.alias || (
                        <span className="text-muted-foreground">—</span>
                      )}
                    </div>
                  </div>
                  <div className="py-3 grid grid-cols-3 gap-4">
                    <span className="text-sm font-medium text-muted-foreground">
                      Description
                    </span>
                    <div className="col-span-2 text-sm text-foreground">
                      {mcpServer.description || (
                        <span className="text-muted-foreground">—</span>
                      )}
                    </div>
                  </div>
                  <div className="py-3 grid grid-cols-3 gap-4">
                    <span className="text-sm font-medium text-muted-foreground">
                      URL
                    </span>
                    <div className="col-span-2 text-sm font-mono text-foreground break-all flex items-center gap-2">
                      {renderUrlWithToggle(mcpServer.url, showFullUrl)}
                      {hasToken && (
                        <button
                          type="button"
                          onClick={() => setShowFullUrl(!showFullUrl)}
                          className="p-1 hover:bg-muted rounded flex-shrink-0"
                        >
                          {showFullUrl ? (
                            <EyeOff className="h-4 w-4 text-muted-foreground" />
                          ) : (
                            <Eye className="h-4 w-4 text-muted-foreground" />
                          )}
                        </button>
                      )}
                    </div>
                  </div>
                  <div className="py-3 grid grid-cols-3 gap-4">
                    <span className="text-sm font-medium text-muted-foreground">
                      Transport
                    </span>
                    <div className="col-span-2">
                      {getTransportBadge(
                        handleTransport(
                          mcpServer.transport,
                          mcpServer.spec_path,
                        ),
                      )}
                    </div>
                  </div>
                  <div className="py-3 grid grid-cols-3 gap-4">
                    <span className="text-sm font-medium text-muted-foreground">
                      Authentication
                    </span>
                    <div className="col-span-2">
                      {getAuthBadge(handleAuth(mcpServer.auth_type))}
                    </div>
                  </div>
                  <div className="py-3 grid grid-cols-3 gap-4">
                    <span className="text-sm font-medium text-muted-foreground">
                      Extra Headers
                    </span>
                    <div className="col-span-2 text-sm text-foreground">
                      {mcpServer.extra_headers &&
                      mcpServer.extra_headers.length > 0 ? (
                        mcpServer.extra_headers.join(", ")
                      ) : (
                        <span className="text-muted-foreground">—</span>
                      )}
                    </div>
                  </div>
                  <div className="py-3 grid grid-cols-3 gap-4">
                    <span className="text-sm font-medium text-muted-foreground">
                      Allow All Keys
                    </span>
                    <div className="col-span-2">
                      {mcpServer.allow_all_keys ? (
                        <span className="inline-flex items-center gap-1 px-2 py-0.5 bg-emerald-50 text-emerald-700 dark:bg-emerald-950/30 dark:text-emerald-300 rounded-full border border-emerald-200 dark:border-emerald-900 text-xs font-medium">
                          <span className="h-1.5 w-1.5 rounded-full bg-emerald-500"></span>
                          Enabled
                        </span>
                      ) : (
                        <span className="inline-flex items-center gap-1 px-2 py-0.5 bg-muted text-muted-foreground rounded-full border border-border text-xs font-medium">
                          Disabled
                        </span>
                      )}
                    </div>
                  </div>
                  <div className="py-3 grid grid-cols-3 gap-4">
                    <span className="text-sm font-medium text-muted-foreground">
                      Network Access
                    </span>
                    <div className="col-span-2">
                      {mcpServer.available_on_public_internet ? (
                        <span className="inline-flex items-center gap-1 px-2 py-0.5 bg-emerald-50 text-emerald-700 dark:bg-emerald-950/30 dark:text-emerald-300 rounded-full border border-emerald-200 dark:border-emerald-900 text-xs font-medium">
                          <span className="h-1.5 w-1.5 rounded-full bg-emerald-500"></span>
                          Public
                        </span>
                      ) : (
                        <span className="inline-flex items-center gap-1 px-2 py-0.5 bg-orange-50 text-orange-700 dark:bg-orange-950/30 dark:text-orange-300 rounded-full border border-orange-200 dark:border-orange-900 text-xs font-medium">
                          <span className="h-1.5 w-1.5 rounded-full bg-orange-500"></span>
                          Internal only
                        </span>
                      )}
                    </div>
                  </div>
                  <div className="py-3 grid grid-cols-3 gap-4">
                    <span className="text-sm font-medium text-muted-foreground">
                      Access Groups
                    </span>
                    <div className="col-span-2">
                      {mcpServer.mcp_access_groups &&
                      mcpServer.mcp_access_groups.length > 0 ? (
                        <div className="flex flex-wrap gap-1.5">
                          {mcpServer.mcp_access_groups.map(
                            // eslint-disable-next-line @typescript-eslint/no-explicit-any
                            (group: any, index: number) => (
                              <span
                                key={index}
                                className="inline-flex items-center text-xs font-medium px-2 py-0.5 rounded bg-muted text-foreground/90 border border-border"
                              >
                                {typeof group === "string"
                                  ? group
                                  : (group?.name ?? "")}
                              </span>
                            ),
                          )}
                        </div>
                      ) : (
                        <span className="text-sm text-muted-foreground">
                          —
                        </span>
                      )}
                    </div>
                  </div>
                  <div className="py-3 grid grid-cols-3 gap-4">
                    <span className="text-sm font-medium text-muted-foreground">
                      Allowed Tools
                    </span>
                    <div className="col-span-2">
                      {mcpServer.allowed_tools &&
                      mcpServer.allowed_tools.length > 0 ? (
                        <div className="flex flex-wrap gap-1.5">
                          {mcpServer.allowed_tools.map(
                            (tool: string, index: number) => (
                              <span
                                key={index}
                                className="inline-flex items-center text-xs font-mono font-medium px-2 py-0.5 rounded bg-blue-50 text-blue-700 dark:bg-blue-950/30 dark:text-blue-300 border border-blue-200 dark:border-blue-900"
                              >
                                {tool}
                              </span>
                            ),
                          )}
                        </div>
                      ) : (
                        <span className="inline-flex items-center text-xs font-medium px-2 py-0.5 rounded bg-emerald-50 text-emerald-700 dark:bg-emerald-950/30 dark:text-emerald-300 border border-emerald-200 dark:border-emerald-900">
                          All tools enabled
                        </span>
                      )}
                    </div>
                  </div>
                  <div className="py-3 grid grid-cols-3 gap-4">
                    <span className="text-sm font-medium text-muted-foreground">
                      Cost
                    </span>
                    <div className="col-span-2">
                      <MCPServerCostDisplay
                        costConfig={mcpServer.mcp_info?.mcp_server_cost_info}
                      />
                    </div>
                  </div>
                </div>
              )}
            </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};
