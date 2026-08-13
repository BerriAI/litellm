import React, { useState } from "react";
import { ArrowLeft, Eye, EyeOff } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

import { MCPServer, handleTransport, handleAuth } from "@/components/mcp_tools/types";
// TODO: Move Tools viewer from index file
import { MCPToolsViewer } from ".";
import MCPServerEdit, { EDIT_OAUTH_UI_STATE_KEY } from "./mcp_server_edit";
import { getSecureItem } from "@/utils/secureStorage";
import MCPServerCostDisplay from "./mcp_server_cost_display";
import { getMaskedAndFullUrl } from "./utils";
import { copyToClipboard as utilCopyToClipboard } from "@/utils/dataUtils";
import { CheckIcon, CopyIcon } from "lucide-react";

interface MCPServerViewProps {
  mcpServer: MCPServer;
  onBack: () => void;
  isProxyAdmin: boolean;
  isEditing: boolean;
  accessToken: string | null;
  userRole: string | null;
  userID: string | null;
  availableAccessGroups: string[];
  initialTabIndex?: number;
}

// True when this render is the return from the edit-settings OAuth redirect for this
// server: the edit form wrote its UI-state snapshot before redirecting. Used to open
// the editing Settings tab on first render instead of defaulting to Overview.
function isReturningFromEditOAuth(isProxyAdmin: boolean, serverId: string): boolean {
  if (typeof window === "undefined" || !isProxyAdmin) {
    return false;
  }
  const stored = getSecureItem(EDIT_OAUTH_UI_STATE_KEY);
  if (!stored) {
    return false;
  }
  try {
    return JSON.parse(stored)?.serverId === serverId;
  } catch {
    return false;
  }
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
  initialTabIndex = 0,
}) => {
  // Open the editing Settings tab on first render when returning from the edit OAuth
  // redirect, so the "token fetched" feedback shows where the user left off (Settings=2).
  const returningFromEditOAuth = isReturningFromEditOAuth(isProxyAdmin, mcpServer.server_id);
  const [editing, setEditing] = useState(isEditing || returningFromEditOAuth);
  const [showFullUrl, setShowFullUrl] = useState(false);
  const [copiedStates, setCopiedStates] = useState<Record<string, boolean>>({});
  const [selectedTabIndex, setSelectedTabIndex] = useState(returningFromEditOAuth ? 2 : initialTabIndex);

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

  const getTransportBadge = (transport: string) => <Badge variant="outline">{transport.toUpperCase()}</Badge>;

  const getAuthBadge = (authType: string) => <Badge variant="outline">{authType}</Badge>;

  return (
    <div className="max-w-full p-4">
      <div className="mb-6">
        <Button variant="ghost" className="mb-4" onClick={onBack}>
          <ArrowLeft />
          Back to All Servers
        </Button>
        <div className="flex items-center gap-2">
          <h1 className="text-2xl font-semibold">{mcpServer.server_name || mcpServer.alias || "Unnamed Server"}</h1>
          <Button
            variant="ghost"
            size="icon-sm"
            aria-label="Copy server name"
            onClick={() => copyToClipboard(mcpServer.server_name || mcpServer.alias, "mcp-server_name")}
          >
            {copiedStates["mcp-server_name"] ? <CheckIcon size={12} /> : <CopyIcon size={12} />}
          </Button>
          {mcpServer.alias && mcpServer.server_name && mcpServer.alias !== mcpServer.server_name && (
            <Badge variant="secondary" className="ml-2 font-mono">
              {mcpServer.alias}
            </Badge>
          )}
        </div>
        <div className="mt-1 flex items-center gap-1.5">
          <p className="font-mono text-xs text-muted-foreground">{mcpServer.server_id}</p>
          <Button
            variant="ghost"
            size="icon-sm"
            aria-label="Copy server id"
            onClick={() => copyToClipboard(mcpServer.server_id, "mcp-server-id")}
          >
            {copiedStates["mcp-server-id"] ? <CheckIcon size={10} /> : <CopyIcon size={10} />}
          </Button>
        </div>
        {mcpServer.description && <p className="mt-2 text-sm text-muted-foreground">{mcpServer.description}</p>}
      </div>

      <Tabs value={String(selectedTabIndex)} onValueChange={(v: unknown) => setSelectedTabIndex(Number(v))}>
        <TabsList className="mb-4">
          <TabsTrigger value="0" className="flex-none">
            Overview
          </TabsTrigger>
          <TabsTrigger value="1" className="flex-none">
            MCP Tools
          </TabsTrigger>
          {isProxyAdmin && (
            <TabsTrigger value="2" className="flex-none">
              Settings
            </TabsTrigger>
          )}
        </TabsList>

        {/* Overview Panel */}
        <TabsContent value="0">
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
            <Card className="p-4">
              <p className="text-xs font-medium tracking-wide text-muted-foreground uppercase">Transport</p>
              <div className="mt-3">
                {getTransportBadge(handleTransport(mcpServer.transport ?? undefined, mcpServer.spec_path ?? undefined))}
              </div>
            </Card>

            <Card className="p-4">
              <p className="text-xs font-medium tracking-wide text-muted-foreground uppercase">Authentication</p>
              <div className="mt-3">{getAuthBadge(handleAuth(mcpServer.auth_type ?? undefined))}</div>
            </Card>

            <Card className="p-4">
              <p className="text-xs font-medium tracking-wide text-muted-foreground uppercase">Host URL</p>
              <div className="mt-3 flex items-center gap-2">
                <p className="overflow-wrap-anywhere font-mono text-sm break-all">
                  {renderUrlWithToggle(mcpServer.url, showFullUrl)}
                </p>
                {/* Only proxy admins may reveal the raw URL — non-admins
                    receive a sanitized server object from the backend
                    with `url=null`, but hide the toggle anyway as
                    defense-in-depth in case the URL ever leaks back
                    into the response. */}
                {hasToken && isProxyAdmin && (
                  <Button
                    variant="ghost"
                    size="icon-sm"
                    aria-label={showFullUrl ? "Hide full URL" : "Show full URL"}
                    onClick={() => setShowFullUrl(!showFullUrl)}
                  >
                    {showFullUrl ? <EyeOff /> : <Eye />}
                  </Button>
                )}
              </div>
            </Card>
          </div>
          <Card className="mt-4 p-4">
            <p className="text-xs font-medium tracking-wide text-muted-foreground uppercase">Cost Configuration</p>
            <div className="mt-3">
              <MCPServerCostDisplay costConfig={mcpServer.mcp_info?.mcp_server_cost_info} />
            </div>
          </Card>
        </TabsContent>

        {/* Tool Panel */}
        <TabsContent value="1">
          <MCPToolsViewer
            serverId={mcpServer.server_id}
            accessToken={accessToken}
            auth_type={mcpServer.auth_type}
            oauth2_flow={mcpServer.oauth2_flow}
            delegate_auth_to_upstream={mcpServer.delegate_auth_to_upstream}
            dcr_bridge={mcpServer.dcr_bridge}
            tokenUrl={mcpServer.token_url}
            userRole={userRole}
            userID={userID}
            serverAlias={mcpServer.alias}
            extraHeaders={mcpServer.extra_headers}
          />
        </TabsContent>

        {/* Settings Panel */}
        <TabsContent value="2">
          <Card className="p-6">
            <div className="mb-4 flex items-center justify-between">
              <h2 className="text-lg font-medium">MCP Server Settings</h2>
              {editing ? null : (
                <Button variant="outline" onClick={() => setEditing(true)}>
                  Edit Settings
                </Button>
              )}
            </div>
            {editing ? (
              <MCPServerEdit
                mcpServer={mcpServer}
                accessToken={accessToken}
                userID={userID}
                onCancel={() => setEditing(false)}
                onSuccess={handleSuccess}
                availableAccessGroups={availableAccessGroups}
              />
            ) : (
              <div className="divide-y divide-border">
                <div className="grid grid-cols-3 gap-4 py-3">
                  <p className="text-sm font-medium text-muted-foreground">Server Name</p>
                  <div className="col-span-2 text-sm">
                    {mcpServer.server_name || <span className="text-muted-foreground">—</span>}
                  </div>
                </div>
                <div className="grid grid-cols-3 gap-4 py-3">
                  <p className="text-sm font-medium text-muted-foreground">Alias</p>
                  <div className="col-span-2 font-mono text-sm">
                    {mcpServer.alias || <span className="text-muted-foreground">—</span>}
                  </div>
                </div>
                <div className="grid grid-cols-3 gap-4 py-3">
                  <p className="text-sm font-medium text-muted-foreground">Description</p>
                  <div className="col-span-2 text-sm">
                    {mcpServer.description || <span className="text-muted-foreground">—</span>}
                  </div>
                </div>
                <div className="grid grid-cols-3 gap-4 py-3">
                  <p className="text-sm font-medium text-muted-foreground">URL</p>
                  <div className="col-span-2 flex items-center gap-2 font-mono text-sm break-all">
                    {renderUrlWithToggle(mcpServer.url, showFullUrl)}
                    {hasToken && (
                      <Button
                        variant="ghost"
                        size="icon-sm"
                        aria-label={showFullUrl ? "Hide full URL" : "Show full URL"}
                        onClick={() => setShowFullUrl(!showFullUrl)}
                      >
                        {showFullUrl ? <EyeOff /> : <Eye />}
                      </Button>
                    )}
                  </div>
                </div>
                <div className="grid grid-cols-3 gap-4 py-3">
                  <p className="text-sm font-medium text-muted-foreground">Transport</p>
                  <div className="col-span-2">
                    {getTransportBadge(handleTransport(mcpServer.transport, mcpServer.spec_path))}
                  </div>
                </div>
                <div className="grid grid-cols-3 gap-4 py-3">
                  <p className="text-sm font-medium text-muted-foreground">Authentication</p>
                  <div className="col-span-2">{getAuthBadge(handleAuth(mcpServer.auth_type))}</div>
                </div>
                <div className="grid grid-cols-3 gap-4 py-3">
                  <p className="text-sm font-medium text-muted-foreground">Extra Headers</p>
                  <div className="col-span-2 text-sm">
                    {mcpServer.extra_headers && mcpServer.extra_headers.length > 0 ? (
                      mcpServer.extra_headers.join(", ")
                    ) : (
                      <span className="text-muted-foreground">—</span>
                    )}
                  </div>
                </div>
                <div className="grid grid-cols-3 gap-4 py-3">
                  <p className="text-sm font-medium text-muted-foreground">Allow All Keys</p>
                  <div className="col-span-2">
                    {mcpServer.allow_all_keys ? (
                      <Badge variant="outline">
                        <span className="h-1.5 w-1.5 rounded-full bg-green-500" />
                        Enabled
                      </Badge>
                    ) : (
                      <Badge variant="outline">Disabled</Badge>
                    )}
                  </div>
                </div>
                <div className="grid grid-cols-3 gap-4 py-3">
                  <p className="text-sm font-medium text-muted-foreground">Network Access</p>
                  <div className="col-span-2">
                    {mcpServer.available_on_public_internet ? (
                      <Badge variant="outline">
                        <span className="h-1.5 w-1.5 rounded-full bg-green-500" />
                        Public
                      </Badge>
                    ) : (
                      <Badge variant="outline">
                        <span className="h-1.5 w-1.5 rounded-full bg-orange-500" />
                        Internal only
                      </Badge>
                    )}
                  </div>
                </div>
                {handleAuth(mcpServer.auth_type) === "oauth2" && (
                  <div className="grid grid-cols-3 gap-4 py-3">
                    <p className="text-sm font-medium text-muted-foreground">Delegate Auth to Upstream</p>
                    <div className="col-span-2">
                      {mcpServer.delegate_auth_to_upstream ? (
                        <Badge variant="outline">
                          <span className="h-1.5 w-1.5 rounded-full bg-green-500" />
                          Enabled (PKCE passthrough)
                        </Badge>
                      ) : (
                        <Badge variant="outline">Disabled</Badge>
                      )}
                    </div>
                  </div>
                )}
                {handleAuth(mcpServer.auth_type) !== "oauth2" &&
                  Array.isArray(mcpServer.extra_headers) &&
                  mcpServer.extra_headers.some((h) => typeof h === "string" && h.toLowerCase() === "authorization") && (
                    <div className="grid grid-cols-3 gap-4 py-3">
                      <p className="text-sm font-medium text-muted-foreground">OAuth Pass-through</p>
                      <div className="col-span-2">
                        {mcpServer.oauth_passthrough ? (
                          <Badge variant="outline">
                            <span className="h-1.5 w-1.5 rounded-full bg-green-500" />
                            Enabled
                          </Badge>
                        ) : (
                          <Badge variant="outline">Disabled</Badge>
                        )}
                      </div>
                    </div>
                  )}
                <div className="grid grid-cols-3 gap-4 py-3">
                  <p className="text-sm font-medium text-muted-foreground">Access Groups</p>
                  <div className="col-span-2">
                    {mcpServer.mcp_access_groups && mcpServer.mcp_access_groups.length > 0 ? (
                      <div className="flex flex-wrap gap-1.5">
                        {mcpServer.mcp_access_groups.map((group: any, index: number) => (
                          <Badge key={index} variant="secondary">
                            {typeof group === "string" ? group : group?.name ?? ""}
                          </Badge>
                        ))}
                      </div>
                    ) : (
                      <span className="text-sm text-muted-foreground">—</span>
                    )}
                  </div>
                </div>
                <div className="grid grid-cols-3 gap-4 py-3">
                  <p className="text-sm font-medium text-muted-foreground">Allowed Tools</p>
                  <div className="col-span-2">
                    {mcpServer.allowed_tools && mcpServer.allowed_tools.length > 0 ? (
                      <div className="flex flex-wrap gap-1.5">
                        {mcpServer.allowed_tools.map((tool: string, index: number) => (
                          <Badge key={index} variant="secondary" className="font-mono">
                            {tool}
                          </Badge>
                        ))}
                      </div>
                    ) : (
                      <Badge variant="outline">All tools enabled</Badge>
                    )}
                  </div>
                </div>
                <div className="grid grid-cols-3 gap-4 py-3">
                  <p className="text-sm font-medium text-muted-foreground">Cost</p>
                  <div className="col-span-2">
                    <MCPServerCostDisplay costConfig={mcpServer.mcp_info?.mcp_server_cost_info} />
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
