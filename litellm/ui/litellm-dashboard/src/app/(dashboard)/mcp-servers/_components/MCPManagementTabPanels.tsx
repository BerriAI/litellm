import MCPSemanticFilterSettings from "@/components/Settings/AdminSettings/MCPSemanticFilterSettings/MCPSemanticFilterSettings";
import MCPToolSearchSettings from "@/components/Settings/AdminSettings/MCPToolSearchSettings/MCPToolSearchSettings";
import { TabsContent } from "@/components/ui/tabs";
import { isAdminRole, isProxyAdminRole, isProxyAdminTierRole } from "@/utils/roles";
import { MCPGatewaySessionsTab } from "./MCPGatewaySessionsTab";
import MCPNetworkSettings from "./MCPNetworkSettings";
import { MCPSubmissionsTab } from "./MCPSubmissionsTab";
import { MCPToolsetsTab } from "./MCPToolsetsTab";
import MCPConnect from "./mcp_connect";

interface MCPManagementTabPanelsProps {
  accessToken: string;
  userRole: string;
  isViewOnly: boolean;
}

export function MCPManagementTabPanels({ accessToken, userRole, isViewOnly }: MCPManagementTabPanelsProps) {
  return (
    <>
      <TabsContent value="toolsets" keepMounted>
        <MCPToolsetsTab accessToken={accessToken} userRole={userRole} />
      </TabsContent>
      <TabsContent value="connect" keepMounted>
        <MCPConnect />
      </TabsContent>
      {isAdminRole(userRole) && (
        <>
          <TabsContent value="semantic-filter" keepMounted>
            <MCPSemanticFilterSettings accessToken={accessToken} />
          </TabsContent>
          <TabsContent value="tool-search" keepMounted>
            <MCPToolSearchSettings accessToken={accessToken} />
          </TabsContent>
          <TabsContent value="network-settings" keepMounted>
            <MCPNetworkSettings accessToken={accessToken} />
          </TabsContent>
          <TabsContent value="submitted" keepMounted>
            <MCPSubmissionsTab accessToken={accessToken} />
          </TabsContent>
        </>
      )}
      {isProxyAdminTierRole(userRole) && (
        <TabsContent value="connections">
          <MCPGatewaySessionsTab accessToken={accessToken} canTerminate={isProxyAdminRole(userRole) && !isViewOnly} />
        </TabsContent>
      )}
    </>
  );
}
