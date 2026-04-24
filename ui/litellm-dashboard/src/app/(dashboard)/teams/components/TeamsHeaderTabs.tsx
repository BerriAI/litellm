import {
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
} from "@/components/ui/tabs";
import { isAdminRole } from "@/utils/roles";
import { RefreshCw } from "lucide-react";
import React from "react";

type TeamsHeaderTabsProps = {
  lastRefreshed: string;
  onRefresh: () => void;
  userRole: string | null;
  yourTeamsPanel: React.ReactNode;
  availableTeamsPanel: React.ReactNode;
  defaultTeamSettingsPanel: React.ReactNode;
};

const TeamsHeaderTabs = ({
  lastRefreshed,
  onRefresh,
  userRole,
  yourTeamsPanel,
  availableTeamsPanel,
  defaultTeamSettingsPanel,
}: TeamsHeaderTabsProps) => {
  const showDefaults = isAdminRole(userRole || "");
  return (
    <Tabs defaultValue="your-teams" className="gap-2 h-[75vh] w-full">
      <TabsList className="flex justify-between mt-2 w-full items-center bg-transparent p-0 h-auto">
        <div className="flex">
          <TabsTrigger value="your-teams">Your Teams</TabsTrigger>
          <TabsTrigger value="available-teams">Available Teams</TabsTrigger>
          {showDefaults && (
            <TabsTrigger value="default-settings">
              Default Team Settings
            </TabsTrigger>
          )}
        </div>
        <div className="flex items-center space-x-2">
          {lastRefreshed && (
            <span className="text-muted-foreground text-sm">
              Last Refreshed: {lastRefreshed}
            </span>
          )}
          <button
            type="button"
            onClick={onRefresh}
            className="p-1.5 rounded-md border border-border shadow-sm hover:bg-muted transition-colors self-center"
            aria-label="Refresh"
          >
            <RefreshCw className="h-3.5 w-3.5" />
          </button>
        </div>
      </TabsList>
      <TabsContent value="your-teams">{yourTeamsPanel}</TabsContent>
      <TabsContent value="available-teams">{availableTeamsPanel}</TabsContent>
      {showDefaults && (
        <TabsContent value="default-settings">
          {defaultTeamSettingsPanel}
        </TabsContent>
      )}
    </Tabs>
  );
};

export default TeamsHeaderTabs;
