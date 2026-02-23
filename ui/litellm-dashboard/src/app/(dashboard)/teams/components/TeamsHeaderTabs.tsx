import { Icon, Tab, TabGroup, TabList, TabPanels, Text } from "@tremor/react";
import { isAdminRole } from "@/utils/roles";
import { RefreshIcon } from "@heroicons/react/outline";
import React from "react";

type TeamsHeaderTabsProps = {
  lastRefreshed: string;
  onRefresh: () => void;
  userRole: string | null;
  children: React.ReactNode;
};

const TeamsHeaderTabs = ({ lastRefreshed, onRefresh, userRole, children }: TeamsHeaderTabsProps) => {
  return (
    <TabGroup className="gap-2 h-[75vh] w-full">
      <TabList className="flex justify-between mt-2 w-full items-center">
        <div className="flex">
          <Tab>Your Teams</Tab>
          <Tab>Available Teams</Tab>
          {isAdminRole(userRole || "") && <Tab>Default Team Settings</Tab>}
        </div>
        <div className="flex items-center space-x-2">
          {lastRefreshed && <Text>Last Refreshed: {lastRefreshed}</Text>}
          <Icon
            icon={RefreshIcon} // Modify as necessary for correct icon name
            variant="shadow"
            size="xs"
            className="self-center"
            onClick={onRefresh}
          />
        </div>
      </TabList>
      <TabPanels>{children}</TabPanels>
    </TabGroup>
  );
};

export default TeamsHeaderTabs;
