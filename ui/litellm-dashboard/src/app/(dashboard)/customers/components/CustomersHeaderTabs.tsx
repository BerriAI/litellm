import React from "react";
import { Tab, TabGroup, TabList, TabPanels, Text } from "@tremor/react";
import { RefreshCw } from "lucide-react";

interface CustomersHeaderTabsProps {
  lastRefreshed: string;
  onRefresh: () => void;
  userRole: string | null;
  children: React.ReactNode;
}

const CustomersHeaderTabs: React.FC<CustomersHeaderTabsProps> = ({
  lastRefreshed,
  onRefresh,
  userRole,
  children,
}) => {
  return (
    <div>
      <div className="flex items-center justify-between mb-2">
        <TabGroup>
          <TabList className="mt-2">
            <Tab>Your Customers</Tab>
            {(userRole === "Admin" || userRole === "Org Admin") && (
              <Tab>Customer Settings</Tab>
            )}
          </TabList>
          <TabPanels>{children}</TabPanels>
        </TabGroup>
        <div className="flex items-center gap-2 text-xs text-gray-500">
          <Text>Last Refreshed: {lastRefreshed}</Text>
          <button
            onClick={onRefresh}
            className="p-1 text-gray-400 hover:text-gray-600 transition-colors"
            aria-label="Refresh"
          >
            <RefreshCw className="w-4 h-4" />
          </button>
        </div>
      </div>
    </div>
  );
};

export default CustomersHeaderTabs;
