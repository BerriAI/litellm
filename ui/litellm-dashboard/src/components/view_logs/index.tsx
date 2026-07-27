import { useState } from "react";
import { Tab, TabGroup, TabList, TabPanel, TabPanels } from "@tremor/react";
import DeletedKeysPage from "../DeletedKeysPage/DeletedKeysPage";
import DeletedTeamsPage from "../DeletedTeamsPage/DeletedTeamsPage";
import AuditLogsPanel from "./AuditLogsPanel";
import RequestLogsPanel from "./RequestLogsPanel";
import { AntDLoadingSpinner } from "../ui/AntDLoadingSpinner";

interface SpendLogsTableProps {
  accessToken: string | null;
  token: string | null;
  userRole: string | null;
  userID: string | null;
  premiumUser: boolean;
}

export default function SpendLogsTable({ accessToken, token, userRole, userID, premiumUser }: SpendLogsTableProps) {
  const [activeTab, setActiveTab] = useState("request logs");

  if (!accessToken || !token || !userRole || !userID) {
    return (
      <div className="flex items-center justify-center h-64">
        <AntDLoadingSpinner size="large" />
      </div>
    );
  }

  return (
    <div className="w-full p-6 overflow-x-hidden box-border">
      <TabGroup defaultIndex={0} onIndexChange={(index) => setActiveTab(index === 0 ? "request logs" : "audit logs")}>
        <TabList>
          <Tab>Request Logs</Tab>
          <Tab>Audit Logs</Tab>
          <Tab>Deleted Keys</Tab>
          <Tab>Deleted Teams</Tab>
        </TabList>
        <TabPanels>
          <TabPanel>
            <RequestLogsPanel
              accessToken={accessToken}
              token={token}
              userRole={userRole}
              userID={userID}
              isActive={activeTab === "request logs"}
            />
          </TabPanel>
          <TabPanel>
            <AuditLogsPanel
              userID={userID}
              userRole={userRole}
              token={token}
              accessToken={accessToken}
              isActive={activeTab === "audit logs"}
              premiumUser={premiumUser}
            />
          </TabPanel>
          <TabPanel>
            <DeletedKeysPage />
          </TabPanel>
          <TabPanel>
            <DeletedTeamsPage />
          </TabPanel>
        </TabPanels>
      </TabGroup>
    </div>
  );
}
