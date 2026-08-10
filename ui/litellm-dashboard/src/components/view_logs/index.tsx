import { useState } from "react";
import { Tab, TabGroup, TabList, TabPanel, TabPanels } from "@tremor/react";
import useCan from "@/app/(dashboard)/hooks/useCan";
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

type LogsTabId = "request logs" | "audit logs" | "deleted keys" | "deleted teams";

interface LogsTab {
  id: LogsTabId;
  label: string;
}

const REQUEST_LOGS_TAB: LogsTab = { id: "request logs", label: "Request Logs" };
const AUDIT_LOGS_TAB: LogsTab = { id: "audit logs", label: "Audit Logs" };
const DELETED_KEYS_TAB: LogsTab = { id: "deleted keys", label: "Deleted Keys" };
const DELETED_TEAMS_TAB: LogsTab = { id: "deleted teams", label: "Deleted Teams" };

export default function SpendLogsTable({ accessToken, token, userRole, userID, premiumUser }: SpendLogsTableProps) {
  const [activeTab, setActiveTab] = useState<LogsTabId>(REQUEST_LOGS_TAB.id);
  const canViewAuditLogs = useCan("viewAuditLogs");
  const canViewDeletedTeams = useCan("viewDeletedTeams");

  if (!accessToken || !token || !userRole || !userID) {
    return (
      <div className="flex items-center justify-center h-64">
        <AntDLoadingSpinner size="large" />
      </div>
    );
  }

  const tabs: LogsTab[] = [
    REQUEST_LOGS_TAB,
    ...(canViewAuditLogs ? [AUDIT_LOGS_TAB] : []),
    DELETED_KEYS_TAB,
    ...(canViewDeletedTeams ? [DELETED_TEAMS_TAB] : []),
  ];

  const renderPanel = (tabId: LogsTabId) => {
    switch (tabId) {
      case "request logs":
        return (
          <RequestLogsPanel
            accessToken={accessToken}
            token={token}
            userRole={userRole}
            userID={userID}
            isActive={activeTab === "request logs"}
          />
        );
      case "audit logs":
        return (
          <AuditLogsPanel
            userID={userID}
            userRole={userRole}
            token={token}
            accessToken={accessToken}
            isActive={activeTab === "audit logs"}
            premiumUser={premiumUser}
          />
        );
      case "deleted keys":
        return <DeletedKeysPage />;
      case "deleted teams":
        return <DeletedTeamsPage />;
    }
  };

  return (
    <div className="w-full p-6 overflow-x-hidden box-border">
      <TabGroup defaultIndex={0} onIndexChange={(index) => setActiveTab(tabs[index].id)}>
        <TabList>
          {tabs.map((tab) => (
            <Tab key={tab.id}>{tab.label}</Tab>
          ))}
        </TabList>
        <TabPanels>
          {tabs.map((tab) => (
            <TabPanel key={tab.id}>{renderPanel(tab.id)}</TabPanel>
          ))}
        </TabPanels>
      </TabGroup>
    </div>
  );
}
