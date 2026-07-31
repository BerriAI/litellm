import { useState } from "react";
import DeletedKeysPage from "../DeletedKeysPage/DeletedKeysPage";
import DeletedTeamsPage from "../DeletedTeamsPage/DeletedTeamsPage";
import AuditLogsPanel from "./AuditLogsPanel";
import RequestLogsPanel from "./RequestLogsPanel";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { UiLoadingSpinner } from "@/components/ui/ui-loading-spinner";

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
      <div role="status" aria-busy="true" aria-label="Loading" className="flex h-64 items-center justify-center">
        <UiLoadingSpinner className="size-8 text-primary" />
      </div>
    );
  }

  return (
    <div className="box-border w-full overflow-x-hidden p-6">
      <Tabs value={activeTab} onValueChange={(value) => setActiveTab(value as string)}>
        <TabsList variant="line">
          <TabsTrigger value="request logs" className="flex-none">
            Request Logs
          </TabsTrigger>
          <TabsTrigger value="audit logs" className="flex-none">
            Audit Logs
          </TabsTrigger>
          <TabsTrigger value="deleted keys" className="flex-none">
            Deleted Keys
          </TabsTrigger>
          <TabsTrigger value="deleted teams" className="flex-none">
            Deleted Teams
          </TabsTrigger>
        </TabsList>
        <TabsContent value="request logs" keepMounted>
          <RequestLogsPanel
            accessToken={accessToken}
            token={token}
            userRole={userRole}
            userID={userID}
            isActive={activeTab === "request logs"}
          />
        </TabsContent>
        <TabsContent value="audit logs" keepMounted>
          <AuditLogsPanel
            userID={userID}
            userRole={userRole}
            token={token}
            accessToken={accessToken}
            isActive={activeTab === "audit logs"}
            premiumUser={premiumUser}
          />
        </TabsContent>
        <TabsContent value="deleted keys" keepMounted>
          <DeletedKeysPage />
        </TabsContent>
        <TabsContent value="deleted teams" keepMounted>
          <DeletedTeamsPage />
        </TabsContent>
      </Tabs>
    </div>
  );
}
