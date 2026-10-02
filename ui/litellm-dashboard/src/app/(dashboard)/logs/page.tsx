"use client";

import { useState } from "react";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import useCan from "@/app/(dashboard)/hooks/useCan";
import DeletedKeysPage from "@/components/DeletedKeysPage/DeletedKeysPage";
import DeletedTeamsPage from "@/components/DeletedTeamsPage/DeletedTeamsPage";
import AuditLogsPanel from "@/components/view_logs/AuditLogsPanel";
import RequestLogsPanel from "@/components/view_logs/RequestLogsPanel";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { UiLoadingSpinner } from "@/components/ui/ui-loading-spinner";

type LogsTab = "request logs" | "audit logs" | "deleted keys" | "deleted teams";

const triggerClassName = "flex-none gap-1.5 px-0 text-[13px] font-normal";
const scrollablePanelClassName = "min-h-0 flex-1 overflow-y-auto";

export default function LogsPage() {
  const { accessToken, userRole, userId, token, premiumUser } = useAuthorized();
  const [activeTab, setActiveTab] = useState<LogsTab>("request logs");
  const canViewAuditLogs = useCan("viewAuditLogs");
  const canViewDeletedTeams = useCan("viewDeletedTeams");

  const credentialsPending = !accessToken || !token;
  const identityPending = !userRole || !userId;

  if (credentialsPending || identityPending) {
    return (
      <div role="status" aria-busy="true" aria-label="Loading" className="flex h-64 items-center justify-center">
        <UiLoadingSpinner className="size-8 text-primary" />
      </div>
    );
  }

  return (
    <div className="flex h-full w-full flex-col px-4 pb-4">
      <Tabs value={activeTab} onValueChange={(value: LogsTab) => setActiveTab(value)} className="min-h-0 flex-1 gap-0">
        <TabsList variant="line" className="-mx-4 h-10 w-auto justify-start gap-4 border-b border-border px-4">
          <TabsTrigger value="request logs" className={triggerClassName}>
            Request Logs
          </TabsTrigger>
          {canViewAuditLogs && (
            <TabsTrigger value="audit logs" className={triggerClassName}>
              Audit Logs
            </TabsTrigger>
          )}
          <TabsTrigger value="deleted keys" className={triggerClassName}>
            Deleted Keys
          </TabsTrigger>
          {canViewDeletedTeams && (
            <TabsTrigger value="deleted teams" className={triggerClassName}>
              Deleted Teams
            </TabsTrigger>
          )}
        </TabsList>

        <TabsContent value="request logs" keepMounted className="flex min-h-0 flex-1 flex-col">
          <RequestLogsPanel
            accessToken={accessToken}
            token={token}
            userRole={userRole}
            userID={userId}
            isActive={activeTab === "request logs"}
          />
        </TabsContent>
        {canViewAuditLogs && (
          <TabsContent value="audit logs" keepMounted className={scrollablePanelClassName}>
            <AuditLogsPanel
              accessToken={accessToken}
              token={token}
              userRole={userRole}
              userID={userId}
              premiumUser={premiumUser ?? false}
              isActive={activeTab === "audit logs"}
            />
          </TabsContent>
        )}
        <TabsContent value="deleted keys" keepMounted className={scrollablePanelClassName}>
          <DeletedKeysPage />
        </TabsContent>
        {canViewDeletedTeams && (
          <TabsContent value="deleted teams" keepMounted className={scrollablePanelClassName}>
            <DeletedTeamsPage />
          </TabsContent>
        )}
      </Tabs>
    </div>
  );
}
