import { useOrganizations } from "@/app/(dashboard)/hooks/organizations/useOrganizations";
import useCan from "@/app/(dashboard)/hooks/useCan";
import { useUrlTab } from "@/hooks/useUrlTab";
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

type LogsTabId = "request logs" | "audit logs" | "deleted keys" | "deleted teams";

const LOGS_TAB_SLUGS = ["request-logs", "audit-logs", "deleted-keys", "deleted-teams"] as const;
type LogsTabSlug = (typeof LOGS_TAB_SLUGS)[number];

interface LogsTab {
  id: LogsTabId;
  slug: LogsTabSlug;
  label: string;
}

const REQUEST_LOGS_TAB: LogsTab = { id: "request logs", slug: "request-logs", label: "Request Logs" };
const AUDIT_LOGS_TAB: LogsTab = { id: "audit logs", slug: "audit-logs", label: "Audit Logs" };
const DELETED_KEYS_TAB: LogsTab = { id: "deleted keys", slug: "deleted-keys", label: "Deleted Keys" };
const DELETED_TEAMS_TAB: LogsTab = { id: "deleted teams", slug: "deleted-teams", label: "Deleted Teams" };

const tabContentClassName = (tabId: LogsTabId): string =>
  tabId === REQUEST_LOGS_TAB.id ? "flex min-h-0 flex-1 flex-col" : "min-h-0 flex-1 overflow-y-auto";

export default function SpendLogsTable({ accessToken, token, userRole, userID, premiumUser }: SpendLogsTableProps) {
  const canViewAuditLogs = useCan("viewAuditLogs");
  const canViewDeletedTeams = useCan("viewDeletedTeams");
  const { isLoading: isOrgMembershipLoading } = useOrganizations();

  const tabs: LogsTab[] = [
    REQUEST_LOGS_TAB,
    ...(canViewAuditLogs ? [AUDIT_LOGS_TAB] : []),
    DELETED_KEYS_TAB,
    ...(canViewDeletedTeams ? [DELETED_TEAMS_TAB] : []),
  ];
  const isRoleResolved = Boolean(userRole) && !isOrgMembershipLoading;
  const visibleSlugs: readonly LogsTabSlug[] = isRoleResolved ? tabs.map((tab) => tab.slug) : LOGS_TAB_SLUGS;
  const [activeSlug, setActiveSlug] = useUrlTab(visibleSlugs, REQUEST_LOGS_TAB.slug);

  if (!accessToken || !token || !userRole || !userID) {
    return (
      <div role="status" aria-busy="true" aria-label="Loading" className="flex h-64 items-center justify-center">
        <UiLoadingSpinner className="size-8 text-primary" />
      </div>
    );
  }

  const renderPanel = (tab: LogsTab) => {
    switch (tab.id) {
      case "request logs":
        return (
          <RequestLogsPanel
            accessToken={accessToken}
            token={token}
            userRole={userRole}
            userID={userID}
            isActive={activeSlug === tab.slug}
          />
        );
      case "audit logs":
        return (
          <AuditLogsPanel
            userID={userID}
            userRole={userRole}
            token={token}
            accessToken={accessToken}
            isActive={activeSlug === tab.slug}
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
    <div className="flex h-full w-full flex-col p-6">
      <Tabs
        value={activeSlug}
        onValueChange={(value) => setActiveSlug(value as LogsTabSlug)}
        className="min-h-0 flex-1"
      >
        <TabsList variant="line">
          {tabs.map((tab) => (
            <TabsTrigger key={tab.slug} value={tab.slug} className="flex-none">
              {tab.label}
            </TabsTrigger>
          ))}
        </TabsList>
        {tabs.map((tab) => (
          <TabsContent key={tab.slug} value={tab.slug} keepMounted className={tabContentClassName(tab.id)}>
            {renderPanel(tab)}
          </TabsContent>
        ))}
      </Tabs>
    </div>
  );
}
