"use client";

import type { ReactNode } from "react";
import { AntDLoadingSpinner } from "@/components/ui/AntDLoadingSpinner";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { logsRoutes } from "@/app/(dashboard)/logs/tabRoutes";
import { useTabRouting } from "@/app/(dashboard)/hooks/useTabRouting";
import { TabRouteBar } from "@/app/(dashboard)/components/TabRouteBar";

const BASE_TAB_KEY = "request-logs";

const TABS = [
  { key: BASE_TAB_KEY, label: "Request Logs" },
  { key: "audit", label: "Audit Logs" },
  { key: "deleted-keys", label: "Deleted Keys" },
  { key: "deleted-teams", label: "Deleted Teams" },
] as const;

export default function LogsLayout({ children }: { children: ReactNode }) {
  const { accessToken, token, userRole, userId } = useAuthorized();
  const { activeKey } = useTabRouting({
    routes: logsRoutes,
    baseTabKey: BASE_TAB_KEY,
    visibleKeys: logsRoutes.slugs,
  });

  const hasCredentials = Boolean(accessToken && token);
  const hasIdentity = Boolean(userRole && userId);

  if (!hasCredentials || !hasIdentity) {
    return (
      <div className="flex items-center justify-center h-64">
        <AntDLoadingSpinner size="large" />
      </div>
    );
  }

  return (
    <div className="w-full p-6 overflow-x-hidden box-border">
      <TabRouteBar routes={logsRoutes} baseTabKey={BASE_TAB_KEY} activeKey={activeKey} tabs={TABS} />
      <div className="mt-4">{children}</div>
    </div>
  );
}
