"use client";

import type { ReactNode } from "react";
import { usePathname } from "next/navigation";
import { useTerminology } from "@/contexts/TerminologyContext";
import { getBreadcrumb } from "@/components/leftnav";
import { CommandPaletteTrigger } from "@/components/CommandPalette/CommandPaletteTrigger";
import { DashboardCommunityLinks } from "@/components/Navbar/DashboardResources";
import { NotificationsBell } from "@/components/Navbar/NotificationsBell/NotificationsBell";
import WorkerDropdown from "@/components/Navbar/WorkerDropdown/WorkerDropdown";
import { useWorker } from "@/hooks/useWorker";
import { clearTokenCookies } from "@/utils/cookieUtils";
import { clearStoredReturnUrl, getLoginUrl } from "@/utils/returnUrlUtils";

export function DashboardHeader({
  accessToken,
  navigationTrigger,
  deploymentHealth,
}: {
  accessToken: string | null;
  navigationTrigger?: ReactNode;
  deploymentHealth?: ReactNode;
}) {
  const term = useTerminology();
  const { title } = getBreadcrumb(usePathname());
  const { isControlPlane, selectedWorker } = useWorker();
  const handleWorkerSwitch = (workerId: string) => {
    clearTokenCookies();
    clearStoredReturnUrl();
    localStorage.removeItem("litellm_selected_worker_id");
    localStorage.removeItem("litellm_worker_url");
    window.location.href = `${getLoginUrl()}?worker=${encodeURIComponent(workerId)}`;
  };
  return (
    <header className="@container h-14 shrink-0 border-b border-border bg-background px-3 sm:px-5">
      <div className="flex h-full items-center gap-3">
        <div className="flex min-w-0 flex-1 items-center gap-2">
          {navigationTrigger}
          <h1 className="truncate text-sm font-medium">{term(title)}</h1>
        </div>
        <CommandPaletteTrigger className="hidden w-56 @min-[640px]:flex @min-[1000px]:w-80" />
        <div className="flex shrink-0 items-center justify-end gap-1">
          <CommandPaletteTrigger mobile className="size-9 @min-[640px]:hidden" />
          {isControlPlane && selectedWorker !== null && <WorkerDropdown onWorkerSwitch={handleWorkerSwitch} />}
          <DashboardCommunityLinks className="hidden @min-[840px]:flex" />
          {deploymentHealth}
          <NotificationsBell accessToken={accessToken} />
        </div>
      </div>
    </header>
  );
}
