"use client";

import type { ReactNode } from "react";
import ThemeToggle from "@/components/ThemeToggle/ThemeToggle";
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
  deploymentHealthLink,
}: {
  accessToken: string | null;
  navigationTrigger?: ReactNode;
  deploymentHealth?: ReactNode;
  deploymentHealthLink?: ReactNode;
}) {
  const term = useTerminology();
  const { title } = getBreadcrumb(usePathname());
  const { isControlPlane, selectedWorker } = useWorker();
  const showWorkerSelector = isControlPlane && selectedWorker !== null;
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
          {showWorkerSelector && (
            <>
              <div className="@min-[840px]:hidden">
                <WorkerDropdown compact onWorkerSwitch={handleWorkerSwitch} />
              </div>
              <div className="hidden @min-[840px]:block">
                <WorkerDropdown onWorkerSwitch={handleWorkerSwitch} />
              </div>
            </>
          )}
          <DashboardCommunityLinks
            className={showWorkerSelector ? "hidden @min-[840px]:flex" : "hidden @min-[400px]:flex"}
          />
          <ThemeToggle />
          {deploymentHealth}
          <NotificationsBell accessToken={accessToken} deploymentHealthLink={deploymentHealthLink} />
        </div>
      </div>
    </header>
  );
}
