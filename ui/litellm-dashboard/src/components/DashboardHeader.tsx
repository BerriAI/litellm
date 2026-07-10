"use client";

import { ChevronRight } from "lucide-react";
import { Separator } from "@/components/ui/separator";
import { getBreadcrumb } from "@/components/leftnav";
import { BlogDropdown } from "@/components/Navbar/BlogDropdown/BlogDropdown";
import { CommunityEngagementButtons } from "@/components/Navbar/CommunityEngagementButtons/CommunityEngagementButtons";
import { NotificationsBell } from "@/components/Navbar/NotificationsBell/NotificationsBell";
import ViewSwitcher from "@/components/Navbar/ViewSwitcher";
import WorkerDropdown from "@/components/Navbar/WorkerDropdown/WorkerDropdown";
import { useWorker } from "@/hooks/useWorker";
import { useDisableShowPrompts } from "@/app/(dashboard)/hooks/useDisableShowPrompts";
import { clearTokenCookies } from "@/utils/cookieUtils";
import { clearStoredReturnUrl } from "@/utils/returnUrlUtils";

interface DashboardHeaderProps {
  page: string;
}

// Top bar for the dashboard shell. Sits only over the content column (the brand
// lives in the sidebar header); mirrors the design's breadcrumb-left / tools-right layout.
export function DashboardHeader({ page }: DashboardHeaderProps) {
  const { section, title } = getBreadcrumb(page);
  const { isControlPlane, selectedWorker } = useWorker();
  const showWorkerSwitch = isControlPlane && selectedWorker !== null;
  const hideCommunityLinks = useDisableShowPrompts();

  const handleWorkerSwitch = (workerId: string) => {
    clearTokenCookies();
    clearStoredReturnUrl();
    localStorage.removeItem("litellm_selected_worker_id");
    localStorage.removeItem("litellm_worker_url");
    window.location.href = `/ui/login?worker=${encodeURIComponent(workerId)}`;
  };

  return (
    <header className="flex h-14 flex-none items-center justify-between gap-4 border-b border-border bg-background px-4">
      <nav aria-label="Breadcrumb" className="flex min-w-0 items-center gap-1.5 text-sm">
        {section && (
          <>
            <span className="whitespace-nowrap text-muted-foreground">{section}</span>
            <ChevronRight className="size-4 flex-none text-muted-foreground" aria-hidden />
          </>
        )}
        <span className="truncate font-medium text-foreground">{title}</span>
      </nav>

      <div className="flex flex-none items-center gap-1">
        {showWorkerSwitch && (
          <>
            <WorkerDropdown onWorkerSwitch={handleWorkerSwitch} />
            <Separator orientation="vertical" className="mx-1.5 h-5" />
          </>
        )}
        <a
          href="https://docs.litellm.ai/docs/"
          target="_blank"
          rel="noopener noreferrer"
          className="inline-flex h-8 items-center rounded-md px-2.5 text-sm font-medium text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
        >
          Docs
        </a>
        <BlogDropdown />
        {!hideCommunityLinks && <CommunityEngagementButtons />}
        <Separator orientation="vertical" className="mx-1.5 h-5" />
        <NotificationsBell />
        <Separator orientation="vertical" className="mx-1.5 h-5" />
        <ViewSwitcher />
      </div>
    </header>
  );
}

export default DashboardHeader;
