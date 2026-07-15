"use client";

import { Button } from "@/components/ui/button";
import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbList,
  BreadcrumbPage,
  BreadcrumbSeparator,
} from "@/components/ui/breadcrumb";
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
  const { title } = getBreadcrumb(page);
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
      <Breadcrumb className="min-w-0">
        <BreadcrumbList className="flex-nowrap">
          <BreadcrumbItem className="flex-none">
            <ViewSwitcher />
          </BreadcrumbItem>
          <BreadcrumbSeparator />
          <BreadcrumbItem className="min-w-0">
            <BreadcrumbPage className="truncate">{title}</BreadcrumbPage>
          </BreadcrumbItem>
        </BreadcrumbList>
      </Breadcrumb>

      <div className="flex flex-none items-center gap-1">
        {showWorkerSwitch && (
          <>
            <WorkerDropdown onWorkerSwitch={handleWorkerSwitch} />
            <Separator orientation="vertical" className="mx-1.5 h-5" />
          </>
        )}
        <Button
          variant="ghost"
          size="sm"
          nativeButton={false}
          render={<a href="https://docs.litellm.ai/docs/" target="_blank" rel="noopener noreferrer" />}
          className="text-muted-foreground"
        >
          Docs
        </Button>
        <BlogDropdown />
        {!hideCommunityLinks && <CommunityEngagementButtons />}
        <Separator orientation="vertical" className="mx-1.5 h-5" />
        <NotificationsBell />
      </div>
    </header>
  );
}

export default DashboardHeader;
