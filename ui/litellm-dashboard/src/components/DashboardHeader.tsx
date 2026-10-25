"use client";

import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbList,
  BreadcrumbPage,
  BreadcrumbSeparator,
} from "@/components/ui/breadcrumb";
import { ToolbarSeparator } from "@/components/shared/ToolbarSeparator";
import { getBreadcrumb } from "@/components/leftnav";
import { BlogDropdown } from "@/components/Navbar/BlogDropdown/BlogDropdown";
import { DocsLink } from "@/components/Navbar/DocsLink/DocsLink";
import { CommunityEngagementButtons } from "@/components/Navbar/CommunityEngagementButtons/CommunityEngagementButtons";
import { NotificationsBell } from "@/components/Navbar/NotificationsBell/NotificationsBell";
import ViewSwitcher from "@/components/Navbar/ViewSwitcher";
import LiteAdmin from "@/components/liteadmin/LiteAdmin";
import ThemeToggle from "@/components/ThemeToggle/ThemeToggle";
import WorkerDropdown from "@/components/Navbar/WorkerDropdown/WorkerDropdown";
import { useWorker } from "@/hooks/useWorker";
import { useDisableShowPrompts } from "@/app/(dashboard)/hooks/useDisableShowPrompts";
import { clearTokenCookies } from "@/utils/cookieUtils";
import { clearStoredReturnUrl, getLoginUrl } from "@/utils/returnUrlUtils";
import { usePathname } from "next/navigation";
import { useState, type ReactNode } from "react";
import { useMediaQuery } from "usehooks-ts";
import { Ellipsis } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Popover, PopoverContent, PopoverTitle, PopoverTrigger } from "@/components/ui/popover";
import { CommandPaletteTrigger } from "@/components/CommandPalette/CommandPaletteTrigger";
import LanguageSelector from "@/components/LanguageSelector";
import { useLanguage } from "@/contexts/LanguageContext";

// Top bar for the dashboard shell. Sits only over the content column (the brand
// lives in the sidebar header); mirrors the design's breadcrumb-left / tools-right layout.
export function DashboardHeader({
  accessToken,
  navigationTrigger,
  deploymentHealth,
}: {
  accessToken: string | null;
  navigationTrigger?: ReactNode;
  deploymentHealth?: ReactNode;
}) {
  const { t } = useLanguage();
  const { title } = getBreadcrumb(usePathname());
  // 侧边栏占用宽度后，平板也使用紧凑菜单，避免语言选择器被挤出窗口
  const isDesktop = useMediaQuery("(min-width: 1280px)", { initializeWithValue: false });
  const [mobileToolsOpen, setMobileToolsOpen] = useState(false);
  if (isDesktop && mobileToolsOpen) setMobileToolsOpen(false);
  const { isControlPlane, selectedWorker } = useWorker();
  const showWorkerSwitch = isControlPlane && selectedWorker !== null;
  const hideCommunityLinks = useDisableShowPrompts();

  const handleWorkerSwitch = (workerId: string) => {
    clearTokenCookies();
    clearStoredReturnUrl();
    localStorage.removeItem("litellm_selected_worker_id");
    localStorage.removeItem("litellm_worker_url");
    window.location.href = `${getLoginUrl()}?worker=${encodeURIComponent(workerId)}`;
  };

  return (
    <header className="flex h-14 flex-none items-center justify-between gap-4 border-b border-border bg-background px-4 max-xl:gap-2 max-md:px-2">
      {navigationTrigger}
      <Breadcrumb className="min-w-0 max-xl:flex-1">
        <BreadcrumbList className="flex-nowrap">
          <BreadcrumbItem className="flex-none max-xl:hidden">
            <ViewSwitcher />
          </BreadcrumbItem>
          <BreadcrumbSeparator className="max-xl:hidden" />
          <BreadcrumbItem className="min-w-0">
            <BreadcrumbPage className="truncate">{t(title)}</BreadcrumbPage>
          </BreadcrumbItem>
        </BreadcrumbList>
      </Breadcrumb>

      <CommandPaletteTrigger />

      <div className="flex flex-none items-center gap-1 max-xl:hidden">
        {showWorkerSwitch && (
          <>
            <WorkerDropdown onWorkerSwitch={handleWorkerSwitch} />
            <ToolbarSeparator />
          </>
        )}
        <LiteAdmin />
        <DocsLink />
        <BlogDropdown />
        {!hideCommunityLinks && <CommunityEngagementButtons />}
        <ToolbarSeparator />
        <LanguageSelector />
        <ThemeToggle />
        {deploymentHealth}
        <NotificationsBell accessToken={accessToken} />
      </div>
      <div className="flex shrink-0 items-center gap-1 xl:hidden">
        <CommandPaletteTrigger mobile />
        {deploymentHealth}
        <NotificationsBell accessToken={accessToken} />
        <Popover open={!isDesktop && mobileToolsOpen} onOpenChange={setMobileToolsOpen}>
          <PopoverTrigger
            render={<Button variant="ghost" size="icon" className="size-11" aria-label={t("More options")} />}
          >
            <Ellipsis />
          </PopoverTrigger>
          <PopoverContent align="end" className="max-w-[calc(100vw-2rem)]">
            <PopoverTitle>{t("Gateway tools")}</PopoverTitle>
            <LanguageSelector />
            <ViewSwitcher />
            {showWorkerSwitch && <WorkerDropdown onWorkerSwitch={handleWorkerSwitch} />}
            <div className="flex flex-wrap items-center gap-2">
              <LiteAdmin />
              <ThemeToggle />
              <DocsLink />
              <BlogDropdown />
              {!hideCommunityLinks && <CommunityEngagementButtons />}
            </div>
          </PopoverContent>
        </Popover>
      </div>
    </header>
  );
}

export default DashboardHeader;
