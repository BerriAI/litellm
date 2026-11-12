"use client";

import { useTranslation } from "react-i18next";
import { useState } from "react";
import Link from "next/link";
import { PanelLeftClose, PanelLeftOpen } from "lucide-react";
import { useTheme } from "@/contexts/ThemeContext";
import { useHealthReadinessDetails } from "@/app/(dashboard)/hooks/healthReadiness/useHealthReadinessDetails";
import { getProxyBaseUrl } from "@/components/networking";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { SidebarHeader, useSidebar } from "@/components/ui/sidebar";
import { uiHref } from "@/utils/uiHref";
import { cn } from "@/lib/cva.config";
import { productNames } from "@/i18n/untranslated/productNames";

const LOGO_CLASS_NAME =
  "h-5 w-auto max-w-[150px] object-contain group-data-[collapsible=icon]:h-6 group-data-[collapsible=icon]:w-6";

function bundledLogoSrc(baseUrl: string, { dark, monogram }: { dark: boolean; monogram: boolean }): string {
  const query = [dark && "theme=dark", monogram && "variant=monogram"].filter(Boolean).join("&");
  return `${baseUrl}/get_image${query ? `?${query}` : ""}`;
}

export function SidebarBrand({
  accessToken,
  homeHref = uiHref(""),
  homeLabel = "LiteLLM home",
}: {
  accessToken: string | null;
  homeHref?: string;
  homeLabel?: string;
}) {
  const { t } = useTranslation();
  const { state, isMobile, toggleSidebar, setOpenMobile } = useSidebar();
  const collapsed = state === "collapsed" && !isMobile;
  const { logoUrl, logoUrlDark } = useTheme();
  const [erroredDarkLogo, setErroredDarkLogo] = useState<string | null>(null);
  const { data: healthData } = useHealthReadinessDetails(accessToken);
  const version = healthData?.litellm_version;
  const baseUrl = getProxyBaseUrl();
  const logoSrc = logoUrl || bundledLogoSrc(baseUrl, { dark: false, monogram: collapsed });
  const reachableDarkLogo = logoUrlDark === erroredDarkLogo ? null : logoUrlDark;
  const darkLogoSrc = reachableDarkLogo || logoUrl || bundledLogoSrc(baseUrl, { dark: true, monogram: collapsed });
  return (
    <SidebarHeader className="h-14 flex-none border-b border-border p-3 group-data-[collapsible=icon]:h-auto group-data-[collapsible=icon]:py-2">
      <div className="flex items-center justify-between gap-2 group-data-[collapsible=icon]:flex-col group-data-[collapsible=icon]:gap-1">
        <div className="flex min-w-0 items-center gap-2">
          <Link
            href={homeHref}
            onClick={() => setOpenMobile(false)}
            className="flex min-w-0 items-center"
            aria-label={homeLabel}
          >
            <img src={logoSrc} alt={productNames.liteLLM} className={cn(LOGO_CLASS_NAME, "dark:hidden")} />
            <img
              src={darkLogoSrc}
              alt=""
              aria-hidden
              onError={() => setErroredDarkLogo(logoUrlDark)}
              className={cn(LOGO_CLASS_NAME, "hidden dark:block")}
            />
          </Link>
          {version && (
            <Badge
              variant="outline"
              render={<a href="https://docs.litellm.ai/release_notes" target="_blank" rel="noopener noreferrer" />}
              className="px-1.5 py-0 font-mono text-[10px] font-medium text-muted-foreground group-data-[collapsible=icon]:hidden"
            >
              v{version}
            </Badge>
          )}
        </div>
        <Button
          variant="ghost"
          size="icon-sm"
          onClick={toggleSidebar}
          aria-expanded={!collapsed}
          aria-label={collapsed ? t("shell.navbar.expandSidebar") : t("shell.navbar.collapseSidebar")}
          className="flex-none text-muted-foreground group-data-[collapsible=icon]:size-9"
        >
          {collapsed ? <PanelLeftOpen /> : <PanelLeftClose />}
        </Button>
      </div>
    </SidebarHeader>
  );
}
