import { useTerminology } from "@/contexts/TerminologyContext";
import { useGrantedApps } from "@/app/(dashboard)/apps/_components/useGrantedApps";
import React from "react";
import { useTranslation } from "react-i18next";
import { usePathname, useRouter } from "next/navigation";
import { Button } from "@/components/ui/button";
import { navMenuButtonClassName } from "@/components/shared/sidebarStyles";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Check, ChevronDown, ChevronsUpDown } from "lucide-react";
import { SidebarMenuButton } from "@/components/ui/sidebar";
import BetaBadge from "@/components/BetaBadge";
import { BUILT_IN_MODES, usePluginMode } from "@/contexts/PluginModeContext";
import { useUISettings } from "@/app/(dashboard)/hooks/uiSettings/useUISettings";
import { routeSegmentForPathname, uiHref } from "@/utils/uiHref";
import { cn } from "@/lib/cva.config";
import i18n from "@/i18n/i18n";
import { productNames } from "@/i18n/untranslated/productNames";

const GATEWAY = "ai-gateway";
const CUSTOMER_PORTAL = "customer-portal";
const CHAT = "chat";
const MOYAI = "moyai";
const LENS = "lens";

function isWorkspaceRoute(pathname: string, workspace: string): boolean {
  return routeSegmentForPathname(pathname) === workspace;
}

function routeWorkspaceLabel(pathname: string): string | undefined {
  if (isWorkspaceRoute(pathname, CHAT)) return i18n.t("shell.viewSwitcher.chat");
  if (isWorkspaceRoute(pathname, MOYAI)) return productNames.moyai;
  if (isWorkspaceRoute(pathname, LENS)) return productNames.lens;
  return undefined;
}

interface ViewSwitcherItem {
  key: string;
  label: React.ReactNode;
  disabled?: boolean;
  onClick?: () => void;
}

interface ViewSwitcherProps {
  className?: string;
  contentClassName?: string;
  collapsed?: boolean;
}

const pluginLabel = (mode: string, plugins: readonly { name: string; display_name: string }[], portalLabel: string) =>
  mode === CUSTOMER_PORTAL ? portalLabel : plugins.find((p) => p.name === mode)?.display_name;

function WorkspaceSwitcherTrigger({
  activeLabel,
  className,
  collapsed,
  compactLens,
  showPortalBadge,
}: {
  activeLabel: string;
  className?: string;
  collapsed: boolean;
  compactLens: boolean;
  showPortalBadge: boolean;
}) {
  const { t } = useTranslation();
  const triggerLabel = t("shell.viewSwitcher.trigger", { workspace: activeLabel });
  return (
    <DropdownMenuTrigger
      aria-label={triggerLabel}
      render={
        compactLens ? (
          <SidebarMenuButton type="button" title={triggerLabel} className={navMenuButtonClassName} />
        ) : (
          <Button
            type="button"
            variant="outline"
            size="sm"
            title={collapsed ? triggerLabel : undefined}
            className={cn(
              "max-w-[220px] gap-1.5 px-3 dark:bg-transparent",
              className,
              collapsed &&
                "mx-auto flex size-9 justify-center gap-0 border-sidebar-border p-0 text-sidebar-foreground/70 hover:bg-sidebar-accent hover:text-sidebar-accent-foreground",
            )}
          />
        )
      }
    >
      {collapsed ? (
        <ChevronDown aria-hidden className={compactLens ? undefined : "size-4 text-muted-foreground"} />
      ) : (
        <>
          <span data-slot="workspace-label" className="truncate">
            {activeLabel}
            {showPortalBadge && (
              <>
                {" "}
                <BetaBadge />
              </>
            )}
          </span>
          <ChevronsUpDown data-slot="workspace-chevron" className="size-3.5 flex-none text-muted-foreground" />
        </>
      )}
    </DropdownMenuTrigger>
  );
}

export default function ViewSwitcher({ className, contentClassName, collapsed = false }: ViewSwitcherProps) {
  const { t } = useTranslation();
  const term = useTerminology();
  const { mode, setMode, plugins } = usePluginMode();
  const { data: uiSettings } = useUISettings();
  const pathname = usePathname();
  const router = useRouter();

  const chatEnabled = Boolean(uiSettings?.values?.enable_chat_ui);
  const { isProxyAdmin, opensCustomerManagement } = useGrantedApps();

  const normalizedPathname = (pathname ?? "").replace(/\/+$/, "");
  const isChatRoute = isWorkspaceRoute(normalizedPathname, CHAT);
  const isMoyaiRoute = isWorkspaceRoute(normalizedPathname, MOYAI);
  const isLensRoute = isWorkspaceRoute(normalizedPathname, LENS);
  const isStandaloneRoute = isChatRoute || isMoyaiRoute;
  const isGatewayModeActive = !isStandaloneRoute && !isLensRoute;

  const gatewayLabel = term(t("shell.viewSwitcher.aiGateway"));
  const activeLabel =
    routeWorkspaceLabel(normalizedPathname) ??
    pluginLabel(mode, plugins, t("customerPortal.switcher.label")) ??
    gatewayLabel;

  const modeEntries = [
    { key: GATEWAY, label: gatewayLabel },
    ...plugins.map((p) => ({ key: p.name, label: p.display_name })),
  ];

  const selectMode = (key: string) => {
    if (key === CUSTOMER_PORTAL) {
      router.push(uiHref(isProxyAdmin ? "customers" : "apps"));
      return;
    }
    if (mode === CUSTOMER_PORTAL && key === GATEWAY) {
      router.push(uiHref("api-keys"));
      return;
    }
    setMode(key);
    if (isStandaloneRoute) {
      window.location.assign(uiHref(""));
      return;
    }
    if (isLensRoute) router.push(uiHref(""));
  };

  const openLens = () => {
    if (!BUILT_IN_MODES.has(mode)) setMode(GATEWAY);
    if (isStandaloneRoute) {
      window.location.assign(uiHref(LENS));
      return;
    }
    router.push(uiHref(LENS));
  };

  const chatItem: ViewSwitcherItem = {
    key: CHAT,
    disabled: !chatEnabled,
    label: (
      <div className="flex w-full items-center justify-between gap-6 py-0.5">
        <span className="flex max-w-[220px] flex-col">
          <span className="font-medium">{t("shell.viewSwitcher.chat")}</span>{" "}
          <span className="text-xs leading-snug text-muted-foreground">{t("shell.viewSwitcher.chatDescription")}</span>{" "}
          {!chatEnabled && (
            <span className="whitespace-normal text-xs leading-snug text-muted-foreground">
              {t("shell.viewSwitcher.chatDisabled")}
            </span>
          )}
        </span>
        {isChatRoute && <Check className="size-4 text-info" />}
      </div>
    ),
    onClick: chatEnabled ? () => window.location.assign(uiHref(CHAT)) : undefined,
  };

  const customerPortalItem: ViewSwitcherItem = {
    key: CUSTOMER_PORTAL,
    label: (
      <div className="flex w-full items-center justify-between gap-6 py-0.5">
        <span className="flex flex-col">
          <span className="font-medium">
            {t("customerPortal.switcher.label")} <BetaBadge />
          </span>{" "}
          <span className="text-xs leading-snug text-muted-foreground">{t("customerPortal.switcher.description")}</span>
        </span>
        {isGatewayModeActive && mode === CUSTOMER_PORTAL && <Check className="size-4 text-info" />}
      </div>
    ),
    onClick: () => selectMode(CUSTOMER_PORTAL),
  };
  const items: ViewSwitcherItem[] = [
    ...modeEntries.map((e) => ({
      key: e.key,
      label: (
        <div className="flex w-full items-center justify-between gap-6 py-0.5">
          <span className="font-medium">{e.label}</span>
          {isGatewayModeActive && e.key === mode && <Check className="size-4 text-info" />}
        </div>
      ),
      onClick: () => selectMode(e.key),
    })),
    ...[customerPortalItem].filter(() => opensCustomerManagement),
    chatItem,
    {
      key: LENS,
      label: (
        <div className="flex w-full items-center justify-between gap-6 py-0.5">
          <span className="flex flex-col">
            <span className="font-medium">
              {productNames.lens} <BetaBadge />
            </span>{" "}
            <span className="text-xs leading-snug text-muted-foreground">
              {t("shell.viewSwitcher.lensDescription")}
            </span>
          </span>
          {isLensRoute && <Check className="size-4 text-info" />}
        </div>
      ),
      onClick: openLens,
    },
  ];

  return (
    <DropdownMenu>
      <WorkspaceSwitcherTrigger
        activeLabel={activeLabel}
        className={className}
        collapsed={collapsed}
        compactLens={collapsed && isLensRoute}
        showPortalBadge={isGatewayModeActive && mode === CUSTOMER_PORTAL}
      />
      <DropdownMenuContent className={cn("w-auto", contentClassName)}>
        {items.map((item) => (
          <DropdownMenuItem key={item.key} disabled={item.disabled} onClick={item.onClick} className="w-full">
            {item.label}
          </DropdownMenuItem>
        ))}
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
