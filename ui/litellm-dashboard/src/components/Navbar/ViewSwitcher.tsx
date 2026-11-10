import { useTerminology } from "@/contexts/TerminologyContext";
import React from "react";
import { useTranslation } from "react-i18next";
import { usePathname, useRouter } from "next/navigation";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Check, ChevronDown, ChevronsUpDown } from "lucide-react";
import { useSidebar } from "@/components/shared/Sidebar";
import BetaBadge from "@/components/BetaBadge";
import { usePluginMode } from "@/contexts/PluginModeContext";
import { useUISettings } from "@/app/(dashboard)/hooks/uiSettings/useUISettings";
import { uiHref } from "@/utils/uiHref";
import { cn } from "@/lib/cva.config";
import i18n from "@/i18n/i18n";
import { productNames } from "@/i18n/untranslated/productNames";

const GATEWAY = "ai-gateway";
const CUSTOMER_PORTAL = "customer-portal";
const CHAT = "chat";
const MOYAI = "moyai";

function isWorkspaceRoute(pathname: string, workspace: string): boolean {
  const routes = [uiHref(workspace)];
  if (process.env.NODE_ENV === "development") routes.push(`/ui/${workspace}`);
  return routes.some((route) => pathname === route || pathname.startsWith(`${route}/`));
}

function activeWorkspaceLabel(
  isChatRoute: boolean,
  isMoyaiRoute: boolean,
  pluginLabel: string | undefined,
  gatewayLabel: string,
): string {
  if (isChatRoute) return i18n.t("shell.viewSwitcher.chat");
  if (isMoyaiRoute) return "Moyai";
  return pluginLabel ?? gatewayLabel;
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
}

const pluginLabel = (mode: string, plugins: readonly { name: string; display_name: string }[], portalLabel: string) =>
  mode === CUSTOMER_PORTAL ? portalLabel : plugins.find((p) => p.name === mode)?.display_name;

export default function ViewSwitcher({ className, contentClassName }: ViewSwitcherProps) {
  const { t } = useTranslation();
  const term = useTerminology();
  const { collapsed } = useSidebar();
  const { mode, setMode, plugins } = usePluginMode();
  const { data: uiSettings } = useUISettings();
  const pathname = usePathname();
  const router = useRouter();

  const chatEnabled = Boolean(uiSettings?.values?.enable_chat_ui);
  const moyaiUrl = (uiSettings?.values?.moyai_url as string | undefined) ?? null;

  const normalizedPathname = (pathname ?? "").replace(/\/+$/, "");
  const isChatRoute = isWorkspaceRoute(normalizedPathname, CHAT);
  const isMoyaiRoute = isWorkspaceRoute(normalizedPathname, MOYAI);
  const isStandaloneRoute = isChatRoute || isMoyaiRoute;

  const gatewayLabel = term(t("shell.viewSwitcher.aiGateway"));
  const activeLabel = activeWorkspaceLabel(
    isChatRoute,
    isMoyaiRoute,
    pluginLabel(mode, plugins, t("customerPortal.switcher.label")),
    gatewayLabel,
  );

  const modeEntries = [
    { key: GATEWAY, label: gatewayLabel },
    ...plugins.map((p) => ({ key: p.name, label: p.display_name })),
  ];

  const selectMode = (key: string) => {
    if (key === CUSTOMER_PORTAL) {
      router.push(uiHref("customers"));
      return;
    }
    if (mode === CUSTOMER_PORTAL && key === GATEWAY) {
      router.push(uiHref("api-keys"));
      return;
    }
    setMode(key);
    if (isStandaloneRoute) {
      window.location.assign(uiHref(""));
    }
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
        {!isStandaloneRoute && mode === CUSTOMER_PORTAL && <Check className="size-4 text-info" />}
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
          {!isStandaloneRoute && e.key === mode && <Check className="size-4 text-info" />}
        </div>
      ),
      onClick: () => selectMode(e.key),
    })),
    customerPortalItem,
    chatItem,
    {
      key: MOYAI,
      label: (
        <div className="flex w-full items-center justify-between gap-6 py-0.5">
          <span className="flex flex-col">
            <span className="font-medium">{productNames.moyai}</span>{" "}
            <span className="text-xs leading-snug text-muted-foreground">
              {t("shell.viewSwitcher.moyaiDescription")}
            </span>
          </span>
          {isMoyaiRoute && <Check className="size-4 text-info" />}
        </div>
      ),
      onClick: () => window.location.assign(moyaiUrl ?? uiHref(MOYAI)),
    },
  ];

  return (
    <DropdownMenu>
      <DropdownMenuTrigger
        aria-label={t("shell.viewSwitcher.trigger", { workspace: activeLabel })}
        render={
          <Button
            type="button"
            variant="outline"
            size="sm"
            title={collapsed ? t("shell.viewSwitcher.trigger", { workspace: activeLabel }) : undefined}
            className={cn(
              "max-w-[220px] gap-1.5 px-3 dark:bg-transparent",
              className,
              collapsed &&
                "mx-auto flex size-9 justify-center gap-0 border-sidebar-border p-0 text-sidebar-foreground/70 hover:bg-sidebar-accent hover:text-sidebar-accent-foreground",
            )}
          />
        }
      >
        {collapsed ? (
          <ChevronDown className="size-4 text-muted-foreground" />
        ) : (
          <>
            <span data-slot="workspace-label" className="truncate">
              {activeLabel}
              {!isStandaloneRoute && mode === CUSTOMER_PORTAL && (
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
