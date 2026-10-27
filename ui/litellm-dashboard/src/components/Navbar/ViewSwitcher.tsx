import React from "react";
import Image from "next/image";
import { usePathname } from "next/navigation";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Check, ChevronsUpDown, LayoutGrid, MessageCircle } from "lucide-react";
import { usePluginMode } from "@/contexts/PluginModeContext";
import { useUISettings } from "@/app/(dashboard)/hooks/uiSettings/useUISettings";
import { uiHref } from "@/utils/uiHref";
import { cn } from "@/lib/cva.config";
import moyaiHead from "../../../public/assets/moyai/moyai-head.svg";

const GATEWAY = "ai-gateway";
const CHAT = "chat";
const MOYAI = "moyai";
type WorkspaceIconName = typeof GATEWAY | typeof CHAT | typeof MOYAI;

function isWorkspaceRoute(pathname: string, workspace: string): boolean {
  const routes = [uiHref(workspace)];
  if (process.env.NODE_ENV === "development") routes.push(`/ui/${workspace}`);
  return routes.some((route) => pathname === route || pathname.startsWith(`${route}/`));
}

function activeWorkspaceFor(
  isChatRoute: boolean,
  isMoyaiRoute: boolean,
  pluginLabel: string | undefined,
): { label: string; icon: WorkspaceIconName } {
  if (isChatRoute) return { label: "Chat", icon: CHAT };
  if (isMoyaiRoute) return { label: "Moyai", icon: MOYAI };
  return { label: pluginLabel ?? "AI Gateway", icon: GATEWAY };
}

function WorkspaceIcon({ workspace }: { workspace: WorkspaceIconName }) {
  if (workspace === CHAT) {
    return (
      <span
        data-slot="workspace-icon"
        aria-hidden="true"
        className="flex size-6 flex-none items-center justify-center rounded-md bg-primary/10 text-primary ring-1 ring-inset ring-primary/15"
      >
        <MessageCircle className="size-4" />
      </span>
    );
  }
  if (workspace === MOYAI) {
    return (
      <span data-slot="workspace-icon" aria-hidden="true" className="flex size-6 flex-none items-center justify-center">
        <Image src={moyaiHead.src} alt="" width={20} height={20} unoptimized className="size-5 object-contain" />
      </span>
    );
  }
  return (
    <span
      data-slot="workspace-icon"
      aria-hidden="true"
      className="flex size-6 flex-none items-center justify-center rounded-md bg-muted text-muted-foreground"
    >
      <LayoutGrid className="size-4" />
    </span>
  );
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

export default function ViewSwitcher({ className, contentClassName }: ViewSwitcherProps) {
  const { mode, setMode, plugins } = usePluginMode();
  const { data: uiSettings } = useUISettings();
  const pathname = usePathname();

  const chatEnabled = Boolean(uiSettings?.values?.enable_chat_ui);
  const moyaiUrl = (uiSettings?.values?.moyai_url as string | undefined) ?? null;

  const normalizedPathname = (pathname ?? "").replace(/\/+$/, "");
  const isChatRoute = isWorkspaceRoute(normalizedPathname, CHAT);
  const isMoyaiRoute = isWorkspaceRoute(normalizedPathname, MOYAI);
  const isStandaloneRoute = isChatRoute || isMoyaiRoute;

  const { label: activeLabel, icon: activeIcon } = activeWorkspaceFor(
    isChatRoute,
    isMoyaiRoute,
    plugins.find((p) => p.name === mode)?.display_name,
  );

  const modeEntries = [
    { key: GATEWAY, label: "AI Gateway" },
    ...plugins.map((p) => ({ key: p.name, label: p.display_name })),
  ];

  const selectMode = (key: string) => {
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
        <span className="flex items-center gap-2">
          <WorkspaceIcon workspace={CHAT} />
          <span className="flex max-w-[220px] flex-col">
            <span className="font-medium">Chat</span>{" "}
            <span className="text-xs leading-snug text-muted-foreground">Chat UI for using LLMs, MCPs, Agents</span>{" "}
            {!chatEnabled && (
              <span className="whitespace-normal text-xs leading-snug text-muted-foreground">
                Admins can enable in Settings
              </span>
            )}
          </span>
        </span>
        {isChatRoute && <Check className="size-4 text-info" />}
      </div>
    ),
    onClick: chatEnabled ? () => window.location.assign(uiHref(CHAT)) : undefined,
  };

  const items: ViewSwitcherItem[] = [
    ...modeEntries.map((e) => ({
      key: e.key,
      label: (
        <div className="flex w-full items-center justify-between gap-6 py-0.5">
          <span className="flex items-center gap-2">
            <WorkspaceIcon workspace={GATEWAY} />
            <span className="font-medium">{e.label}</span>
          </span>
          {!isStandaloneRoute && e.key === mode && <Check className="size-4 text-info" />}
        </div>
      ),
      onClick: () => selectMode(e.key),
    })),
    chatItem,
    {
      key: MOYAI,
      label: (
        <div className="flex w-full items-center justify-between gap-6 py-0.5">
          <span className="flex items-center gap-2">
            <WorkspaceIcon workspace={MOYAI} />
            <span className="flex flex-col">
              <span className="font-medium">Moyai</span>{" "}
              <span className="text-xs leading-snug text-muted-foreground">Cloud Coding Agent</span>
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
        aria-label={`Switch workspace, current: ${activeLabel}`}
        render={
          <Button
            type="button"
            variant="outline"
            size="sm"
            className={cn("max-w-[220px] gap-1.5 pl-1.5 pr-2", className)}
          />
        }
      >
        <WorkspaceIcon workspace={activeIcon} />
        <span data-slot="workspace-label" className="truncate">
          {activeLabel}
        </span>
        <ChevronsUpDown data-slot="workspace-chevron" className="size-3.5 flex-none text-muted-foreground" />
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
