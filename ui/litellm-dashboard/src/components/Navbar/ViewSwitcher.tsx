import React from "react";
import { usePathname } from "next/navigation";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Check, ChevronsUpDown, LayoutGrid } from "lucide-react";
import { usePluginMode } from "@/contexts/PluginModeContext";
import { useUISettings } from "@/app/(dashboard)/hooks/uiSettings/useUISettings";
import { migratedHref } from "@/utils/migratedPages";

const GATEWAY = "ai-gateway";
const CHAT = "chat";

interface ViewSwitcherItem {
  key: string;
  label: React.ReactNode;
  disabled?: boolean;
  onClick?: () => void;
}

export default function ViewSwitcher() {
  const { mode, setMode, plugins } = usePluginMode();
  const { data: uiSettings } = useUISettings();
  const pathname = usePathname();

  const chatEnabled = Boolean(uiSettings?.values?.enable_chat_ui);

  const chatHref = migratedHref(CHAT);
  const normalizedPathname = (pathname ?? "").replace(/\/+$/, "");
  const isChatRoute = chatEnabled && (normalizedPathname === chatHref || normalizedPathname.startsWith(`${chatHref}/`));

  const activeLabel = isChatRoute ? "Chat" : plugins.find((p) => p.name === mode)?.display_name ?? "AI Gateway";

  const modeEntries = [
    { key: GATEWAY, label: "AI Gateway" },
    ...plugins.map((p) => ({ key: p.name, label: p.display_name })),
  ];

  const selectMode = (key: string) => {
    setMode(key);
    // The chat route lives outside the dashboard SPA shell that reacts to `mode`,
    // so switching modes from there needs a real navigation, not just state.
    if (isChatRoute) {
      window.location.assign(migratedHref(""));
    }
  };

  const chatItem: ViewSwitcherItem = chatEnabled
    ? {
        key: CHAT,
        label: (
          <div className="flex items-center justify-between gap-6 py-0.5">
            <span className="font-medium">Chat</span>
            {isChatRoute && <Check className="size-4 text-info" />}
          </div>
        ),
        onClick: () => window.location.assign(migratedHref(CHAT)),
      }
    : {
        key: CHAT,
        disabled: true,
        label: (
          <div className="flex max-w-[220px] flex-col py-0.5">
            <span className="font-medium">Chat</span>
            <span className="whitespace-normal text-xs leading-snug text-muted-foreground">
              Admins can enable in Settings
            </span>
          </div>
        ),
      };

  const items: ViewSwitcherItem[] = [
    ...modeEntries.map((e) => ({
      key: e.key,
      label: (
        <div className="flex items-center justify-between gap-6 py-0.5">
          <span className="font-medium">{e.label}</span>
          {!isChatRoute && e.key === mode && <Check className="size-4 text-info" />}
        </div>
      ),
      onClick: () => selectMode(e.key),
    })),
    chatItem,
  ];

  return (
    <DropdownMenu>
      <DropdownMenuTrigger
        render={
          <button
            type="button"
            className="flex h-8 max-w-[220px] items-center gap-1.5 rounded-md border border-border bg-background pl-1.5 pr-2 text-sm font-medium text-foreground transition-colors hover:bg-accent"
          />
        }
      >
        <span className="flex size-5 flex-none items-center justify-center rounded bg-muted text-muted-foreground">
          <LayoutGrid className="size-[13px]" />
        </span>
        <span className="truncate">{activeLabel}</span>
        <ChevronsUpDown className="size-3.5 flex-none text-muted-foreground" />
      </DropdownMenuTrigger>
      <DropdownMenuContent className="w-auto">
        {items.map((item) => (
          <DropdownMenuItem key={item.key} disabled={item.disabled} onClick={item.onClick}>
            {item.label}
          </DropdownMenuItem>
        ))}
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
