import React from "react";
import { usePathname } from "next/navigation";
import { Dropdown } from "antd";
import { AppstoreOutlined, CheckOutlined } from "@ant-design/icons";
import { ChevronsUpDown } from "lucide-react";
import type { MenuProps } from "antd";
import { usePluginMode } from "@/contexts/PluginModeContext";
import { useUISettings } from "@/app/(dashboard)/hooks/uiSettings/useUISettings";
import { migratedHref } from "@/utils/migratedPages";

const GATEWAY = "ai-gateway";
const CHAT = "chat";

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

  const chatItem = chatEnabled
    ? {
        key: CHAT,
        label: (
          <div className="flex items-center justify-between gap-6 py-0.5">
            <span className="font-medium">Chat</span>
            {isChatRoute && <CheckOutlined className="text-blue-600" />}
          </div>
        ),
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

  const items: MenuProps["items"] = [
    ...modeEntries.map((e) => ({
      key: e.key,
      label: (
        <div className="flex items-center justify-between gap-6 py-0.5">
          <span className="font-medium">{e.label}</span>
          {!isChatRoute && e.key === mode && <CheckOutlined className="text-blue-600" />}
        </div>
      ),
    })),
    chatItem,
  ];

  const onClick: MenuProps["onClick"] = ({ key }) => {
    if (key === CHAT) {
      window.location.assign(migratedHref(CHAT));
      return;
    }
    setMode(key);
    // The chat route lives outside the dashboard SPA shell that reacts to `mode`,
    // so switching modes from there needs a real navigation, not just state.
    if (isChatRoute) {
      window.location.assign(migratedHref(""));
    }
  };

  return (
    <Dropdown menu={{ items, onClick, selectedKeys: [isChatRoute ? CHAT : mode] }} trigger={["click"]}>
      <button
        type="button"
        className="flex h-8 max-w-[220px] items-center gap-1.5 rounded-md border border-border bg-background pl-1.5 pr-2 text-sm font-medium text-foreground transition-colors hover:bg-accent"
      >
        <span className="flex size-5 flex-none items-center justify-center rounded bg-muted text-muted-foreground">
          <AppstoreOutlined className="text-[13px]" />
        </span>
        <span className="truncate">{activeLabel}</span>
        <ChevronsUpDown className="size-3.5 flex-none text-muted-foreground" />
      </button>
    </Dropdown>
  );
}
