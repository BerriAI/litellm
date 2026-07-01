import React from "react";
import { Dropdown } from "antd";
import { AppstoreOutlined, CheckOutlined, DownOutlined } from "@ant-design/icons";
import type { MenuProps } from "antd";
import { usePluginMode } from "@/contexts/PluginModeContext";
import { useUISettings } from "@/app/(dashboard)/hooks/uiSettings/useUISettings";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { isAdminRole } from "@/utils/roles";
import { migratedHref } from "@/utils/migratedPages";

const GATEWAY = "ai-gateway";
const CHAT = "chat";
const ADMIN_SETTINGS_KEY = "admin-panel";

export default function ViewSwitcher() {
  const { mode, setMode, plugins } = usePluginMode();
  const { data: uiSettings } = useUISettings();
  const { userRole } = useAuthorized();

  const chatEnabled = Boolean(uiSettings?.values?.enable_chat_ui);
  const isAdmin = isAdminRole(userRole ?? "");
  // Admins see the Chat entry even when disabled, so they can discover and turn
  // it on; everyone else only sees it once it's enabled for users.
  const showChatEntry = chatEnabled || isAdmin;

  // Only render a switcher when there is somewhere to switch to.
  if (plugins.length === 0 && !showChatEntry) return null;

  const activeLabel = plugins.find((p) => p.name === mode)?.display_name ?? "AI Gateway";

  const modeEntries = [
    { key: GATEWAY, label: "AI Gateway" },
    ...plugins.map((p) => ({ key: p.name, label: p.display_name })),
  ];

  const items: MenuProps["items"] = [
    ...modeEntries.map((e) => ({
      key: e.key,
      label: (
        <div className="flex items-center justify-between gap-6 py-0.5">
          <span className="font-medium">{e.label}</span>
          {e.key === mode && <CheckOutlined className="text-blue-600" />}
        </div>
      ),
    })),
    ...(showChatEntry
      ? [
          {
            key: CHAT,
            label: chatEnabled ? (
              <div className="flex items-center justify-between gap-6 py-0.5">
                <span className="font-medium">Chat</span>
              </div>
            ) : (
              <div className="flex flex-col gap-0.5 py-0.5">
                <span className="font-medium text-gray-400">Chat</span>
                <span className="text-xs text-gray-400">Enable in Admin Settings &rarr; UI Settings</span>
              </div>
            ),
          },
        ]
      : []),
  ];

  const onClick: MenuProps["onClick"] = ({ key }) => {
    if (key === CHAT) {
      // Enabled: go to the chat app. Disabled (admin-only): send them to the
      // admin panel where the enable_chat_ui toggle lives.
      window.location.assign(migratedHref(chatEnabled ? CHAT : ADMIN_SETTINGS_KEY));
      return;
    }
    setMode(key);
  };

  return (
    <Dropdown menu={{ items, onClick, selectedKeys: [mode] }} trigger={["click"]}>
      <button
        type="button"
        className="flex items-center gap-2 rounded-md border border-gray-200 px-2.5 py-1.5 text-sm font-medium text-gray-700 transition-colors hover:bg-gray-50"
      >
        <AppstoreOutlined className="text-gray-500" />
        <span>{activeLabel}</span>
        <DownOutlined className="text-[10px] text-gray-400" />
      </button>
    </Dropdown>
  );
}
