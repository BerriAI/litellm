import React from "react";
import { Dropdown } from "antd";
import { AppstoreOutlined, CheckOutlined, DownOutlined } from "@ant-design/icons";
import type { MenuProps } from "antd";
import { usePluginMode } from "@/contexts/PluginModeContext";

const GATEWAY = "ai-gateway";

export default function ViewSwitcher() {
  const { mode, setMode, plugins } = usePluginMode();

  // Only a switcher when there is at least one plugin to switch to.
  if (plugins.length === 0) return null;

  const activeLabel = plugins.find((p) => p.name === mode)?.display_name ?? "AI Gateway";

  const entries = [
    { value: GATEWAY, label: "AI Gateway" },
    ...plugins.map((p) => ({ value: p.name, label: p.display_name })),
  ];

  const items: MenuProps["items"] = entries.map((e) => ({
    key: e.value,
    label: (
      <div className="flex items-center justify-between gap-6 py-0.5">
        <span className="font-medium">{e.label}</span>
        {e.value === mode && <CheckOutlined className="text-blue-600" />}
      </div>
    ),
  }));

  const onClick: MenuProps["onClick"] = ({ key }) => setMode(key);

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
