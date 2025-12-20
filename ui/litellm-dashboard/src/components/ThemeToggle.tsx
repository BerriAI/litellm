import React from "react";
import { Dropdown } from "antd";
import type { MenuProps } from "antd";
import { useTheme } from "@/contexts/ThemeContext";

const SunIcon = () => (
  <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
    <path
      strokeLinecap="round"
      strokeLinejoin="round"
      strokeWidth={2}
      d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z"
    />
  </svg>
);

const MoonIcon = () => (
  <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
    <path
      strokeLinecap="round"
      strokeLinejoin="round"
      strokeWidth={2}
      d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z"
    />
  </svg>
);

const SystemIcon = () => (
  <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
    <path
      strokeLinecap="round"
      strokeLinejoin="round"
      strokeWidth={2}
      d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z"
    />
  </svg>
);

const ThemeToggle: React.FC = () => {
  const { themeMode, setThemeMode, isDarkMode } = useTheme();

  const menuItems: MenuProps["items"] = [
    {
      key: "light",
      label: (
        <div className="flex items-center px-3 py-2 text-gray-700 dark:text-gray-200">
          <SunIcon />
          <span className="ml-2">Light</span>
          {themeMode === "light" && <span className="ml-auto text-blue-500">✓</span>}
        </div>
      ),
      onClick: () => setThemeMode("light"),
    },
    {
      key: "dark",
      label: (
        <div className="flex items-center px-3 py-2 text-gray-700 dark:text-gray-200">
          <MoonIcon />
          <span className="ml-2">Dark</span>
          {themeMode === "dark" && <span className="ml-auto text-blue-500">✓</span>}
        </div>
      ),
      onClick: () => setThemeMode("dark"),
    },
    {
      key: "system",
      label: (
        <div className="flex items-center px-3 py-2 text-gray-700 dark:text-gray-200">
          <SystemIcon />
          <span className="ml-2">System</span>
          {themeMode === "system" && <span className="ml-auto text-blue-500">✓</span>}
        </div>
      ),
      onClick: () => setThemeMode("system"),
    },
  ];

  return (
    <Dropdown
      menu={{
        items: menuItems,
        style: {
          padding: "4px",
          borderRadius: "8px",
        },
      }}
      trigger={["click"]}
      placement="bottomRight"
    >
      <button
        className="flex items-center justify-center w-9 h-9 text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white hover:bg-gray-100 dark:hover:bg-[#252525] rounded-lg transition-colors"
        title={`Theme: ${themeMode}`}
      >
        {isDarkMode ? <MoonIcon /> : <SunIcon />}
      </button>
    </Dropdown>
  );
};

export default ThemeToggle;
