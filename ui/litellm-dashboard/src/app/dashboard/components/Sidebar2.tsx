"use client";

import React from "react";
import Link from "next/link";
import { usePathname, useSearchParams } from "next/navigation";
import { Layout, Menu, ConfigProvider } from "antd";
import {
  KeyOutlined,
  PlayCircleOutlined,
  BlockOutlined,
  BarChartOutlined,
  TeamOutlined,
  BankOutlined,
  UserOutlined,
  SettingOutlined,
  ApiOutlined,
  AppstoreOutlined,
  DatabaseOutlined,
  FileTextOutlined,
  LineChartOutlined,
  SafetyOutlined,
  ExperimentOutlined,
  ToolOutlined,
  TagsOutlined,
  BgColorsOutlined,
} from "@ant-design/icons";
import { all_admin_roles, internalUserRoles, isAdminRole, rolesWithWriteAccess } from "@/utils/roles";
import UsageIndicator from "@/components/usage_indicator";

const { Sider } = Layout;

interface SidebarProps {
  accessToken: string | null;
  userRole: string;
  /** Used to highlight a menu item when pathname doesn't match (e.g., on non-routed pages) */
  defaultSelectedKey?: string;
  collapsed?: boolean;
}

interface MenuItem {
  key: string;
  page: string;
  label: string;
  roles?: string[];
  children?: MenuItem[];
  icon?: React.ReactNode;
}

const Sidebar2: React.FC<SidebarProps> = ({ accessToken, userRole, defaultSelectedKey, collapsed = false }) => {
  const menuItems: MenuItem[] = [
    { key: "1", page: "api-keys", label: "Virtual Keys", icon: <KeyOutlined style={{ fontSize: 18 }} /> },
    {
      key: "3",
      page: "llm-playground",
      label: "Test Key",
      icon: <PlayCircleOutlined style={{ fontSize: 18 }} />,
      roles: rolesWithWriteAccess,
    },
    {
      key: "2",
      page: "models",
      label: "Models + Endpoints",
      icon: <BlockOutlined style={{ fontSize: 18 }} />,
      roles: rolesWithWriteAccess,
    },
    {
      key: "12",
      page: "new_usage",
      label: "Usage",
      icon: <BarChartOutlined style={{ fontSize: 18 }} />,
      roles: [...all_admin_roles, ...internalUserRoles],
    },
    { key: "6", page: "teams", label: "Teams", icon: <TeamOutlined style={{ fontSize: 18 }} /> },
    {
      key: "17",
      page: "organizations",
      label: "Organizations",
      icon: <BankOutlined style={{ fontSize: 18 }} />,
      roles: all_admin_roles,
    },
    {
      key: "5",
      page: "users",
      label: "Internal Users",
      icon: <UserOutlined style={{ fontSize: 18 }} />,
      roles: all_admin_roles,
    },
    { key: "14", page: "api_ref", label: "API Reference", icon: <ApiOutlined style={{ fontSize: 18 }} /> },
    { key: "16", page: "model-hub-table", label: "Model Hub", icon: <AppstoreOutlined style={{ fontSize: 18 }} /> },
    { key: "15", page: "logs", label: "Logs", icon: <LineChartOutlined style={{ fontSize: 18 }} /> },
    {
      key: "11",
      page: "guardrails",
      label: "Guardrails",
      icon: <SafetyOutlined style={{ fontSize: 18 }} />,
      roles: all_admin_roles,
    },
    {
      key: "26",
      page: "tools",
      label: "Tools",
      icon: <ToolOutlined style={{ fontSize: 18 }} />,
      children: [
        { key: "18", page: "mcp-servers", label: "MCP Servers", icon: <ToolOutlined style={{ fontSize: 18 }} /> },
        {
          key: "21",
          page: "vector-stores",
          label: "Vector Stores",
          icon: <DatabaseOutlined style={{ fontSize: 18 }} />,
          roles: all_admin_roles,
        },
      ],
    },
    {
      key: "experimental",
      page: "experimental",
      label: "Experimental",
      icon: <ExperimentOutlined style={{ fontSize: 18 }} />,
      children: [
        {
          key: "9",
          page: "caching",
          label: "Caching",
          icon: <DatabaseOutlined style={{ fontSize: 18 }} />,
          roles: all_admin_roles,
        },
        {
          key: "25",
          page: "prompts",
          label: "Prompts",
          icon: <FileTextOutlined style={{ fontSize: 18 }} />,
          roles: all_admin_roles,
        },
        {
          key: "10",
          page: "budgets",
          label: "Budgets",
          icon: <BankOutlined style={{ fontSize: 18 }} />,
          roles: all_admin_roles,
        },
        {
          key: "20",
          page: "transform-request",
          label: "API Playground",
          icon: <ApiOutlined style={{ fontSize: 18 }} />,
          roles: [...all_admin_roles, ...internalUserRoles],
        },
        {
          key: "19",
          page: "tag-management",
          label: "Tag Management",
          icon: <TagsOutlined style={{ fontSize: 18 }} />,
          roles: all_admin_roles,
        },
        { key: "4", page: "usage", label: "Old Usage", icon: <BarChartOutlined style={{ fontSize: 18 }} /> },
      ],
    },
    {
      key: "settings",
      page: "settings",
      label: "Settings",
      icon: <SettingOutlined style={{ fontSize: 18 }} />,
      roles: all_admin_roles,
      children: [
        {
          key: "11",
          page: "general-settings",
          label: "Router Settings",
          icon: <SettingOutlined style={{ fontSize: 18 }} />,
          roles: all_admin_roles,
        },
        {
          key: "8",
          page: "settings",
          label: "Logging & Alerts",
          icon: <SettingOutlined style={{ fontSize: 18 }} />,
          roles: all_admin_roles,
        },
        {
          key: "13",
          page: "admin-panel",
          label: "Admin Settings",
          icon: <SettingOutlined style={{ fontSize: 18 }} />,
          roles: all_admin_roles,
        },
        {
          key: "14",
          page: "ui-theme",
          label: "UI Theme",
          icon: <BgColorsOutlined style={{ fontSize: 18 }} />,
          roles: all_admin_roles,
        },
      ],
    },
  ];

  // Role filtering (preserves original visibility behavior)
  const filteredMenuItems: MenuItem[] = menuItems
    .filter((item) => !item.roles || item.roles.includes(userRole))
    .map((item) => ({
      ...item,
      children: item.children?.filter((child) => !child.roles || child.roles.includes(userRole)),
    }));

  // Highlight selection based on pathname or ?page=
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const pageParam = searchParams.get("page") || undefined;

  const findMenuItemKey = (page: string): string => {
    const top = filteredMenuItems.find((i) => i.page === page);
    if (top) return top.key;
    for (const i of filteredMenuItems) {
      const child = i.children?.find((c) => c.page === page);
      if (child) return child.key;
    }
    return "1";
  };

  const selectedMenuKey =
    pathname === "/dashboard/virtual-keys"
      ? "1"
      : pageParam
        ? findMenuItemKey(pageParam)
        : defaultSelectedKey
          ? findMenuItemKey(defaultSelectedKey)
          : "1";

  // Root-only routing helper: always replace everything after the domain
  const rootWithPage = (p: string) => ({ pathname: "/", query: { page: p } });

  // Convert to AntD Menu items:
  // - "Virtual Keys" still routes to /virtual-keys
  // - All other items (and children) route to "/?page=<page>"
  const antdItems = filteredMenuItems.map((item) => {
    const isVirtualKeys = item.key === "1";
    const label = isVirtualKeys ? (
      <Link href="/virtual-keys">Virtual Keys</Link>
    ) : (
      <Link href={rootWithPage(item.page)}>{item.label}</Link>
    );

    return {
      key: item.key,
      icon: item.icon,
      label,
      children: item.children?.map((child) => ({
        key: child.key,
        icon: child.icon,
        label: <Link href={rootWithPage(child.page)}>{child.label}</Link>,
      })),
    };
  });

  return (
    <Layout style={{ minHeight: "100vh" }}>
      <Sider
        theme="light"
        width={220}
        collapsed={collapsed}
        collapsedWidth={80}
        collapsible
        trigger={null}
        style={{
          transition: "all 0.3s cubic-bezier(0.4, 0, 0.2, 1)",
          position: "relative",
        }}
      >
        <ConfigProvider
          theme={{
            components: {
              Menu: { iconSize: 18, fontSize: 14 },
            },
          }}
        >
          <Menu
            mode="inline"
            selectedKeys={[selectedMenuKey]}
            defaultOpenKeys={collapsed ? [] : ["llm-tools"]} /* kept to match original look */
            inlineCollapsed={collapsed}
            className="custom-sidebar-menu"
            style={{ borderRight: 0, backgroundColor: "transparent", fontSize: 14 }}
            items={antdItems as any}
          />
        </ConfigProvider>

        {isAdminRole(userRole) && !collapsed && <UsageIndicator accessToken={accessToken} width={220} />}
      </Sider>
    </Layout>
  );
};

export default Sidebar2;
