"use client";

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
  AuditOutlined,
} from "@ant-design/icons";
// import {
//   all_admin_roles,
//   rolesWithWriteAccess,
//   internalUserRoles,
//   isAdminRole,
// } from "../utils/roles";
// import UsageIndicator from "./usage_indicator";
import * as React from "react";
import { useRouter, usePathname } from "next/navigation";
import { all_admin_roles, internalUserRoles, isAdminRole, rolesWithWriteAccess } from "@/utils/roles";
import UsageIndicator from "@/components/UsageIndicator";
import { serverRootPath } from "@/components/networking";

const { Sider } = Layout;

// -------- Types --------
interface SidebarProps {
  accessToken: string | null;
  userRole: string;
  /** Fallback selection id (legacy), used if path can't be matched */
  defaultSelectedKey: string;
  collapsed?: boolean;
}

interface MenuItemCfg {
  key: string;
  page: string; // legacy id; we map this to a path below
  label: string;
  roles?: string[];
  children?: MenuItemCfg[];
  icon?: React.ReactNode;
}

/** ---------- Base URL helpers ---------- */
/**
 * Normalizes NEXT_PUBLIC_BASE_URL to either "/" or "/ui/" (always with a trailing slash).
 * Supported env values: "" or "ui/".
 * Also considers the serverRootPath from the proxy config (e.g., "/my-custom-path").
 */
const getBasePath = () => {
  const raw = process.env.NEXT_PUBLIC_BASE_URL ?? "";
  const trimmed = raw.replace(/^\/+|\/+$/g, ""); // strip leading/trailing slashes
  const uiPath = trimmed ? `/${trimmed}/` : "/";

  // If serverRootPath is set and not "/", prepend it to the UI path
  if (serverRootPath && serverRootPath !== "/") {
    // Remove trailing slash from serverRootPath and ensure uiPath has no leading slash for proper joining
    const cleanServerRoot = serverRootPath.replace(/\/+$/, "");
    const cleanUiPath = uiPath.replace(/^\/+/, "");
    return `${cleanServerRoot}/${cleanUiPath}`;
  }

  return uiPath;
};

/** Map legacy `page` ids to real app routes (relative, no leading slash). */
const routeFor = (slug: string): string => {
  switch (slug) {
    // top level
    case "api-keys":
      return "virtual-keys";
    case "llm-playground":
      return "test-key";
    case "models":
      return "models-and-endpoints";
    case "new_usage":
      return "usage";
    case "teams":
      return "teams";
    case "organizations":
      return "organizations";
    case "users":
      return "users";
    case "api_ref":
      return "api-reference";
    case "model-hub-table":
      // If you intend the newer in-dashboard page, use "model-hub".
      return "model-hub";
    case "logs":
      return "logs";
    case "guardrails":
      return "guardrails";
    case "policies":
      return "policies";

    // tools
    case "mcp-servers":
      return "tools/mcp-servers";
    case "vector-stores":
      return "tools/vector-stores";

    // experimental
    case "caching":
      return "experimental/caching";
    case "prompts":
      return "experimental/prompts";
    case "budgets":
      return "experimental/budgets";
    case "transform-request":
      return "experimental/api-playground";
    case "tag-management":
      return "experimental/tag-management";
    case "claude-code-plugins":
      return "experimental/claude-code-plugins";
    case "usage": // "Old Usage"
      return "experimental/old-usage";

    // settings
    case "general-settings":
      return "settings/router-settings";
    case "settings": // "Logging & Alerts"
      return "settings/logging-and-alerts";
    case "admin-panel":
      return "settings/admin-settings";
    case "ui-theme":
      return "settings/ui-theme";

    default:
      // treat as already a relative path
      return slug.replace(/^\/+/, "");
  }
};

/** Prefix base path ("/" or "/ui/") */
const toHref = (slugOrPath: string) => {
  const base = getBasePath(); // "/" or "/ui/"
  const rel = routeFor(slugOrPath).replace(/^\/+|\/+$/g, "");
  return `${base}${rel}`;
};

// ----- Menu config (unchanged labels/icons; same appearance) -----
const menuItems: MenuItemCfg[] = [
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
  {
    key: "16",
    page: "model-hub-table",
    label: "Model Hub",
    icon: <AppstoreOutlined style={{ fontSize: 18 }} />,
  },
  { key: "15", page: "logs", label: "Logs", icon: <LineChartOutlined style={{ fontSize: 18 }} /> },
  {
    key: "11",
    page: "guardrails",
    label: "Guardrails",
    icon: <SafetyOutlined style={{ fontSize: 18 }} />,
    roles: all_admin_roles,
  },
  {
    key: "28",
    page: "policies",
    label: "Policies",
    icon: <AuditOutlined style={{ fontSize: 18 }} />,
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
      {
        key: "27",
        page: "claude-code-plugins",
        label: "Claude Code Plugins",
        icon: <ToolOutlined style={{ fontSize: 18 }} />,
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
        icon: <SettingOutlined style={{ fontSize: 18 }} />,
        roles: all_admin_roles,
      },
    ],
  },
];

const Sidebar2: React.FC<SidebarProps> = ({ accessToken, userRole, defaultSelectedKey, collapsed = false }) => {
  const router = useRouter();
  const pathname = usePathname() || "/";

  // ----- Filter by role without mutating originals -----
  const filteredMenuItems = React.useMemo<MenuItemCfg[]>(() => {
    return menuItems
      .filter((item) => !item.roles || item.roles.includes(userRole))
      .map((item) => ({
        ...item,
        children: item.children ? item.children.filter((c) => !c.roles || c.roles.includes(userRole)) : undefined,
      }));
  }, [userRole]);

  // ----- Compute selected key from current path -----
  const selectedMenuKey = React.useMemo(() => {
    const base = getBasePath();
    // strip base prefix and leading slash -> "virtual-keys", "tools/mcp-servers", etc.
    const rel = pathname.startsWith(base) ? pathname.slice(base.length) : pathname.replace(/^\/+/, "");
    const relLower = rel.toLowerCase();

    const matchesPath = (slug: string) => {
      const route = routeFor(slug).toLowerCase();
      return relLower === route || relLower.startsWith(`${route}/`);
    };

    // search top-level
    for (const item of filteredMenuItems) {
      if (!item.children && matchesPath(item.page)) return item.key;
      if (item.children) {
        for (const child of item.children) {
          if (matchesPath(child.page)) return child.key;
        }
      }
    }

    // fallback to legacy defaultSelectedKey mapping
    const fallback = filteredMenuItems.find((i) => i.page === defaultSelectedKey)?.key;
    if (fallback) return fallback;

    for (const item of filteredMenuItems) {
      if (item.children?.some((c) => c.page === defaultSelectedKey)) {
        const child = item.children.find((c) => c.page === defaultSelectedKey)!;
        return child.key;
      }
    }

    return "1";
  }, [pathname, filteredMenuItems, defaultSelectedKey]);

  // ----- Navigation -----
  const goTo = (slug: string) => {
    const href = toHref(slug);
    router.push(href);
  };

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
              Menu: {
                iconSize: 18,
                fontSize: 14,
              },
            },
          }}
        >
          <Menu
            mode="inline"
            selectedKeys={[selectedMenuKey]}
            defaultOpenKeys={collapsed ? [] : ["llm-tools"]} // kept to preserve original appearance
            inlineCollapsed={collapsed}
            className="custom-sidebar-menu"
            style={{
              borderRight: 0,
              backgroundColor: "transparent",
              fontSize: "14px",
            }}
            items={filteredMenuItems.map((item) => ({
              key: item.key,
              icon: item.icon,
              label: item.label,
              children: item.children?.map((child) => ({
                key: child.key,
                icon: child.icon,
                label: child.label,
                onClick: () => goTo(child.page),
              })),
              onClick: !item.children ? () => goTo(item.page) : undefined,
            }))}
          />
        </ConfigProvider>
        {isAdminRole(userRole) && !collapsed && <UsageIndicator accessToken={accessToken} width={220} />}
      </Sider>
    </Layout>
  );
};

export default Sidebar2;
