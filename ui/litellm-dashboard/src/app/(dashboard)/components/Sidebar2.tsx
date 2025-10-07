import { Layout, Menu } from "antd";
import { usePathname } from "next/navigation";
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
import { ConfigProvider } from "antd";
import useFeatureFlags from "@/hooks/useFeatureFlags";
import { all_admin_roles, internalUserRoles, isAdminRole, rolesWithWriteAccess } from "@/utils/roles";
import UsageIndicator from "@/components/usage_indicator";
import React from "react";
const { Sider } = Layout;

interface SidebarProps {
  accessToken: string | null;
  setPage: (page: string) => void;
  userRole: string;
  defaultSelectedKey: string;
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

/** ---- BASE URL HELPERS ---- */
function normalizeBasePrefix(raw: string | undefined | null): string {
  const trimmed = (raw ?? "").trim();
  if (!trimmed) return ""; // no base
  const core = trimmed.replace(/^\/+/, "").replace(/\/+$/, "");
  return core ? `/${core}` : "";
}
const BASE_PREFIX = normalizeBasePrefix(process.env.NEXT_PUBLIC_BASE_URL);

/** Build an absolute path under the configured base. */
function withBase(path: string): string {
  // path can be "/virtual-keys" or "/?page=..."
  const p = path.startsWith("/") ? path : `/${path}`;
  return `${BASE_PREFIX}${p}` || p;
}

/** History-based navigation to prevent hard reloads / suspense flashes */
function softNavigate(url: string, replace = false) {
  if (typeof window === "undefined") return;
  if (replace) window.history.replaceState(null, "", url);
  else window.history.pushState(null, "", url);
}
/** -------------------------------- */

const Sidebar2: React.FC<SidebarProps> = ({
  accessToken,
  setPage,
  userRole,
  defaultSelectedKey,
  collapsed = false,
}) => {
  const pathname = usePathname();
  const { refactoredUIFlag } = useFeatureFlags();

  const menuItems: MenuItem[] = [
    { key: "1", page: "api-keys", label: "Virtual Keys", icon: <KeyOutlined style={{ fontSize: "18px" }} /> },
    {
      key: "3",
      page: "llm-playground",
      label: "Test Key",
      icon: <PlayCircleOutlined style={{ fontSize: "18px" }} />,
      roles: rolesWithWriteAccess,
    },
    {
      key: "2",
      page: "models",
      label: "Models + Endpoints",
      icon: <BlockOutlined style={{ fontSize: "18px" }} />,
      roles: rolesWithWriteAccess,
    },
    {
      key: "12",
      page: "new_usage",
      label: "Usage",
      icon: <BarChartOutlined style={{ fontSize: "18px" }} />,
      roles: [...all_admin_roles, ...internalUserRoles],
    },
    { key: "6", page: "teams", label: "Teams", icon: <TeamOutlined style={{ fontSize: "18px" }} /> },
    {
      key: "17",
      page: "organizations",
      label: "Organizations",
      icon: <BankOutlined style={{ fontSize: "18px" }} />,
      roles: all_admin_roles,
    },
    {
      key: "5",
      page: "users",
      label: "Internal Users",
      icon: <UserOutlined style={{ fontSize: "18px" }} />,
      roles: all_admin_roles,
    },
    { key: "14", page: "api_ref", label: "API Reference", icon: <ApiOutlined style={{ fontSize: "18px" }} /> },
    { key: "16", page: "model-hub-table", label: "Model Hub", icon: <AppstoreOutlined style={{ fontSize: "18px" }} /> },
    { key: "15", page: "logs", label: "Logs", icon: <LineChartOutlined style={{ fontSize: "18px" }} /> },
    {
      key: "11",
      page: "guardrails",
      label: "Guardrails",
      icon: <SafetyOutlined style={{ fontSize: "18px" }} />,
      roles: all_admin_roles,
    },
    {
      key: "26",
      page: "tools",
      label: "Tools",
      icon: <ToolOutlined style={{ fontSize: "18px" }} />,
      children: [
        { key: "18", page: "mcp-servers", label: "MCP Servers", icon: <ToolOutlined style={{ fontSize: "18px" }} /> },
        {
          key: "21",
          page: "vector-stores",
          label: "Vector Stores",
          icon: <DatabaseOutlined style={{ fontSize: "18px" }} />,
          roles: all_admin_roles,
        },
      ],
    },
    {
      key: "experimental",
      page: "experimental",
      label: "Experimental",
      icon: <ExperimentOutlined style={{ fontSize: "18px" }} />,
      children: [
        {
          key: "9",
          page: "caching",
          label: "Caching",
          icon: <DatabaseOutlined style={{ fontSize: "18px" }} />,
          roles: all_admin_roles,
        },
        {
          key: "25",
          page: "prompts",
          label: "Prompts",
          icon: <FileTextOutlined style={{ fontSize: "18px" }} />,
          roles: all_admin_roles,
        },
        {
          key: "10",
          page: "budgets",
          label: "Budgets",
          icon: <BankOutlined style={{ fontSize: "18px" }} />,
          roles: all_admin_roles,
        },
        {
          key: "20",
          page: "transform-request",
          label: "API Playground",
          icon: <ApiOutlined style={{ fontSize: "18px" }} />,
          roles: [...all_admin_roles, ...internalUserRoles],
        },
        {
          key: "19",
          page: "tag-management",
          label: "Tag Management",
          icon: <TagsOutlined style={{ fontSize: "18px" }} />,
          roles: all_admin_roles,
        },
        { key: "4", page: "usage", label: "Old Usage", icon: <BarChartOutlined style={{ fontSize: "18px" }} /> },
      ],
    },
    {
      key: "settings",
      page: "settings",
      label: "Settings",
      icon: <SettingOutlined style={{ fontSize: "18px" }} />,
      roles: all_admin_roles,
      children: [
        {
          key: "11",
          page: "general-settings",
          label: "Router Settings",
          icon: <SettingOutlined style={{ fontSize: "18px" }} />,
          roles: all_admin_roles,
        },
        {
          key: "8",
          page: "settings",
          label: "Logging & Alerts",
          icon: <SettingOutlined style={{ fontSize: "18px" }} />,
          roles: all_admin_roles,
        },
        {
          key: "13",
          page: "admin-panel",
          label: "Admin Settings",
          icon: <SettingOutlined style={{ fontSize: "18px" }} />,
          roles: all_admin_roles,
        },
        {
          key: "14",
          page: "ui-theme",
          label: "UI Theme",
          icon: <BgColorsOutlined style={{ fontSize: "18px" }} />,
          roles: all_admin_roles,
        },
      ],
    },
  ];

  const findMenuItemKey = (page: string): string => {
    const topLevelItem = menuItems.find((item) => item.page === page);
    if (topLevelItem) return topLevelItem.key;
    for (const item of menuItems) {
      if (item.children) {
        const childItem = item.children.find((child) => child.page === page);
        if (childItem) return childItem.key;
      }
    }
    return "1";
  };

  const selectedMenuKey = findMenuItemKey(defaultSelectedKey);

  const filteredMenuItems = menuItems.filter((item) => {
    const hasParentAccess = !item.roles || item.roles.includes(userRole);
    if (!hasParentAccess) return false;
    if (item.children) {
      item.children = item.children.filter((child) => !child.roles || child.roles.includes(userRole));
    }
    return true;
  });

  // Helper: update /?page=... under the configured base, WITHOUT triggering App Router nav
  const pushToRootWithPage = (page: string, useReplace = false) => {
    const params = new URLSearchParams();
    params.set("page", page);
    const url = withBase(`/?${params.toString()}`);
    softNavigate(url, useReplace);
  };

  const navigateToPage = (page: string) => {
    if (page === "api-keys") {
      if (refactoredUIFlag) {
        // vanity URL, keep SPA alive
        softNavigate(withBase("/virtual-keys"));
        return; // don't call setPage to keep parity, UI already shows api-keys by default
      }
      pushToRootWithPage(page);
      setPage(page);
      return;
    }

    if (refactoredUIFlag) {
      const onVirtualKeys =
        typeof window !== "undefined" && window.location.pathname.startsWith(withBase("/virtual-keys"));
      pushToRootWithPage(page, onVirtualKeys);
    } else {
      pushToRootWithPage(page);
    }
    setPage(page);
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
        style={{ transition: "all 0.3s cubic-bezier(0.4, 0, 0.2, 1)", position: "relative" }}
      >
        <ConfigProvider theme={{ components: { Menu: { iconSize: 18, fontSize: 14 } } }}>
          <Menu
            mode="inline"
            selectedKeys={[selectedMenuKey]}
            defaultOpenKeys={collapsed ? [] : ["llm-tools"]}
            inlineCollapsed={collapsed}
            className="custom-sidebar-menu"
            style={{ borderRight: 0, backgroundColor: "transparent", fontSize: "14px" }}
            items={filteredMenuItems.map((item) => ({
              key: item.key,
              icon: item.icon,
              label: item.label,
              children: item.children?.map((child) => ({
                key: child.key,
                icon: child.icon,
                label: child.label,
                onClick: () => navigateToPage(child.page),
              })),
              onClick: !item.children ? () => navigateToPage(item.page) : undefined,
            }))}
          />
        </ConfigProvider>
        {isAdminRole(userRole) && !collapsed && <UsageIndicator accessToken={accessToken} width={220} />}
      </Sider>
    </Layout>
  );
};

export default Sidebar2;
