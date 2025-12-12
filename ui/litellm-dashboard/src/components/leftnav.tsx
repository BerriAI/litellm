import {
  ApiOutlined,
  AppstoreOutlined,
  BankOutlined,
  BarChartOutlined,
  BgColorsOutlined,
  BlockOutlined,
  CreditCardOutlined,
  DatabaseOutlined,
  ExperimentOutlined,
  FileTextOutlined,
  KeyOutlined,
  LineChartOutlined,
  PlayCircleOutlined,
  RobotOutlined,
  SafetyOutlined,
  SearchOutlined,
  SettingOutlined,
  TagsOutlined,
  TeamOutlined,
  ToolOutlined,
  UserOutlined,
} from "@ant-design/icons";
import { Badge, ConfigProvider, Layout, Menu } from "antd";
import type { MenuProps } from "antd";
import { all_admin_roles, internalUserRoles, isAdminRole, rolesWithWriteAccess } from "../utils/roles";
import UsageIndicator from "./usage_indicator";
const { Sider } = Layout;

// Define the props type
interface SidebarProps {
  accessToken: string | null;
  setPage: (page: string) => void;
  userRole: string;
  defaultSelectedKey: string;
  collapsed?: boolean;
}

// Menu item configuration
interface MenuItem {
  key: string;
  page: string;
  label: string | React.ReactNode;
  roles?: string[];
  children?: MenuItem[];
  icon?: React.ReactNode;
}

// Group configuration
interface MenuGroup {
  groupLabel: string;
  items: MenuItem[];
  roles?: string[];
}

const Sidebar: React.FC<SidebarProps> = ({ accessToken, setPage, userRole, defaultSelectedKey, collapsed = false }) => {
  // Navigate to page helper
  const navigateToPage = (page: string) => {
    const newSearchParams = new URLSearchParams(window.location.search);
    newSearchParams.set("page", page);
    window.history.pushState(null, "", `?${newSearchParams.toString()}`);
    setPage(page);
  };

  // Menu groups organized by category
  const menuGroups: MenuGroup[] = [
    {
      groupLabel: "AI GATEWAY",
      items: [
        {
          key: "api-keys",
          page: "api-keys",
          label: "Virtual Keys",
          icon: <KeyOutlined />,
        },
        {
          key: "llm-playground",
          page: "llm-playground",
          label: "Playground",
          icon: <PlayCircleOutlined />,
          roles: rolesWithWriteAccess,
        },
        {
          key: "models",
          page: "models",
          label: "Models + Endpoints",
          icon: <BlockOutlined />,
          roles: rolesWithWriteAccess,
        },
        {
          key: "agents",
          page: "agents",
          label: (
            <span className="flex items-center gap-4">
              Agents <Badge color="blue" count="New" />
            </span>
          ),
          icon: <RobotOutlined />,
          roles: rolesWithWriteAccess,
        },
        {
          key: "mcp-servers",
          page: "mcp-servers",
          label: "MCP Servers",
          icon: <ToolOutlined />,
        },
        {
          key: "guardrails",
          page: "guardrails",
          label: "Guardrails",
          icon: <SafetyOutlined />,
          roles: all_admin_roles,
        },
        {
          key: "tools",
          page: "tools",
          label: "Tools",
          icon: <ToolOutlined />,
          children: [
            {
              key: "search-tools",
              page: "search-tools",
              label: "Search Tools",
              icon: <SearchOutlined />,
            },
            {
              key: "vector-stores",
              page: "vector-stores",
              label: "Vector Stores",
              icon: <DatabaseOutlined />,
              roles: all_admin_roles,
            },
          ],
        },
      ],
    },
    {
      groupLabel: "OBSERVABILITY",
      items: [
        {
          key: "new_usage",
          page: "new_usage",
          icon: <BarChartOutlined />,
          roles: [...all_admin_roles, ...internalUserRoles],
          label: (
            <span className="flex items-center gap-4">
              Usage <Badge color="blue" count="New" />
            </span>
        ),
        },
        {
          key: "logs",
          page: "logs",
          label: "Logs",
          icon: <LineChartOutlined />,
        },
      ],
    },
    {
      groupLabel: "ACCESS CONTROL",
      items: [
        {
          key: "users",
          page: "users",
          label: "Internal Users",
          icon: <UserOutlined />,
          roles: all_admin_roles,
        },
        {
          key: "teams",
          page: "teams",
          label: "Teams",
          icon: <TeamOutlined />,
        },
        {
          key: "organizations",
          page: "organizations",
          label: "Organizations",
          icon: <BankOutlined />,
          roles: all_admin_roles,
        },
        {
          key: "budgets",
          page: "budgets",
          label: "Budgets",
          icon: <CreditCardOutlined />,
          roles: all_admin_roles,
        },
      ],
    },
    {
      groupLabel: "DEVELOPER TOOLS",
      items: [
        {
          key: "api_ref",
          page: "api_ref",
          label: "API Reference",
          icon: <ApiOutlined />,
        },
        {
          key: "model-hub-table",
          page: "model-hub-table",
          label: "AI Hub",
          icon: <AppstoreOutlined />,
        },
        {
          key: "experimental",
          page: "experimental",
          label: "Experimental",
          icon: <ExperimentOutlined />,
          children: [
            {
              key: "caching",
              page: "caching",
              label: "Caching",
              icon: <DatabaseOutlined />,
              roles: all_admin_roles,
            },
            {
              key: "prompts",
              page: "prompts",
              label: "Prompts",
              icon: <FileTextOutlined />,
              roles: all_admin_roles,
            },
            {
              key: "transform-request",
              page: "transform-request",
              label: "API Playground",
              icon: <ApiOutlined />,
              roles: [...all_admin_roles, ...internalUserRoles],
            },
            {
              key: "tag-management",
              page: "tag-management",
              label: "Tag Management",
              icon: <TagsOutlined />,
              roles: all_admin_roles,
            },
            {
              key: "4",
              page: "usage",
              label: "Old Usage",
              icon: <BarChartOutlined />,
            },
          ],
        },
      ],
    },
    {
      groupLabel: "SETTINGS",
      roles: all_admin_roles,
      items: [
        {
          key: "settings",
          page: "settings",
          label: "Settings",
          icon: <SettingOutlined />,
          roles: all_admin_roles,
          children: [
            {
              key: "router-settings",
              page: "router-settings",
              label: "Router Settings",
              icon: <SettingOutlined />,
              roles: all_admin_roles,
            },
            {
              key: "logging-and-alerts",
              page: "logging-and-alerts",
              label: "Logging & Alerts",
              icon: <SettingOutlined />,
              roles: all_admin_roles,
            },
            {
              key: "admin-panel",
              page: "admin-panel",
              label: "Admin Settings",
              icon: <SettingOutlined />,
              roles: all_admin_roles,
            },
            {
              key: "cost-tracking",
              page: "cost-tracking",
              label: "Cost Tracking",
              icon: <BarChartOutlined />,
              roles: all_admin_roles,
            },
            {
              key: "ui-theme",
              page: "ui-theme",
              label: "UI Theme",
              icon: <BgColorsOutlined />,
              roles: all_admin_roles,
            },
          ],
        },
      ],
    },
  ];

  // Filter items based on user role
  const filterItemsByRole = (items: MenuItem[]): MenuItem[] => {
    return items
      .filter((item) => !item.roles || item.roles.includes(userRole))
      .map((item) => ({
        ...item,
        children: item.children ? filterItemsByRole(item.children) : undefined,
      }));
  };

  // Build menu items with groups
  const buildMenuItems = (): MenuProps["items"] => {
    const items: MenuProps["items"] = [];

    menuGroups.forEach((group) => {
      // Check if group has role restriction
      if (group.roles && !group.roles.includes(userRole)) {
        return;
      }

      const filteredItems = filterItemsByRole(group.items);
      if (filteredItems.length === 0) return;

      // Add group with items
      items.push({
        type: "group",
        label: collapsed ? null : (
          <span
            style={{
              fontSize: "10px",
              fontWeight: 600,
              color: "#6b7280",
              letterSpacing: "0.05em",
              padding: "12px 0 4px 12px",
              display: "block",
              marginBottom: "2px",
            }}
          >
            {group.groupLabel}
          </span>
        ),
        children: filteredItems.map((item) => ({
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
        })),
      });
    });

    return items;
  };

  // Find selected menu key
  const findMenuItemKey = (page: string): string => {
    for (const group of menuGroups) {
      for (const item of group.items) {
        if (item.page === page) return item.key;
        if (item.children) {
          const child = item.children.find((c) => c.page === page);
          if (child) return child.key;
        }
      }
    }
    return "api-keys";
  };

  const selectedMenuKey = findMenuItemKey(defaultSelectedKey);

  return (
    <Layout>
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
                iconSize: 15,
                fontSize: 13,
                itemMarginInline: 4,
                itemPaddingInline: 8,
                itemHeight: 30,
                itemBorderRadius: 6,
                subMenuItemBorderRadius: 6,
                groupTitleFontSize: 10,
                groupTitleLineHeight: 1.5,
              },
            },
          }}
        >
          <Menu
            mode="inline"
            selectedKeys={[selectedMenuKey]}
            defaultOpenKeys={[]}
            inlineCollapsed={collapsed}
            className="custom-sidebar-menu"
            style={{
              borderRight: 0,
              backgroundColor: "transparent",
              fontSize: "13px",
              paddingTop: "4px",
            }}
            items={buildMenuItems()}
          />
        </ConfigProvider>
        {isAdminRole(userRole) && !collapsed && <UsageIndicator accessToken={accessToken} width={220} />}
      </Sider>
    </Layout>
  );
};

export default Sidebar;
