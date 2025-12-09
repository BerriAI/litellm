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
import { ConfigProvider, Layout, Menu } from "antd";
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

// Create a more comprehensive menu item configuration
interface MenuItem {
  key: string;
  page: string;
  label: string;
  roles?: string[];
  children?: MenuItem[]; // Add children property for submenus
  icon?: React.ReactNode;
}

const Sidebar: React.FC<SidebarProps> = ({ accessToken, setPage, userRole, defaultSelectedKey, collapsed = false }) => {
  // Note: If a menu item does not have a role, it is visible to all roles.
  const menuItems: MenuItem[] = [
    {
      key: "1",
      page: "api-keys",
      label: "Virtual Keys",
      icon: <KeyOutlined style={{ fontSize: "18px" }} />,
    },
    {
      key: "3",
      page: "llm-playground",
      label: "Playground",
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
    {
      key: "10",
      page: "budgets",
      label: "Budgets",
      icon: <CreditCardOutlined style={{ fontSize: "18px" }} />,
      roles: all_admin_roles,
    },
    { key: "14", page: "api_ref", label: "API Reference", icon: <ApiOutlined style={{ fontSize: "18px" }} /> },
    {
      key: "16",
      page: "model-hub-table",
      label: "AI Hub",
      icon: <AppstoreOutlined style={{ fontSize: "18px" }} />,
    },
    { key: "15", page: "logs", label: "Logs", icon: <LineChartOutlined style={{ fontSize: "18px" }} /> },
    {
      key: "11",
      page: "guardrails",
      label: "Guardrails",
      icon: <SafetyOutlined style={{ fontSize: "18px" }} />,
      roles: all_admin_roles,
    },
    { key: "18", page: "mcp-servers", label: "MCP Servers", icon: <ToolOutlined style={{ fontSize: "18px" }} /> },
    {
      key: "26",
      page: "tools",
      label: "Tools",
      icon: <ToolOutlined style={{ fontSize: "18px" }} />,
      children: [
        {
          key: "28",
          page: "search-tools",
          label: "Search Tools",
          icon: <SearchOutlined style={{ fontSize: "18px" }} />,
        },
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
          key: "29",
          page: "agents",
          label: "Agents",
          icon: <RobotOutlined style={{ fontSize: "18px" }} />,
          roles: rolesWithWriteAccess,
        },
        {
          key: "25",
          page: "prompts",
          label: "Prompts",
          icon: <FileTextOutlined style={{ fontSize: "18px" }} />,
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
          key: "27",
          page: "cost-tracking-settings",
          label: "Cost Tracking",
          icon: <BarChartOutlined style={{ fontSize: "18px" }} />,
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
  // Find the menu item that matches the default page, including in submenus
  const findMenuItemKey = (page: string): string => {
    // Check top-level items
    const topLevelItem = menuItems.find((item) => item.page === page);
    if (topLevelItem) return topLevelItem.key;

    // Check submenu items
    for (const item of menuItems) {
      if (item.children) {
        const childItem = item.children.find((child) => child.page === page);
        if (childItem) return childItem.key;
      }
    }
    return "1"; // Default to first item if not found
  };

  const selectedMenuKey = findMenuItemKey(defaultSelectedKey);

  const filteredMenuItems = menuItems.filter((item) => {
    // Check if parent item has roles and user has access
    const hasParentAccess = !item.roles || item.roles.includes(userRole);

    console.log(`Menu item ${item.label}: roles=${item.roles}, userRole=${userRole}, hasAccess=${hasParentAccess}`);

    if (!hasParentAccess) return false;

    // Filter children if they exist
    if (item.children) {
      item.children = item.children.filter((child) => !child.roles || child.roles.includes(userRole));
    }

    return true;
  });

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
          transition: "all 0.3s cubic-bezier(0.4, 0, 0.2, 1)", // Material Design easing
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
            defaultOpenKeys={collapsed ? [] : ["llm-tools"]}
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
                onClick: () => {
                  const newSearchParams = new URLSearchParams(window.location.search);
                  newSearchParams.set("page", child.page);
                  window.history.pushState(null, "", `?${newSearchParams.toString()}`);
                  setPage(child.page);
                },
              })),
              onClick: !item.children
                ? () => {
                    const newSearchParams = new URLSearchParams(window.location.search);
                    newSearchParams.set("page", item.page);
                    window.history.pushState(null, "", `?${newSearchParams.toString()}`);
                    setPage(item.page);
                  }
                : undefined,
            }))}
          />
        </ConfigProvider>
        {isAdminRole(userRole) && !collapsed && <UsageIndicator accessToken={accessToken} width={220} />}
      </Sider>
    </Layout>
  );
};

export default Sidebar;
