import { Layout, Menu } from "antd";
import Link from "next/link";
import { List } from "postcss/lib/list";
import { Text } from "@tremor/react";
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
  LineOutlined,
  LineChartOutlined,
  ExperimentOutlined
} from '@ant-design/icons';

const { Sider } = Layout;

// Define the props type
interface SidebarProps {
  setPage: (page: string) => void;
  userRole: string;
  defaultSelectedKey: string;
}

// Create a more comprehensive menu item configuration
interface MenuItem {
  key: string;
  page: string;
  label: string;
  roles?: string[];
  children?: MenuItem[];  // Add children property for submenus
  icon?: React.ReactNode;
}

const old_admin_roles = ["Admin", "Admin Viewer"];
const v2_admin_role_names = ["proxy_admin", "proxy_admin_viewer", "org_admin"];
const all_admin_roles = [...old_admin_roles, ...v2_admin_role_names];
const rolesAllowedToSeeUsage = ["Admin", "Admin Viewer", "Internal User", "Internal Viewer"];


// Note: If a menu item does not have a role, it is visible to all roles.
const menuItems: MenuItem[] = [
  { key: "1", page: "api-keys", label: "Virtual Keys", icon: <KeyOutlined /> },
  { key: "3", page: "llm-playground", label: "Test Key", icon: <PlayCircleOutlined /> },
  { key: "2", page: "models", label: "Models", icon: <BlockOutlined />, roles: all_admin_roles },
  { key: "4", page: "usage", label: "Usage", icon: <BarChartOutlined /> },
  { key: "6", page: "teams", label: "Teams", icon: <TeamOutlined /> },
  { key: "17", page: "organizations", label: "Organizations", icon: <BankOutlined />, roles: all_admin_roles },
  { key: "5", page: "users", label: "Internal Users", icon: <UserOutlined />, roles: all_admin_roles },
  { key: "14", page: "api_ref", label: "API Reference", icon: <ApiOutlined /> },
  { key: "16", page: "model-hub", label: "Model Hub", icon: <AppstoreOutlined /> },
  { 
    key: "experimental", 
    page: "experimental",
    label: "Experimental", 
    icon: <ExperimentOutlined />,
    roles: all_admin_roles,
    children: [
      { key: "15", page: "logs", label: "Logs", icon: <LineChartOutlined />, roles: all_admin_roles },
      { key: "9", page: "caching", label: "Caching", icon: <DatabaseOutlined />, roles: all_admin_roles },
      { key: "10", page: "budgets", label: "Budgets", icon: <BankOutlined />, roles: all_admin_roles },
    ]
  },
  {
    key: "settings",
    page: "settings",
    label: "Settings",
    icon: <SettingOutlined />,
    roles: all_admin_roles,
    children: [
      { key: "11", page: "general-settings", label: "Router Settings", icon: <SettingOutlined />, roles: all_admin_roles },
      { key: "12", page: "pass-through-settings", label: "Pass-Through", icon: <ApiOutlined />, roles: all_admin_roles },
      { key: "8", page: "settings", label: "Logging & Alerts", icon: <SettingOutlined />, roles: all_admin_roles },
      { key: "13", page: "admin-panel", label: "Admin Settings", icon: <SettingOutlined />, roles: all_admin_roles },
    ]
  }
];

const Sidebar: React.FC<SidebarProps> = ({
  setPage,
  userRole,
  defaultSelectedKey,
}) => {
  // Find the menu item that matches the default page, including in submenus
  const findMenuItemKey = (page: string): string => {
    // Check top-level items
    const topLevelItem = menuItems.find(item => item.page === page);
    if (topLevelItem) return topLevelItem.key;

    // Check submenu items
    for (const item of menuItems) {
      if (item.children) {
        const childItem = item.children.find(child => child.page === page);
        if (childItem) return childItem.key;
      }
    }
    return "1"; // Default to first item if not found
  };

  const selectedMenuKey = findMenuItemKey(defaultSelectedKey);

  const filteredMenuItems = menuItems.filter(item => 
    !item.roles || item.roles.includes(userRole)
  );

  return (
    <Layout style={{ minHeight: "100vh" }}>
      <Sider theme="light" width={220}>
        <Menu
          mode="inline"
          selectedKeys={[selectedMenuKey]}
          style={{ 
            borderRight: 0,
            backgroundColor: 'transparent',
            fontSize: '14px',
          }}
          items={filteredMenuItems.map(item => ({
            key: item.key,
            icon: item.icon,
            label: item.label,
            children: item.children?.map(child => ({
              key: child.key,
              icon: child.icon,
              label: child.label,
              onClick: () => {
                const newSearchParams = new URLSearchParams(window.location.search);
                newSearchParams.set('page', child.page);
                window.history.pushState(null, '', `?${newSearchParams.toString()}`);
                setPage(child.page);
              }
            })),
            onClick: !item.children ? () => {
              const newSearchParams = new URLSearchParams(window.location.search);
              newSearchParams.set('page', item.page);
              window.history.pushState(null, '', `?${newSearchParams.toString()}`);
              setPage(item.page);
            } : undefined
          }))}
        />
      </Sider>
    </Layout>
  );
};

export default Sidebar;
