import { Layout, Menu } from "antd";
import Link from "next/link";
import { List } from "postcss/lib/list";
import { Text } from "@tremor/react";

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
}

const old_admin_roles = ["Admin", "Admin Viewer"];
const v2_admin_role_names = ["proxy_admin", "proxy_admin_viewer", "org_admin"];
const all_admin_roles = [...old_admin_roles, ...v2_admin_role_names];
const rolesAllowedToSeeUsage = ["Admin", "Admin Viewer", "Internal User", "Internal Viewer"];


// Note: If a menu item does not have a role, it is visible to all roles.
const menuItems: MenuItem[] = [
  { key: "1", page: "api-keys", label: "Virtual Keys" }, // all roles
  { key: "3", page: "llm-playground", label: "Test Key" }, // all roles
  { key: "2", page: "models", label: "Models", roles: all_admin_roles },
  { key: "4", page: "usage", label: "Usage"}, // all roles
  { key: "6", page: "teams", label: "Teams" },
  { key: "5", page: "users", label: "Internal Users", roles: all_admin_roles },
  { key: "8", page: "settings", label: "Logging & Alerts", roles: all_admin_roles },
  { key: "9", page: "caching", label: "Caching", roles: all_admin_roles },
  { key: "10", page: "budgets", label: "Budgets", roles: all_admin_roles },
  { key: "11", page: "general-settings", label: "Router Settings", roles: all_admin_roles },
  { key: "12", page: "pass-through-settings", label: "Pass-Through", roles: all_admin_roles },
  { key: "13", page: "admin-panel", label: "Admin Settings", roles: all_admin_roles },
  { key: "14", page: "api_ref", label: "API Reference" }, // all roles
  { key: "16", page: "model-hub", label: "Model Hub" }, // all roles
];

// The Sidebar component can now be simplified to:
const Sidebar: React.FC<SidebarProps> = ({
  setPage,
  userRole,
  defaultSelectedKey,
}) => {
  // Find the menu item that matches the default page to get its key
  const selectedMenuItem = menuItems.find(item => item.page === defaultSelectedKey);
  const selectedMenuKey = selectedMenuItem?.key || "1";

  const filteredMenuItems = menuItems.filter(item => 
    !item.roles || item.roles.includes(userRole)
  );

  return (
    <Layout style={{ minHeight: "100vh", maxWidth: userRole === "Admin Viewer" ? "120px" : "145px" }}>
      <Sider width={userRole === "Admin Viewer" ? 120 : 145}>
        <Menu
          mode="inline"
          selectedKeys={[selectedMenuKey]}
          style={{ height: "100%", borderRight: 0 }}
        >
          {filteredMenuItems.map(item => (
            <Menu.Item 
            key={item.key} 
            onClick={() => {
              const newSearchParams = new URLSearchParams(window.location.search);
              newSearchParams.set('page', item.page);
              window.history.pushState(null, '', `?${newSearchParams.toString()}`);
              setPage(item.page);
            }}
          >
            <Text>{item.label}</Text>
          </Menu.Item>
          ))}
        </Menu>
      </Sider>
    </Layout>
  );
};

export default Sidebar;
