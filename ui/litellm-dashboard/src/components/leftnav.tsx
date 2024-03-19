import { Layout, Menu } from "antd";
import Link from "next/link";
import { List } from "postcss/lib/list";

const { Sider } = Layout;

// Define the props type
interface SidebarProps {
  setPage: React.Dispatch<React.SetStateAction<string>>;
  userRole: string;
  defaultSelectedKey: string[] | null;
}

const Sidebar: React.FC<SidebarProps> = ({
  setPage,
  userRole,
  defaultSelectedKey,
}) => {
  if (userRole == "Admin Viewer") {
    return (
      <Layout style={{ minHeight: "100vh", maxWidth: "120px" }}>
        <Sider width={120}>
          <Menu
            mode="inline"
            defaultSelectedKeys={
              defaultSelectedKey ? defaultSelectedKey : ["4"]
            }
            style={{ height: "100%", borderRight: 0 }}
          >
            <Menu.Item key="4" onClick={() => setPage("api-keys")}>
              API Keys
            </Menu.Item>
            <Menu.Item key="2" onClick={() => setPage("models")}>
              Models
            </Menu.Item>
            <Menu.Item key="3" onClick={() => setPage("llm-playground")}>
              Chat UI
            </Menu.Item>
            <Menu.Item key="1" onClick={() => setPage("usage")}>
              Usage
            </Menu.Item>
          </Menu>
        </Sider>
      </Layout>
    );
  }
  return (
    <Layout style={{ minHeight: "100vh", maxWidth: "120px" }}>
      <Sider width={120}>
        <Menu
          mode="inline"
          defaultSelectedKeys={defaultSelectedKey ? defaultSelectedKey : ["1"]}
          style={{ height: "100%", borderRight: 0 }}
        >
          <Menu.Item key="1" onClick={() => setPage("api-keys")}>
            API Keys
          </Menu.Item>
          <Menu.Item key="2" onClick={() => setPage("models")}>
            Models
          </Menu.Item>
          <Menu.Item key="3" onClick={() => setPage("llm-playground")}>
            Chat UI
          </Menu.Item>
          <Menu.Item key="4" onClick={() => setPage("usage")}>
            Usage
          </Menu.Item>
          {userRole == "Admin" ? (
            <Menu.Item key="5" onClick={() => setPage("users")}>
              Users
            </Menu.Item>
          ) : null}
          {userRole == "Admin" ? (
            <Menu.Item key="6" onClick={() => setPage("teams")}>
              Teams
            </Menu.Item>
          ) : null}
          {userRole == "Admin" ? (
            <Menu.Item key="7" onClick={() => setPage("admin-panel")}>
              Admin
            </Menu.Item>
          ) : null}
        </Menu>
      </Sider>
    </Layout>
  );
};

export default Sidebar;
