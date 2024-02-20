import { Layout, Menu } from "antd";
import Link from "next/link";

const { Sider } = Layout;

// Define the props type
interface SidebarProps {
  setPage: React.Dispatch<React.SetStateAction<string>>;
  userRole: string;
}

const Sidebar: React.FC<SidebarProps> = ({ setPage, userRole }) => {
  return (
    <Layout style={{ minHeight: "100vh", maxWidth: "120px" }}>
      <Sider width={120}>
        <Menu
          mode="inline"
          defaultSelectedKeys={["1"]}
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
          {
            userRole == "Admin" ? 
              <Menu.Item key="5" onClick={() => setPage("users")}>
                Users
              </Menu.Item>
            : null
          }
        </Menu>
      </Sider>
    </Layout>
  );
};

export default Sidebar;
