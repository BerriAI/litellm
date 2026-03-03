import { useProxyConnection } from "@/contexts/ProxyConnectionContext";
import { CloudServerOutlined, CheckCircleFilled, DownOutlined, SettingOutlined } from "@ant-design/icons";
import type { MenuProps } from "antd";
import { Button, Divider, Dropdown, Space, Tag, Typography } from "antd";
import React from "react";

const { Text } = Typography;

interface ProxySwitcherProps {
  onManageClick: () => void;
}

const ProxySwitcher: React.FC<ProxySwitcherProps> = ({ onManageClick }) => {
  const { connections, activeConnection, switchConnection } = useProxyConnection();

  // Only show when there are multiple connections configured
  if (connections.length <= 1) return null;

  const items: MenuProps["items"] = [
    ...connections.map((conn) => ({
      key: conn.id,
      label: (
        <Space>
          {conn.id === activeConnection?.id && <CheckCircleFilled style={{ color: "#52c41a" }} />}
          <span>{conn.name}</span>
          {conn.isDefault && <Tag color="blue">Local</Tag>}
        </Space>
      ),
      onClick: () => {
        if (conn.id !== activeConnection?.id) {
          switchConnection(conn.id);
        }
      },
    })),
    { type: "divider" as const },
    {
      key: "manage",
      icon: <SettingOutlined />,
      label: "Manage Connections",
      onClick: onManageClick,
    },
  ];

  return (
    <Dropdown menu={{ items }} trigger={["click"]}>
      <Button type="text" size="small">
        <Space>
          <CloudServerOutlined />
          <Text ellipsis style={{ maxWidth: 150 }}>
            {activeConnection?.name || "Default"}
          </Text>
          {activeConnection && !activeConnection.isDefault && <Tag color="orange">Remote</Tag>}
          <DownOutlined style={{ fontSize: 10 }} />
        </Space>
      </Button>
    </Dropdown>
  );
};

export default ProxySwitcher;
