import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { useDisableShowPrompts } from "@/app/(dashboard)/hooks/useDisableShowPrompts";
import { useDisableUsageIndicator } from "@/app/(dashboard)/hooks/useDisableUsageIndicator";
import {
  emitLocalStorageChange,
  getLocalStorageItem,
  removeLocalStorageItem,
  setLocalStorageItem,
} from "@/utils/localStorageUtils";
import {
  CrownOutlined,
  DownOutlined,
  LogoutOutlined,
  MailOutlined,
  SafetyOutlined,
  UserOutlined,
} from "@ant-design/icons";
import type { MenuProps } from "antd";
import { Button, Divider, Dropdown, Space, Switch, Tag, Tooltip, Typography } from "antd";
import React, { useEffect, useState } from "react";

const { Text } = Typography;

interface UserDropdownProps {
  onLogout: () => void;
}

const UserDropdown: React.FC<UserDropdownProps> = ({ onLogout }) => {
  const { userId, userEmail, userRole, premiumUser } = useAuthorized();
  const disableShowPrompts = useDisableShowPrompts();
  const disableUsageIndicator = useDisableUsageIndicator();
  const [disableShowNewBadge, setDisableShowNewBadge] = useState(false);

  useEffect(() => {
    const storedValue = getLocalStorageItem("disableShowNewBadge");
    setDisableShowNewBadge(storedValue === "true");
  }, []);

  const userItems: MenuProps["items"] = [
    {
      key: "logout",
      label: (
        <Space>
          <LogoutOutlined />
          Logout
        </Space>
      ),
      onClick: onLogout,
    },
  ];

  const renderUserInfoSection = () => (
    <Space direction="vertical" size="small" style={{ width: "100%", padding: "12px" }}>
      <Space style={{ width: "100%", justifyContent: "space-between" }}>
        <Space>
          <MailOutlined />
          <Text type="secondary">{userEmail || "-"}</Text>
        </Space>
        {premiumUser ? (
          <Tag
            icon={<CrownOutlined />}
            color="gold"
          >
            Premium
          </Tag>
        ) : (
          <Tooltip title="Upgrade to Premium for advanced features" placement="left">
            <Tag
              icon={<CrownOutlined />}
            >
              Standard
            </Tag>
          </Tooltip>
        )}
      </Space>
      <Divider style={{ margin: "8px 0" }} />
      <Space style={{ width: "100%", justifyContent: "space-between" }}>
        <Space>
          <UserOutlined />
          <Text type="secondary">User ID</Text>
        </Space>
        <Text
          copyable
          ellipsis
          style={{ maxWidth: "150px" }}
          title={userId || "-"}
        >
          {userId || "-"}
        </Text>
      </Space>
      <Space style={{ width: "100%", justifyContent: "space-between" }}>
        <Space>
          <SafetyOutlined />
          <Text type="secondary">Role</Text>
        </Space>
        <Text>{userRole}</Text>
      </Space>
      <Divider style={{ margin: "8px 0" }} />
      <Space style={{ width: "100%", justifyContent: "space-between" }}>
        <Text type="secondary">Hide New Feature Indicators</Text>
        <Switch
          size="small"
          checked={disableShowNewBadge}
          onChange={(checked) => {
            setDisableShowNewBadge(checked);
            if (checked) {
              setLocalStorageItem("disableShowNewBadge", "true");
              emitLocalStorageChange("disableShowNewBadge");
            } else {
              removeLocalStorageItem("disableShowNewBadge");
              emitLocalStorageChange("disableShowNewBadge");
            }
          }}
          aria-label="Toggle hide new feature indicators"
        />
      </Space>
      <Space style={{ width: "100%", justifyContent: "space-between" }}>
        <Text type="secondary">Hide All Prompts</Text>
        <Switch
          size="small"
          checked={disableShowPrompts}
          onChange={(checked) => {
            if (checked) {
              setLocalStorageItem("disableShowPrompts", "true");
              emitLocalStorageChange("disableShowPrompts");
            } else {
              removeLocalStorageItem("disableShowPrompts");
              emitLocalStorageChange("disableShowPrompts");
            }
          }}
          aria-label="Toggle hide all prompts"
        />
      </Space>
      <Space style={{ width: "100%", justifyContent: "space-between" }}>
        <Text type="secondary">Hide Usage Indicator</Text>
        <Switch
          size="small"
          checked={disableUsageIndicator}
          onChange={(checked) => {
            if (checked) {
              setLocalStorageItem("disableUsageIndicator", "true");
              emitLocalStorageChange("disableUsageIndicator");
            } else {
              removeLocalStorageItem("disableUsageIndicator");
              emitLocalStorageChange("disableUsageIndicator");
            }
          }}
          aria-label="Toggle hide usage indicator"
        />
      </Space>
    </Space>
  );

  return (
    <Dropdown
      menu={{ items: userItems }}
      popupRender={(menu) => (
        <div
          className="bg-white rounded-lg shadow-lg"
        >
          {renderUserInfoSection()}
          <Divider style={{ margin: 0 }} />
          {React.cloneElement(menu as React.ReactElement, {
            style: { boxShadow: "none" },
          })}
        </div>
      )}
    >
      <Button type="text" >
        <Space>
          <UserOutlined />
          <Text>User</Text>
          <DownOutlined />
        </Space>
      </Button>
    </Dropdown>
  );
};

export default UserDropdown;
