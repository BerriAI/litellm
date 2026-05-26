import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { useDisableBlogPosts } from "@/app/(dashboard)/hooks/useDisableBlogPosts";
import { useDisableBouncingIcon } from "@/app/(dashboard)/hooks/useDisableBouncingIcon";
import { useDisableShowPrompts } from "@/app/(dashboard)/hooks/useDisableShowPrompts";
import { useDisableUsageIndicator } from "@/app/(dashboard)/hooks/useDisableUsageIndicator";
import {
  emitLocalStorageChange,
  getLocalStorageItem,
  removeLocalStorageItem,
  setLocalStorageItem,
} from "@/utils/localStorageUtils";
import { navAccountDisplayName } from "@/components/Navbar/navDisplayName";
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

function hueFromString(seed: string): number {
  let h = 0;
  for (let i = 0; i < seed.length; i += 1) {
    h = seed.charCodeAt(i) + ((h << 5) - h);
  }
  return Math.abs(h) % 360;
}

function initialsFromIdentity(email: string | null, userId: string | null): string {
  const local = email?.split("@")[0]?.trim();
  if (local) {
    const parts = local
      .replace(/[^a-zA-Z0-9]+/g, " ")
      .trim()
      .split(/\s+/)
      .filter(Boolean);
    if (parts.length >= 2) {
      return `${parts[0]!.charAt(0)}${parts[1]!.charAt(0)}`.toUpperCase();
    }
    if (parts.length === 1) {
      const p = parts[0]!;
      return p.length >= 2 ? p.slice(0, 2).toUpperCase() : `${p.charAt(0)}`.toUpperCase();
    }
  }
  if (userId && userId.length >= 2) {
    return userId.slice(0, 2).toUpperCase();
  }
  if (userId && userId.length === 1) {
    return `${userId.toUpperCase()}•`;
  }
  return "?";
}

interface UserDropdownProps {
  onLogout: () => void;
}

const UserDropdown: React.FC<UserDropdownProps> = ({ onLogout }) => {
  const { userId, userEmail, userRole, premiumUser } = useAuthorized();
  const disableShowPrompts = useDisableShowPrompts();
  const disableUsageIndicator = useDisableUsageIndicator();
  const disableBlogPosts = useDisableBlogPosts();
  const disableBouncingIcon = useDisableBouncingIcon();
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
          <Tag icon={<CrownOutlined />} color="gold">
            Premium
          </Tag>
        ) : (
          <Tooltip title="Upgrade to Premium for advanced features" placement="left">
            <Tag icon={<CrownOutlined />}>Standard</Tag>
          </Tooltip>
        )}
      </Space>
      <Divider style={{ margin: "8px 0" }} />
      <Space style={{ width: "100%", justifyContent: "space-between" }}>
        <Space>
          <UserOutlined />
          <Text type="secondary">User ID</Text>
        </Space>
        <Text copyable ellipsis style={{ maxWidth: "150px" }} title={userId || "-"}>
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
      <Space style={{ width: "100%", justifyContent: "space-between" }}>
        <Text type="secondary">Hide Blog Posts</Text>
        <Switch
          size="small"
          checked={disableBlogPosts}
          onChange={(checked) => {
            if (checked) {
              setLocalStorageItem("disableBlogPosts", "true");
              emitLocalStorageChange("disableBlogPosts");
            } else {
              removeLocalStorageItem("disableBlogPosts");
              emitLocalStorageChange("disableBlogPosts");
            }
          }}
          aria-label="Toggle hide blog posts"
        />
      </Space>
      <Space style={{ width: "100%", justifyContent: "space-between" }}>
        <Text type="secondary">Hide Bouncing Icon</Text>
        <Switch
          size="small"
          checked={disableBouncingIcon}
          onChange={(checked) => {
            if (checked) {
              setLocalStorageItem("disableBouncingIcon", "true");
              emitLocalStorageChange("disableBouncingIcon");
            } else {
              removeLocalStorageItem("disableBouncingIcon");
              emitLocalStorageChange("disableBouncingIcon");
            }
          }}
          aria-label="Toggle hide bouncing icon"
        />
      </Space>
    </Space>
  );

  const seed = userEmail || userId || "user";
  const initials = initialsFromIdentity(userEmail, userId);
  const hue = hueFromString(seed);
  const displayName = navAccountDisplayName(userEmail, userId);

  return (
    <Dropdown
      trigger={["click"]}
      menu={{ items: userItems }}
      popupRender={(menu) => (
        <div className="rounded-lg bg-white shadow-lg">
          {renderUserInfoSection()}
          <Divider style={{ margin: 0 }} />
          {React.cloneElement(menu as React.ReactElement, {
            style: { boxShadow: "none" },
          })}
        </div>
      )}
    >
      <Button
        type="text"
        className="!flex max-w-[min(200px,34vw)] items-center gap-2 !rounded-md !py-0.5 !pl-1 !pr-2 transition-colors hover:!bg-gray-100"
        aria-label={`Account menu — ${userRole ?? "Unknown role"} — signed in as ${userEmail || userId || "unknown"}`}
        aria-haspopup="menu"
      >
        <span
          className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full text-xs font-semibold text-white shadow-inner ring-1 ring-black/5"
          style={{ backgroundColor: `hsl(${hue} 46% 38%)` }}
          aria-hidden
        >
          {initials}
        </span>
        <span className="hidden min-w-0 truncate text-left text-sm font-medium leading-none text-gray-900 md:inline">
          {displayName}
        </span>
        <DownOutlined className="hidden shrink-0 text-[10px] text-gray-400 md:inline" aria-hidden />
      </Button>
    </Dropdown>
  );
};

export default UserDropdown;
