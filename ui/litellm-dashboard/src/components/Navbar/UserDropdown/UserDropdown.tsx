import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { useDisableBlogPosts } from "@/app/(dashboard)/hooks/useDisableBlogPosts";
import { useDisableBouncingIcon } from "@/app/(dashboard)/hooks/useDisableBouncingIcon";
import { useDisableShowPrompts } from "@/app/(dashboard)/hooks/useDisableShowPrompts";
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
import { ChevronsUpDown } from "lucide-react";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { cn } from "@/lib/cva.config";
import React, { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";

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
  // "navbar" (default): compact top-right trigger. "sidebar": full-width footer
  // trigger whose menu opens upward, for the redesigned sidebar dock.
  variant?: "navbar" | "sidebar";
  // Sidebar rail mode: render the avatar only (no name/role).
  collapsed?: boolean;
}

const UserDropdown: React.FC<UserDropdownProps> = ({ onLogout, variant = "navbar", collapsed = false }) => {
  const { t } = useTranslation("common");
  const { userId, userEmail, userRoleLabel: userRole, premiumUser } = useAuthorized();
  const disableShowPrompts = useDisableShowPrompts();
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
          {t("nav.account.logout")}
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
            {t("nav.account.premium")}
          </Tag>
        ) : (
          <Tooltip title={t("nav.account.upgradeTooltip")} placement="left">
            <Tag icon={<CrownOutlined />}>{t("nav.account.standard")}</Tag>
          </Tooltip>
        )}
      </Space>
      <Divider style={{ margin: "8px 0" }} />
      <Space style={{ width: "100%", justifyContent: "space-between" }}>
        <Space>
          <UserOutlined />
          <Text type="secondary">{t("nav.account.userId")}</Text>
        </Space>
        <Text copyable ellipsis style={{ maxWidth: "150px" }} title={userId || "-"}>
          {userId || "-"}
        </Text>
      </Space>
      <Space style={{ width: "100%", justifyContent: "space-between" }}>
        <Space>
          <SafetyOutlined />
          <Text type="secondary">{t("nav.account.role")}</Text>
        </Space>
        <Text>{userRole}</Text>
      </Space>
      <Divider style={{ margin: "8px 0" }} />
      <Space style={{ width: "100%", justifyContent: "space-between" }}>
        <Text type="secondary">{t("nav.account.hideNewFeatureIndicators")}</Text>
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
          aria-label={t("nav.account.toggleNewFeatureIndicators")}
        />
      </Space>
      <Space style={{ width: "100%", justifyContent: "space-between" }}>
        <Text type="secondary">{t("nav.account.hideAllPrompts")}</Text>
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
          aria-label={t("nav.account.toggleAllPrompts")}
        />
      </Space>
      <Space style={{ width: "100%", justifyContent: "space-between" }}>
        <Text type="secondary">{t("nav.account.hideBlogPosts")}</Text>
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
          aria-label={t("nav.account.toggleBlogPosts")}
        />
      </Space>
      <Space style={{ width: "100%", justifyContent: "space-between" }}>
        <Text type="secondary">{t("nav.account.hideBouncingIcon")}</Text>
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
          aria-label={t("nav.account.toggleBouncingIcon")}
        />
      </Space>
    </Space>
  );

  const seed = userEmail || userId || "user";
  const initials = initialsFromIdentity(userEmail, userId);
  const hue = hueFromString(seed);
  const displayName = navAccountDisplayName(userEmail, userId);
  const localizedDisplayName = displayName === "Account" ? t("nav.account.fallbackName") : displayName;
  const accountMenuLabel = t("nav.account.menuAria", {
    role: userRole ?? t("nav.account.unknownRole"),
    identity: userEmail || userId || t("nav.account.unknownIdentity"),
  });

  return (
    <Dropdown
      trigger={["click"]}
      placement={variant === "sidebar" ? "topLeft" : "bottomRight"}
      menu={{ items: userItems }}
      popupRender={(menu) => (
        <div className="rounded-lg bg-white shadow-lg" data-testid="user-dropdown-panel">
          {renderUserInfoSection()}
          <Divider style={{ margin: 0 }} />
          {React.cloneElement(menu as React.ReactElement, {
            style: { boxShadow: "none" },
          })}
        </div>
      )}
    >
      {variant === "sidebar" ? (
        <button
          type="button"
          className={cn(
            "flex w-full items-center rounded-lg border border-transparent transition-colors hover:bg-sidebar-accent",
            collapsed ? "justify-center px-0 py-1" : "gap-2.5 px-2 py-1.5 text-left",
          )}
          aria-label={accountMenuLabel}
          aria-haspopup="menu"
          title={collapsed ? localizedDisplayName : undefined}
        >
          <Avatar className="size-[30px] shadow-inner ring-1 ring-black/5" aria-hidden>
            <AvatarFallback className="font-semibold text-white" style={{ backgroundColor: `hsl(${hue} 46% 38%)` }}>
              {initials}
            </AvatarFallback>
          </Avatar>
          {!collapsed && (
            <>
              <span className="min-w-0 flex-1 leading-tight">
                <span className="block truncate text-[13px] font-medium text-sidebar-foreground">
                  {localizedDisplayName}
                </span>
                {userRole && <span className="block truncate text-[11px] text-muted-foreground">{userRole}</span>}
              </span>
              <ChevronsUpDown size={16} strokeWidth={1.75} className="shrink-0 text-muted-foreground" aria-hidden />
            </>
          )}
        </button>
      ) : (
        <Button
          type="text"
          className="flex! max-w-[min(200px,34vw)] items-center gap-2 rounded-md! py-0.5! pl-1! pr-2! transition-colors hover:bg-gray-100!"
          aria-label={accountMenuLabel}
          aria-haspopup="menu"
        >
          <Avatar className="shadow-inner ring-1 ring-black/5" aria-hidden>
            <AvatarFallback className="font-semibold text-white" style={{ backgroundColor: `hsl(${hue} 46% 38%)` }}>
              {initials}
            </AvatarFallback>
          </Avatar>
          <span className="hidden min-w-0 truncate text-left text-sm font-medium leading-none text-gray-900 md:inline">
            {localizedDisplayName}
          </span>
          <DownOutlined className="hidden shrink-0 text-[10px] text-gray-400 md:inline" aria-hidden />
        </Button>
      )}
    </Dropdown>
  );
};

export default UserDropdown;
