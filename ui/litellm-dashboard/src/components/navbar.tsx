import Link from "next/link";
import React, { useState, useEffect } from "react";
import type { MenuProps } from "antd";
import { Dropdown, Tooltip, Switch } from "antd";
import { getProxyBaseUrl } from "@/components/networking";
import {
  UserOutlined,
  LogoutOutlined,
  CrownOutlined,
  MailOutlined,
  SafetyOutlined,
  MenuFoldOutlined,
  MenuUnfoldOutlined,
} from "@ant-design/icons";
import { clearTokenCookies } from "@/utils/cookieUtils";
import { fetchProxySettings } from "@/utils/proxyUtils";
import { useTheme } from "@/contexts/ThemeContext";
import { clearMCPAuthTokens } from "./mcp_tools/mcp_auth_storage";
import useFeatureFlags from "@/hooks/useFeatureFlags";

interface NavbarProps {
  userID: string | null;
  userEmail: string | null;
  userRole: string | null;
  premiumUser: boolean;
  proxySettings: any;
  setProxySettings: React.Dispatch<React.SetStateAction<any>>;
  accessToken: string | null;
  isPublicPage: boolean;
  sidebarCollapsed?: boolean;
  onToggleSidebar?: () => void;
}

const Navbar: React.FC<NavbarProps> = ({
  userID,
  userEmail,
  userRole,
  premiumUser,
  proxySettings,
  setProxySettings,
  accessToken,
  isPublicPage = false,
  sidebarCollapsed = false,
  onToggleSidebar,
}) => {
  const baseUrl = getProxyBaseUrl();
  const [logoutUrl, setLogoutUrl] = useState("");
  const { logoUrl } = useTheme();
  const { refactoredUIFlag, setRefactoredUIFlag } = useFeatureFlags();

  // Simple logo URL: use custom logo if available, otherwise default
  const imageUrl = logoUrl || `${baseUrl}/get_image`;

  useEffect(() => {
    const initializeProxySettings = async () => {
      if (accessToken) {
        const settings = await fetchProxySettings(accessToken);
        console.log("response from fetchProxySettings", settings);
        if (settings) {
          setProxySettings(settings);
        }
      }
    };

    initializeProxySettings();
  }, [accessToken]);

  useEffect(() => {
    setLogoutUrl(proxySettings?.PROXY_LOGOUT_URL || "");
  }, [proxySettings]);

  const handleLogout = () => {
    clearTokenCookies();
    clearMCPAuthTokens(); // Clear MCP auth tokens on logout
    window.location.href = logoutUrl;
  };

  const userItems: MenuProps["items"] = [
    {
      key: "user-info",
      // Prevent dropdown from closing when interacting with the toggle
      onClick: (info) => info.domEvent?.stopPropagation(),
      label: (
        <div className="px-3 py-3 border-b border-gray-100">
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center">
              <UserOutlined className="mr-2 text-gray-700" />
              <span className="text-sm font-semibold text-gray-900">{userID}</span>
            </div>
            {premiumUser ? (
              <Tooltip title="Premium User" placement="left">
                <div className="flex items-center bg-gradient-to-r from-amber-500 to-yellow-500 text-white px-2 py-0.5 rounded-full cursor-help">
                  <CrownOutlined className="mr-1 text-xs" />
                  <span className="text-xs font-medium">Premium</span>
                </div>
              </Tooltip>
            ) : (
              <Tooltip title="Upgrade to Premium for advanced features" placement="left">
                <div className="flex items-center bg-gray-100 text-gray-500 px-2 py-0.5 rounded-full cursor-help">
                  <CrownOutlined className="mr-1 text-xs" />
                  <span className="text-xs font-medium">Standard</span>
                </div>
              </Tooltip>
            )}
          </div>
          <div className="space-y-2">
            <div className="flex items-center text-sm">
              <SafetyOutlined className="mr-2 text-gray-400 text-xs" />
              <span className="text-gray-500 text-xs">Role</span>
              <span className="ml-auto text-gray-700 font-medium">{userRole}</span>
            </div>
            <div className="flex items-center text-sm">
              <MailOutlined className="mr-2 text-gray-400 text-xs" />
              <span className="text-gray-500 text-xs">Email</span>
              <span className="ml-auto text-gray-700 font-medium truncate max-w-[150px]" title={userEmail || "Unknown"}>
                {userEmail || "Unknown"}
              </span>
            </div>

            {/* NEW: Feature flag label + toggle below the email field */}
            <div className="flex items-center text-sm pt-2 mt-2 border-t border-gray-100">
              <span className="text-gray-500 text-xs">Refactored UI</span>
              <Switch
                className="ml-auto"
                size="small"
                checked={refactoredUIFlag}
                onChange={(checked) => setRefactoredUIFlag(checked)}
                aria-label="Toggle refactored UI feature flag"
              />
            </div>
          </div>
        </div>
      ),
    },
    {
      key: "logout",
      label: (
        <div className="flex items-center py-2 px-3 hover:bg-gray-50 rounded-md mx-1 my-1" onClick={handleLogout}>
          <LogoutOutlined className="mr-3 text-gray-600" />
          <span className="text-gray-800">Logout</span>
        </div>
      ),
    },
  ];

  return (
    <nav className="bg-white border-b border-gray-200 sticky top-0 z-10">
      <div className="w-full">
        <div className="flex items-center h-14 px-4">
          {" "}
          {/* Increased height from h-12 to h-14 */}
          {/* Left side with collapse toggle and logo */}
          <div className="flex items-center flex-shrink-0">
            {/* Collapse/Expand Toggle Button - Larger */}
            {onToggleSidebar && (
              <button
                onClick={onToggleSidebar}
                className="flex items-center justify-center w-10 h-10 mr-2 text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded transition-colors"
                title={sidebarCollapsed ? "Expand sidebar" : "Collapse sidebar"}
              >
                <span className="text-lg">{sidebarCollapsed ? <MenuUnfoldOutlined /> : <MenuFoldOutlined />}</span>
              </button>
            )}

            <Link href="/" className="flex items-center">
              <img src={imageUrl} alt="LiteLLM Brand" className="h-10 w-auto" />
            </Link>
          </div>
          {/* Right side nav items */}
          <div className="flex items-center space-x-5 ml-auto">
            <a
              href="https://docs.litellm.ai/docs/"
              target="_blank"
              rel="noopener noreferrer"
              className="text-sm text-gray-600 hover:text-gray-900 transition-colors"
            >
              Docs
            </a>

            {!isPublicPage && (
              <Dropdown
                menu={{
                  items: userItems,
                  className: "min-w-[200px]",
                  style: {
                    padding: "8px",
                    marginTop: "8px",
                    borderRadius: "12px",
                    boxShadow: "0 4px 24px rgba(0, 0, 0, 0.08)",
                  },
                }}
                overlayStyle={{
                  minWidth: "200px",
                }}
              >
                <button className="inline-flex items-center text-sm text-gray-600 hover:text-gray-900 transition-colors">
                  User
                  <svg className="ml-1 w-5 h-5 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M19 9l-7 7-7-7" />
                  </svg>
                </button>
              </Dropdown>
            )}
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;
