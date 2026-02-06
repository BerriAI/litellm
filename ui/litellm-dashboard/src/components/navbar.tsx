import { useHealthReadiness } from "@/app/(dashboard)/hooks/healthReadiness/useHealthReadiness";
import { getProxyBaseUrl } from "@/components/networking";
import { useTheme } from "@/contexts/ThemeContext";
import { clearTokenCookies } from "@/utils/cookieUtils";
import { fetchProxySettings } from "@/utils/proxyUtils";
import {
  MenuFoldOutlined,
  MenuUnfoldOutlined,
  MoonOutlined,
  SunOutlined,
} from "@ant-design/icons";
import { Switch, Tag } from "antd";
import Link from "next/link";
import React, { useEffect, useState } from "react";
import { CommunityEngagementButtons } from "./Navbar/CommunityEngagementButtons/CommunityEngagementButtons";
import UserDropdown from "./Navbar/UserDropdown/UserDropdown";

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
  isDarkMode: boolean;
  toggleDarkMode: () => void;
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
  isDarkMode,
  toggleDarkMode
}) => {
  const baseUrl = getProxyBaseUrl();
  const [logoutUrl, setLogoutUrl] = useState("");
  const { logoUrl } = useTheme();
  const { data: healthData } = useHealthReadiness();
  const version = healthData?.litellm_version;

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
    window.location.href = logoutUrl;
  };

  return (
    <nav className="bg-white border-b border-gray-200 sticky top-0 z-10">
      <div className="w-full">
        <div className="flex items-center h-14 px-4">
          <div className="flex items-center flex-shrink-0">
            {onToggleSidebar && (
              <button
                onClick={onToggleSidebar}
                className="flex items-center justify-center w-10 h-10 mr-2 text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded transition-colors"
                title={sidebarCollapsed ? "Expand sidebar" : "Collapse sidebar"}
              >
                <span className="text-lg">{sidebarCollapsed ? <MenuUnfoldOutlined /> : <MenuFoldOutlined />}</span>
              </button>
            )}

            <div className="flex items-center gap-2">
              <Link href={baseUrl ? baseUrl : "/"} className="flex items-center">
                <div className="relative">
                  <div className="h-10 max-w-48 flex items-center justify-center overflow-hidden">
                    <img
                      src={imageUrl}
                      alt="LiteLLM Brand"
                      className="max-w-full max-h-full w-auto h-auto object-contain"
                    />
                  </div>
                </div>
              </Link>
              {version && (
                <div className="relative">
                  <span
                    className="absolute -top-1 -left-2 text-lg animate-bounce"
                    style={{ animationDuration: "2s" }}
                    title="Thanks for using LiteLLM!"
                  >
                    ❄️
                  </span>
                  <Tag className="relative text-xs font-medium cursor-pointer z-10">
                    <a
                      href="https://docs.litellm.ai/release_notes"
                      target="_blank"
                      rel="noopener noreferrer"
                      className="flex-shrink-0"
                    >
                      v{version}
                    </a>
                  </Tag>
                </div>
              )}
            </div>
          </div>
          {/* Right side nav items */}
          <div className="flex items-center space-x-5 ml-auto">
            <CommunityEngagementButtons />
            {/* Dark mode is currently a work in progress. To test, you can change 'false' to 'true' below.
            Do not set this to true by default until all components are confirmed to support dark mode styles. */}
            {false && <Switch
              data-testid="dark-mode-toggle"
              checked={isDarkMode}
              onChange={toggleDarkMode}
              checkedChildren={<MoonOutlined />}
              unCheckedChildren={<SunOutlined />}
            />}
            <a
              href="https://docs.litellm.ai/docs/"
              target="_blank"
              rel="noopener noreferrer"
              className="text-sm text-gray-600 hover:text-gray-900 transition-colors"
            >
              Docs
            </a>

            {!isPublicPage && (
              <UserDropdown onLogout={handleLogout} />
            )}
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;
