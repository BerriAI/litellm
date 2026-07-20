import { useHealthReadinessDetails } from "@/app/(dashboard)/hooks/healthReadiness/useHealthReadinessDetails";
import { useDisableBouncingIcon } from "@/app/(dashboard)/hooks/useDisableBouncingIcon";
import { useDisableShowPrompts } from "@/app/(dashboard)/hooks/useDisableShowPrompts";
import { useWorker } from "@/hooks/useWorker";
import { getProxyBaseUrl } from "@/components/networking";
import { useTheme } from "@/contexts/ThemeContext";
import { clearTokenCookies } from "@/utils/cookieUtils";
import { clearStoredReturnUrl, getLoginUrl } from "@/utils/returnUrlUtils";
import useProxySettings from "@/app/(dashboard)/hooks/proxySettings/useProxySettings";
import { DownOutlined, MenuFoldOutlined, MenuUnfoldOutlined } from "@ant-design/icons";
import { Tag } from "antd";
import Link from "next/link";
import React from "react";
import { BlogDropdown } from "./Navbar/BlogDropdown/BlogDropdown";
import { CommunityEngagementButtons } from "./Navbar/CommunityEngagementButtons/CommunityEngagementButtons";
import { NAV_PRODUCT_LINK_CLASS } from "./Navbar/navProductLinkClass";
import { NotificationsBell } from "./Navbar/NotificationsBell/NotificationsBell";
import UserDropdown from "./Navbar/UserDropdown/UserDropdown";
import ViewSwitcher from "./Navbar/ViewSwitcher";
import WorkerDropdown from "./Navbar/WorkerDropdown/WorkerDropdown";

interface NavbarProps {
  accessToken: string | null;
  isPublicPage: boolean;
  sidebarCollapsed?: boolean;
  onToggleSidebar?: () => void;
}

const Navbar: React.FC<NavbarProps> = ({
  accessToken,
  isPublicPage = false,
  sidebarCollapsed = false,
  onToggleSidebar,
}) => {
  const baseUrl = getProxyBaseUrl();
  const proxySettings = useProxySettings(accessToken);
  const { logoUrl } = useTheme();
  const { data: healthData } = useHealthReadinessDetails(accessToken);
  const version = healthData?.litellm_version;
  const disableBouncingIcon = useDisableBouncingIcon();
  const hideCommunityLinks = useDisableShowPrompts();
  const { isControlPlane, selectedWorker } = useWorker();
  const showWorkerSwitch = isControlPlane && selectedWorker !== null;

  const imageUrl = logoUrl || `${baseUrl}/get_image`;

  const handleLogout = () => {
    clearTokenCookies();
    localStorage.removeItem("litellm_selected_worker_id");
    localStorage.removeItem("litellm_worker_url");
    window.location.href = proxySettings.PROXY_LOGOUT_URL || "";
  };

  const handleWorkerSwitch = (workerId: string) => {
    clearTokenCookies();
    clearStoredReturnUrl();
    localStorage.removeItem("litellm_selected_worker_id");
    localStorage.removeItem("litellm_worker_url");
    window.location.href = `${getLoginUrl()}?worker=${encodeURIComponent(workerId)}`;
  };

  return (
    <nav className="sticky top-0 z-10 border-b border-gray-200 bg-white">
      <div className="w-full">
        <div className="flex h-14 items-center px-4">
          <div className="flex shrink-0 items-center">
            {onToggleSidebar && (
              <button
                onClick={onToggleSidebar}
                className="mr-2 flex h-9 w-9 items-center justify-center rounded-md text-gray-600 transition-colors hover:bg-gray-100 hover:text-gray-900"
                title={sidebarCollapsed ? "Expand sidebar" : "Collapse sidebar"}
              >
                <span className="text-lg">{sidebarCollapsed ? <MenuUnfoldOutlined /> : <MenuFoldOutlined />}</span>
              </button>
            )}

            <div className="flex items-center gap-2">
              <Link href={baseUrl ? baseUrl : "/"} className="flex items-center">
                <div className="relative">
                  <div className="flex h-10 max-w-48 items-center justify-center overflow-hidden">
                    <img
                      src={imageUrl}
                      alt="LiteLLM Brand"
                      className="h-auto max-h-full w-auto max-w-full object-contain"
                    />
                  </div>
                </div>
              </Link>
              {version && (
                <div className="relative">
                  {!disableBouncingIcon && (
                    <span
                      className="absolute -left-2 -top-1 animate-bounce text-lg"
                      style={{ animationDuration: "2s" }}
                      title="Thanks for using LiteLLM!"
                    >
                      🌑
                    </span>
                  )}
                  <Tag className="relative z-10 cursor-pointer text-xs font-medium">
                    <a
                      href="https://docs.litellm.ai/release_notes"
                      target="_blank"
                      rel="noopener noreferrer"
                      className="shrink-0"
                    >
                      v{version}
                    </a>
                  </Tag>
                </div>
              )}
            </div>
          </div>

          {!isPublicPage && (
            <div className="ml-4 flex shrink-0 items-center border-l border-gray-200 pl-4">
              <ViewSwitcher />
            </div>
          )}

          <div className="ml-auto flex min-w-0 flex-1 items-center justify-end gap-4">
            {showWorkerSwitch && (
              <div className="flex shrink-0 items-center">
                <WorkerDropdown onWorkerSwitch={handleWorkerSwitch} />
              </div>
            )}

            <nav
              aria-label="Product documentation"
              className={`flex min-w-0 items-center gap-2 ${showWorkerSwitch ? "border-l border-gray-200 pl-4" : ""}`}
            >
              <a
                href="https://docs.litellm.ai/docs/"
                target="_blank"
                rel="noopener noreferrer"
                className={NAV_PRODUCT_LINK_CLASS}
              >
                Docs
                {/* Layout parity with Blog chevron — intentional single-level link */}
                <DownOutlined className="pointer-events-none text-[10px] opacity-0" aria-hidden />
              </a>
              <BlogDropdown />
            </nav>

            {!hideCommunityLinks && (
              <div className="flex shrink-0 items-center border-l border-gray-200 pl-4">
                <CommunityEngagementButtons />
              </div>
            )}

            {!isPublicPage && (
              <div className="flex shrink-0 items-center border-l border-gray-200 pl-4">
                <div className="flex items-center gap-0.5 rounded-lg bg-gray-50 px-1 py-0 transition-colors hover:bg-gray-100">
                  <NotificationsBell />
                  <span className="mx-0.5 h-6 w-px shrink-0 bg-gray-200" aria-hidden />
                  <UserDropdown onLogout={handleLogout} />
                </div>
              </div>
            )}
          </div>
          {/* Dark mode toggle: keep disabled until the dashboard supports dark styles end-to-end. */}
        </div>
      </div>
    </nav>
  );
};

export default Navbar;
