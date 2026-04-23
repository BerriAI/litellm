import { useHealthReadiness } from "@/app/(dashboard)/hooks/healthReadiness/useHealthReadiness";
import { useDisableBouncingIcon } from "@/app/(dashboard)/hooks/useDisableBouncingIcon";
import { getProxyBaseUrl } from "@/components/networking";
import { useTheme } from "@/contexts/ThemeContext";
import { clearTokenCookies } from "@/utils/cookieUtils";
import { clearStoredReturnUrl } from "@/utils/returnUrlUtils";
import { fetchProxySettings } from "@/utils/proxyUtils";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import { Menu, Moon, PanelLeftClose, Sun } from "lucide-react";
import Link from "next/link";
import React, { useEffect, useState } from "react";
import { BlogDropdown } from "./Navbar/BlogDropdown/BlogDropdown";
import { CommunityEngagementButtons } from "./Navbar/CommunityEngagementButtons/CommunityEngagementButtons";
import UserDropdown from "./Navbar/UserDropdown/UserDropdown";
import WorkerDropdown from "./Navbar/WorkerDropdown/WorkerDropdown";

interface NavbarProps {
  userID: string | null;
  userEmail: string | null;
  userRole: string | null;
  premiumUser: boolean;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  proxySettings: any;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  setProxySettings: React.Dispatch<React.SetStateAction<any>>;
  accessToken: string | null;
  isPublicPage: boolean;
  sidebarCollapsed?: boolean;
  onToggleSidebar?: () => void;
  isDarkMode: boolean;
  toggleDarkMode: () => void;
}

const Navbar: React.FC<NavbarProps> = ({
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  userID,
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  userEmail,
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  userRole,
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  premiumUser,
  proxySettings,
  setProxySettings,
  accessToken,
  isPublicPage = false,
  sidebarCollapsed = false,
  onToggleSidebar,
  isDarkMode,
  toggleDarkMode,
}) => {
  const baseUrl = getProxyBaseUrl();
  const [logoutUrl, setLogoutUrl] = useState("");
  const { logoUrl } = useTheme();
  const { data: healthData } = useHealthReadiness();
  const version = healthData?.litellm_version;
  const disableBouncingIcon = useDisableBouncingIcon();

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
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [accessToken]);

  useEffect(() => {
    setLogoutUrl(proxySettings?.PROXY_LOGOUT_URL || "");
  }, [proxySettings]);

  const handleLogout = () => {
    clearTokenCookies();
    localStorage.removeItem("litellm_selected_worker_id");
    localStorage.removeItem("litellm_worker_url");
    window.location.href = logoutUrl;
  };

  const handleWorkerSwitch = (workerId: string) => {
    clearTokenCookies();
    clearStoredReturnUrl();
    localStorage.removeItem("litellm_selected_worker_id");
    localStorage.removeItem("litellm_worker_url");
    window.location.href = `/ui/login?worker=${encodeURIComponent(workerId)}`;
  };

  return (
    <nav className="bg-background border-b border-border sticky top-0 z-10">
      <div className="w-full">
        <div className="flex items-center h-14 px-4">
          <div className="flex items-center flex-shrink-0">
            {onToggleSidebar && (
              <button
                onClick={onToggleSidebar}
                className="flex items-center justify-center w-10 h-10 mr-2 text-muted-foreground hover:text-foreground hover:bg-muted rounded transition-colors"
                title={sidebarCollapsed ? "Expand sidebar" : "Collapse sidebar"}
                aria-label={
                  sidebarCollapsed ? "Expand sidebar" : "Collapse sidebar"
                }
              >
                {sidebarCollapsed ? (
                  <Menu className="h-5 w-5" />
                ) : (
                  <PanelLeftClose className="h-5 w-5" />
                )}
              </button>
            )}

            <div className="flex items-center gap-2">
              <Link
                href={baseUrl ? baseUrl : "/"}
                className="flex items-center"
              >
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
                  {!disableBouncingIcon && (
                    <span
                      className="absolute -top-1 -left-2 text-lg animate-bounce"
                      style={{ animationDuration: "2s" }}
                      title="Thanks for using LiteLLM!"
                    >
                      🌑
                    </span>
                  )}
                  <Badge
                    variant="outline"
                    className="relative text-xs font-medium cursor-pointer z-10"
                  >
                    <a
                      href="https://docs.litellm.ai/release_notes"
                      target="_blank"
                      rel="noopener noreferrer"
                      className="flex-shrink-0"
                    >
                      v{version}
                    </a>
                  </Badge>
                </div>
              )}
            </div>
          </div>
          <div className="flex items-center space-x-5 ml-auto">
            <WorkerDropdown onWorkerSwitch={handleWorkerSwitch} />
            <CommunityEngagementButtons />
            {/* Dark mode is currently a work in progress. */}
            {false && (
              <div className="flex items-center gap-1">
                <Sun className="h-3.5 w-3.5" />
                <Switch
                  data-testid="dark-mode-toggle"
                  checked={isDarkMode}
                  onCheckedChange={toggleDarkMode}
                />
                <Moon className="h-3.5 w-3.5" />
              </div>
            )}
            <Button variant="ghost" asChild>
              <a
                href="https://docs.litellm.ai/docs/"
                target="_blank"
                rel="noopener noreferrer"
              >
                Docs
              </a>
            </Button>
            <BlogDropdown />

            {!isPublicPage && <UserDropdown onLogout={handleLogout} />}
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;
