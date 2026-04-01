"use client";

import APIReferenceView from "@/app/(dashboard)/api-reference/APIReferenceView";
import SidebarProvider from "@/app/(dashboard)/components/SidebarProvider";
import OldModelDashboard from "@/app/(dashboard)/models-and-endpoints/ModelsAndEndpointsView";
import PlaygroundPage from "@/app/(dashboard)/playground/page";
import AdminPanel from "@/components/AdminPanel";
import AgentsPanel from "@/components/agents";
import BudgetPanel from "@/components/budgets/budget_panel";
import CacheDashboard from "@/components/cache_dashboard";
import ClaudeCodePluginsPanel from "@/components/claude_code_plugins";
import { fetchTeams } from "@/components/common_components/fetch_teams";
import LoadingScreen from "@/components/common_components/LoadingScreen";
import { CostTrackingSettings } from "@/components/CostTrackingSettings";
import GeneralSettings from "@/components/general_settings";
import GuardrailsPanel from "@/components/guardrails";
import PoliciesPanel from "@/components/policies";
import { Team } from "@/components/key_team_helpers/key_list";
import { MCPServers } from "@/components/mcp_tools";
import ModelHubTable from "@/components/AIHub/ModelHubTable";
import Navbar from "@/components/navbar";
import { getUiConfig, Organization, proxyBaseUrl, setGlobalLitellmHeaderName, getInProductNudgesCall } from "@/components/networking";
import NewUsagePage from "@/components/UsagePage/components/UsagePageView";
import OldTeams from "@/components/OldTeams";
import { fetchUserModels } from "@/components/organisms/create_key_button";
import Organizations, { fetchOrganizations } from "@/components/organizations";
import PassThroughSettings from "@/components/pass_through_settings";
import PromptsPanel from "@/components/prompts";
import PublicModelHub from "@/components/public_model_hub";
import { SearchTools } from "@/components/SearchTools";
import Settings from "@/components/settings";
import { SurveyPrompt, SurveyModal, ClaudeCodePrompt, ClaudeCodeModal } from "@/components/survey";
import TagManagement from "@/components/tag_management";
import TransformRequestPanel from "@/components/transform_request";
import UIThemeSettings from "@/components/ui_theme_settings";
import Usage from "@/components/usage";
import UserDashboard from "@/components/user_dashboard";
import { AccessGroupsPage } from "@/components/AccessGroups/AccessGroupsPage";
import VectorStoreManagement from "@/components/vector_store_management";
import SpendLogsTable from "@/components/view_logs";
import ViewUserDashboard from "@/components/view_users";
import { ThemeProvider } from "@/contexts/ThemeContext";
import { isJwtExpired } from "@/utils/jwtUtils";
import { isAdminRole } from "@/utils/roles";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { jwtDecode } from "jwt-decode";
import { useSearchParams } from "next/navigation";
import { Suspense, useEffect, useState } from "react";
import { ConfigProvider, theme } from "antd";

function getCookie(name: string) {
  // Safer cookie read + decoding; handles '=' inside values
  const match = document.cookie.split("; ").find((row) => row.startsWith(name + "="));
  if (!match) return null;
  const value = match.slice(name.length + 1);
  try {
    return decodeURIComponent(value);
  } catch {
    return value;
  }
}

function deleteCookie(name: string, path = "/") {
  // Best-effort client-side clear (works for non-HttpOnly cookies without Domain)
  document.cookie = `${name}=; Max-Age=0; Path=${path}`;
}

function formatUserRole(userRole: string) {
  if (!userRole) {
    return "Undefined Role";
  }
  switch (userRole.toLowerCase()) {
    case "app_owner":
      return "App Owner";
    case "demo_app_owner":
      return "App Owner";
    case "app_admin":
      return "Admin";
    case "proxy_admin":
      return "Admin";
    case "proxy_admin_viewer":
      return "Admin Viewer";
    case "org_admin":
      return "Org Admin";
    case "internal_user":
      return "Internal User";
    case "internal_user_viewer":
    case "internal_viewer": // TODO:remove if deprecated
      return "Internal Viewer";
    case "app_user":
      return "App User";
    default:
      return "Unknown Role";
  }
}

interface ProxySettings {
  PROXY_BASE_URL: string;
  PROXY_LOGOUT_URL: string;
  LITELLM_UI_API_DOC_BASE_URL?: string | null;
}

const queryClient = new QueryClient();

function CreateKeyPageContent() {
  const [userRole, setUserRole] = useState("");
  const [premiumUser, setPremiumUser] = useState(false);
  const [disabledPersonalKeyCreation, setDisabledPersonalKeyCreation] = useState(false);
  const [userEmail, setUserEmail] = useState<null | string>(null);
  const [teams, setTeams] = useState<Team[] | null>(null);
  const [keys, setKeys] = useState<null | any[]>([]);
  const [organizations, setOrganizations] = useState<Organization[]>([]);
  const [userModels, setUserModels] = useState<string[]>([]);
  const [proxySettings, setProxySettings] = useState<ProxySettings>({
    PROXY_BASE_URL: "",
    PROXY_LOGOUT_URL: "",
  });

  const [showSSOBanner, setShowSSOBanner] = useState<boolean>(true);
  const searchParams = useSearchParams()!;
  const [modelData, setModelData] = useState<any>({ data: [] });
  const [token, setToken] = useState<string | null>(null);
  const [createClicked, setCreateClicked] = useState<boolean>(false);
  const [authLoading, setAuthLoading] = useState(true);
  const [userID, setUserID] = useState<string | null>(null);

  // Survey state - always show by default
  const [showSurveyPrompt, setShowSurveyPrompt] = useState(true);
  const [showSurveyModal, setShowSurveyModal] = useState(false);

  // Claude Code feedback state
  const [isClaudeCode, setIsClaudeCode] = useState(false);
  const [showClaudeCodePrompt, setShowClaudeCodePrompt] = useState(false);
  const [showClaudeCodeModal, setShowClaudeCodeModal] = useState(false);

  // Dark mode state
  const [isDarkMode, setIsDarkMode] = useState(false);
  const toggleDarkMode = () => {
    setIsDarkMode(!isDarkMode);
  };

  const invitation_id = searchParams.get("invitation_id");

  // Get page from URL, default to 'api-keys' if not present
  const [page, setPage] = useState(() => {
    return searchParams.get("page") || "api-keys";
  });

  // Custom setPage function that updates URL
  const updatePage = (newPage: string) => {
    // Update URL without full page reload
    const newSearchParams = new URLSearchParams(searchParams);
    newSearchParams.set("page", newPage);

    // Use Next.js router to update URL
    window.history.pushState(null, "", `?${newSearchParams.toString()}`);

    setPage(newPage);
  };

  const [accessToken, setAccessToken] = useState<string | null>(null);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);

  const toggleSidebar = () => {
    setSidebarCollapsed(!sidebarCollapsed);
  };

  const addKey = (data: any) => {
    setKeys((prevData) => (prevData ? [...prevData, data] : [data]));
    setCreateClicked(() => !createClicked);
  };
  const redirectToLogin = authLoading === false && token === null && invitation_id === null;

  useEffect(() => {
    let cancelled = false;

    (async () => {
      try {
        await getUiConfig(); // ensures proxyBaseUrl etc. are ready
      } catch {
        // proceed regardless; we still need to decide auth state
      }

      if (cancelled) return;

      const raw = getCookie("token");
      const valid = raw && !isJwtExpired(raw) ? raw : null;

      // If token exists but is invalid/expired, clear it so downstream code
      // doesn't keep trying to use it and cause redirect spasms.
      if (raw && !valid) {
        deleteCookie("token", "/");
      }

      if (!cancelled) {
        setToken(valid);
        setAuthLoading(false);
      }
    })();

    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (redirectToLogin) {
      // Replace instead of assigning to avoid back-button loops
      const dest = (proxyBaseUrl || "") + "/ui/login";
      window.location.replace(dest);
    }
  }, [redirectToLogin]);

  useEffect(() => {
    if (!token) {
      return;
    }

    // Defensive: re-check expiry in case cookie changed after mount
    if (isJwtExpired(token)) {
      deleteCookie("token", "/");
      setToken(null);
      return;
    }

    let decoded: any = null;
    try {
      decoded = jwtDecode(token);
    } catch {
      // Malformed token → treat as unauthenticated
      deleteCookie("token", "/");
      setToken(null);
      return;
    }

    if (decoded) {
      // set accessToken
      setAccessToken(decoded.key);

      setDisabledPersonalKeyCreation(decoded.disabled_non_admin_personal_key_creation);

      // check if userRole is defined
      if (decoded.user_role) {
        const formattedUserRole = formatUserRole(decoded.user_role);
        setUserRole(formattedUserRole);
        if (formattedUserRole == "Admin Viewer") {
          setPage("usage");
        }
      }

      if (decoded.user_email) {
        setUserEmail(decoded.user_email);
      }

      if (decoded.login_method) {
        setShowSSOBanner(decoded.login_method == "username_password" ? true : false);
      }

      if (decoded.premium_user) {
        setPremiumUser(decoded.premium_user);
      }

      if (decoded.auth_header_name) {
        setGlobalLitellmHeaderName(decoded.auth_header_name);
      }

      if (decoded.user_id) {
        setUserID(decoded.user_id);
      }
    }
  }, [token]);

  useEffect(() => {
    if (accessToken && userID && userRole) {
      fetchUserModels(userID, userRole, accessToken, setUserModels);
    }
    if (accessToken && userID && userRole) {
      fetchTeams(accessToken, userID, userRole, null, setTeams);
    }
    if (accessToken) {
      fetchOrganizations(accessToken, setOrganizations);
    }
  }, [accessToken, userID, userRole]);

  // Fetch in-product nudges configuration from backend
  useEffect(() => {
    if (accessToken && token) {
      (async () => {
        try {
          const nudgesConfig = await getInProductNudgesCall(accessToken);
          const isUsingClaudeCode = nudgesConfig?.is_claude_code_enabled || false;
          setIsClaudeCode(isUsingClaudeCode);

          // Show Claude Code prompt on login if enabled
          if (isUsingClaudeCode) {
            setShowClaudeCodePrompt(true);
            // Don't show the regular survey prompt if showing Claude Code prompt
            setShowSurveyPrompt(false);
          }
        } catch (error) {
          console.error("Failed to fetch in-product nudges:", error);
          // Silently fail and don't show Claude Code nudge
        }
      })();
    }
  }, [accessToken, token]);

  // Auto-dismiss survey prompt after 15 seconds
  useEffect(() => {
    if (showSurveyPrompt && !showSurveyModal) {
      const timer = setTimeout(() => {
        setShowSurveyPrompt(false);
      }, 15000);
      return () => clearTimeout(timer);
    }
  }, [showSurveyPrompt, showSurveyModal]);

  // Auto-dismiss Claude Code prompt after 15 seconds
  useEffect(() => {
    if (showClaudeCodePrompt && !showClaudeCodeModal) {
      const timer = setTimeout(() => {
        setShowClaudeCodePrompt(false);
      }, 15000);
      return () => clearTimeout(timer);
    }
  }, [showClaudeCodePrompt, showClaudeCodeModal]);

  const handleOpenSurvey = () => {
    setShowSurveyPrompt(false);
    setShowSurveyModal(true);
  };

  const handleDismissSurveyPrompt = () => {
    setShowSurveyPrompt(false);
  };

  const handleSurveyComplete = () => {
    setShowSurveyModal(false);
  };

  const handleSurveyModalClose = () => {
    // If they close the modal without completing, show the prompt again
    setShowSurveyModal(false);
    setShowSurveyPrompt(true);
  };

  const handleOpenClaudeCode = () => {
    setShowClaudeCodePrompt(false);
    setShowClaudeCodeModal(true);
  };

  const handleDismissClaudeCodePrompt = () => {
    setShowClaudeCodePrompt(false);
  };

  const handleClaudeCodeComplete = () => {
    setShowClaudeCodeModal(false);
  };

  const handleClaudeCodeModalClose = () => {
    // If they close the modal without completing, show the prompt again
    setShowClaudeCodeModal(false);
    setShowClaudeCodePrompt(true);
  };

  if (authLoading || redirectToLogin) {
    return <LoadingScreen />;
  }

  return (
    <Suspense fallback={<LoadingScreen />}>
      <QueryClientProvider client={queryClient}>
        <ConfigProvider theme={{
          algorithm: isDarkMode ? theme.darkAlgorithm : theme.defaultAlgorithm,
        }}>
          <ThemeProvider accessToken={accessToken}>
            {invitation_id ? (
              <UserDashboard
                userID={userID}
                userRole={userRole}
                premiumUser={premiumUser}
                teams={teams}
                keys={keys}
                setUserRole={setUserRole}
                userEmail={userEmail}
                setUserEmail={setUserEmail}
                setTeams={setTeams}
                setKeys={setKeys}
                organizations={organizations}
                addKey={addKey}
                createClicked={createClicked}
              />
            ) : (
              <div className="flex flex-col min-h-screen">
                <Navbar
                  userID={userID}
                  userRole={userRole}
                  premiumUser={premiumUser}
                  userEmail={userEmail}
                  setProxySettings={setProxySettings}
                  proxySettings={proxySettings}
                  accessToken={accessToken}
                  isPublicPage={false}
                  sidebarCollapsed={sidebarCollapsed}
                  onToggleSidebar={toggleSidebar}
                  isDarkMode={isDarkMode}
                  toggleDarkMode={toggleDarkMode}
                />
                <div className="flex flex-1">
                  <div className="mt-2">
                    <SidebarProvider setPage={updatePage} defaultSelectedKey={page} sidebarCollapsed={sidebarCollapsed} />
                  </div>

                  {page == "api-keys" ? (
                    <UserDashboard
                      userID={userID}
                      userRole={userRole}
                      premiumUser={premiumUser}
                      teams={teams}
                      keys={keys}
                      setUserRole={setUserRole}
                      userEmail={userEmail}
                      setUserEmail={setUserEmail}
                      setTeams={setTeams}
                      setKeys={setKeys}
                      organizations={organizations}
                      addKey={addKey}
                      createClicked={createClicked}
                    />
                  ) : page == "models" ? (
                    <OldModelDashboard
                      token={token}
                      keys={keys}
                      modelData={modelData}
                      setModelData={setModelData}
                      premiumUser={premiumUser}
                      teams={teams}
                    />
                  ) : page == "llm-playground" ? (
                    <PlaygroundPage />
                  ) : page == "users" ? (
                    <ViewUserDashboard
                      userID={userID}
                      userRole={userRole}
                      token={token}
                      keys={keys}
                      teams={teams}
                      accessToken={accessToken}
                      setKeys={setKeys}
                    />
                  ) : page == "teams" ? (
                    <OldTeams
                      teams={teams}
                      setTeams={setTeams}
                      accessToken={accessToken}
                      userID={userID}
                      userRole={userRole}
                      organizations={organizations}
                      premiumUser={premiumUser}
                      searchParams={searchParams}
                    />
                  ) : page == "organizations" ? (
                    <Organizations
                      organizations={organizations}
                      setOrganizations={setOrganizations}
                      userModels={userModels}
                      accessToken={accessToken}
                      userRole={userRole}
                      premiumUser={premiumUser}
                    />
                  ) : page == "admin-panel" ? (
                    <AdminPanel
                      proxySettings={proxySettings}
                    />
                  ) : page == "api_ref" ? (
                    <APIReferenceView proxySettings={proxySettings} />
                  ) : page == "logging-and-alerts" ? (
                    <Settings userID={userID} userRole={userRole} accessToken={accessToken} premiumUser={premiumUser} />
                  ) : page == "budgets" ? (
                    <BudgetPanel accessToken={accessToken} />
                  ) : page == "guardrails" ? (
                    <GuardrailsPanel accessToken={accessToken} userRole={userRole} />
                  ) : page == "policies" ? (
                    <PoliciesPanel accessToken={accessToken} userRole={userRole} />
                  ) : page == "agents" ? (
                    <AgentsPanel accessToken={accessToken} userRole={userRole} />
                  ) : page == "prompts" ? (
                    <PromptsPanel accessToken={accessToken} userRole={userRole} />
                  ) : page == "transform-request" ? (
                    <TransformRequestPanel accessToken={accessToken} />
                  ) : page == "router-settings" ? (
                    <GeneralSettings
                      userID={userID}
                      userRole={userRole}
                      accessToken={accessToken}
                      modelData={modelData}
                    />
                  ) : page == "ui-theme" ? (
                    <UIThemeSettings userID={userID} userRole={userRole} accessToken={accessToken} />
                  ) : page == "cost-tracking" ? (
                    <CostTrackingSettings userID={userID} userRole={userRole} accessToken={accessToken} />
                  ) : page == "model-hub-table" ? (
                    isAdminRole(userRole) ? (
                      <ModelHubTable
                        accessToken={accessToken}
                        publicPage={false}
                        premiumUser={premiumUser}
                        userRole={userRole}
                      />
                    ) : (
                      <PublicModelHub accessToken={accessToken} isEmbedded={true} />
                    )
                  ) : page == "caching" ? (
                    <CacheDashboard
                      userID={userID}
                      userRole={userRole}
                      token={token}
                      accessToken={accessToken}
                      premiumUser={premiumUser}
                    />
                  ) : page == "pass-through-settings" ? (
                    <PassThroughSettings
                      userID={userID}
                      userRole={userRole}
                      accessToken={accessToken}
                      modelData={modelData}
                      premiumUser={premiumUser}
                    />
                  ) : page == "logs" ? (
                    <SpendLogsTable
                      userID={userID}
                      userRole={userRole}
                      token={token}
                      accessToken={accessToken}
                      allTeams={(teams as Team[]) ?? []}
                      premiumUser={premiumUser}
                    />
                  ) : page == "mcp-servers" ? (
                    <MCPServers accessToken={accessToken} userRole={userRole} userID={userID} />
                  ) : page == "search-tools" ? (
                    <SearchTools accessToken={accessToken} userRole={userRole} userID={userID} />
                  ) : page == "tag-management" ? (
                    <TagManagement accessToken={accessToken} userRole={userRole} userID={userID} />
                  ) : page == "claude-code-plugins" ? (
                    <ClaudeCodePluginsPanel accessToken={accessToken} userRole={userRole} />
                  ) : page == "access-groups" ? (
                    <AccessGroupsPage />
                  ) : page == "vector-stores" ? (
                    <VectorStoreManagement accessToken={accessToken} userRole={userRole} userID={userID} />
                  ) : page == "new_usage" ? (
                    <NewUsagePage
                      teams={(teams as Team[]) ?? []}
                      organizations={(organizations as Organization[]) ?? []}
                    />
                  ) : (
                    <Usage
                      userID={userID}
                      userRole={userRole}
                      token={token}
                      accessToken={accessToken}
                      keys={keys}
                      premiumUser={premiumUser}
                    />
                  )}
                </div>

                {/* Survey Components */}
                <SurveyPrompt
                  isVisible={showSurveyPrompt}
                  onOpen={handleOpenSurvey}
                  onDismiss={handleDismissSurveyPrompt}
                />
                <SurveyModal
                  isOpen={showSurveyModal}
                  onClose={handleSurveyModalClose}
                  onComplete={handleSurveyComplete}
                />

                {/* Claude Code Components */}
                <ClaudeCodePrompt
                  isVisible={showClaudeCodePrompt}
                  onOpen={handleOpenClaudeCode}
                  onDismiss={handleDismissClaudeCodePrompt}
                />
                <ClaudeCodeModal
                  isOpen={showClaudeCodeModal}
                  onClose={handleClaudeCodeModalClose}
                  onComplete={handleClaudeCodeComplete}
                />
              </div>
            )}
          </ThemeProvider>
        </ConfigProvider>
      </QueryClientProvider>
    </Suspense>
  );
}

export default function CreateKeyPage() {
  return (
    <Suspense fallback={<LoadingScreen />}>
      <CreateKeyPageContent />
    </Suspense>
  );
}
