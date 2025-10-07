"use client";

import React, { useEffect, useMemo, useState } from "react";
import { jwtDecode } from "jwt-decode";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { Team } from "@/components/key_team_helpers/key_list";
import Navbar from "@/components/navbar";
import { ThemeProvider } from "@/contexts/ThemeContext";
import UserDashboard from "@/components/user_dashboard";
import ModelDashboard from "@/components/templates/model_dashboard";
import ViewUserDashboard from "@/components/view_users";
import Teams from "@/components/teams";
import Organizations from "@/components/organizations";
import { fetchOrganizations } from "@/components/organizations";
import AdminPanel from "@/components/admins";
import Settings from "@/components/settings";
import GeneralSettings from "@/components/general_settings";
import PassThroughSettings from "@/components/pass_through_settings";
import BudgetPanel from "@/components/budgets/budget_panel";
import SpendLogsTable from "@/components/view_logs";
import ModelHubTable from "@/components/model_hub_table";
import NewUsagePage from "@/components/new_usage";
import APIRef from "@/components/api_ref";
import ChatUI from "@/components/chat_ui/ChatUI";
import Usage from "@/components/usage";
import CacheDashboard from "@/components/cache_dashboard";
import { getUiConfig, proxyBaseUrl, setGlobalLitellmHeaderName } from "@/components/networking";
import { Organization } from "@/components/networking";
import GuardrailsPanel from "@/components/guardrails";
import PromptsPanel from "@/components/prompts";
import TransformRequestPanel from "@/components/transform_request";
import { fetchUserModels } from "@/components/organisms/create_key_button";
import { fetchTeams } from "@/components/common_components/fetch_teams";
import { MCPServers } from "@/components/mcp_tools";
import TagManagement from "@/components/tag_management";
import VectorStoreManagement from "@/components/vector_store_management";
import UIThemeSettings from "@/components/ui_theme_settings";
import { UiLoadingSpinner } from "@/components/ui/ui-loading-spinner";
import { cx } from "@/lib/cva.config";
import Sidebar2 from "@/app/(dashboard)/components/Sidebar2";

/** ---- BASE URL HELPERS ---- */
function normalizeBasePrefix(raw: string | undefined | null): string {
  const trimmed = (raw ?? "").trim();
  if (!trimmed) return "";
  const core = trimmed.replace(/^\/+/, "").replace(/\/+$/, "");
  return core ? `/${core}` : "";
}
const BASE_PREFIX = normalizeBasePrefix(process.env.NEXT_PUBLIC_BASE_URL);
function withBase(path: string): string {
  const p = path.startsWith("/") ? path : `/${path}`;
  return `${BASE_PREFIX}${p}` || p;
}
/** -------------------------------- */

function getCookie(name: string) {
  if (typeof document === "undefined") return null;
  const cookieValue = document.cookie.split("; ").find((row) => row.startsWith(name + "="));
  return cookieValue ? cookieValue.split("=")[1] : null;
}

function formatUserRole(userRole: string) {
  if (!userRole) return "Undefined Role";
  switch (userRole.toLowerCase()) {
    case "app_owner":
    case "demo_app_owner":
      return "App Owner";
    case "app_admin":
    case "proxy_admin":
      return "Admin";
    case "proxy_admin_viewer":
      return "Admin Viewer";
    case "org_admin":
      return "Org Admin";
    case "internal_user":
      return "Internal User";
    case "internal_user_viewer":
    case "internal_viewer":
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
}

const queryClient = new QueryClient();

function LoadingScreen() {
  return (
    <div className={cx("h-screen", "flex items-center justify-center gap-4")}>
      <div className="text-lg font-medium py-2 pr-4 border-r border-r-gray-200">🚅 LiteLLM</div>
      <div className="flex items-center justify-center gap-2">
        <UiLoadingSpinner className="size-4" />
        <span className="text-gray-600 text-sm">Loading...</span>
      </div>
    </div>
  );
}

/** Derive the app "page" from the URL without triggering App Router navigations */
function getPageFromLocation(loc: Location): string {
  const sp = new URLSearchParams(loc.search);
  const p = sp.get("page");
  if (p) return p;
  // vanity route for refactored UI
  if (loc.pathname.endsWith("/virtual-keys")) return "api-keys";
  return "api-keys";
}

export default function CreateKeyPage() {
  const [userRole, setUserRole] = useState("");
  const [premiumUser, setPremiumUser] = useState(false);
  const [disabledPersonalKeyCreation, setDisabledPersonalKeyCreation] = useState(false);
  const [userEmail, setUserEmail] = useState<null | string>(null);
  const [teams, setTeams] = useState<Team[] | null>(null);
  const [keys, setKeys] = useState<null | any[]>([]);
  const [organizations, setOrganizations] = useState<Organization[]>([]);
  const [userModels, setUserModels] = useState<string[]>([]);
  const [proxySettings, setProxySettings] = useState<ProxySettings>({ PROXY_BASE_URL: "", PROXY_LOGOUT_URL: "" });
  const [showSSOBanner, setShowSSOBanner] = useState<boolean>(true);

  // Stable local mirror of URLSearchParams (no suspense, no flashes)
  const [searchParams, setSearchParams] = useState<URLSearchParams>(() =>
    typeof window === "undefined" ? new URLSearchParams() : new URLSearchParams(window.location.search),
  );
  const invitation_id = useMemo(() => searchParams.get("invitation_id"), [searchParams]);

  // Local page state (drives UI)
  const [page, setPage] = useState<string>(() =>
    typeof window === "undefined" ? "api-keys" : getPageFromLocation(window.location),
  );

  // Keep history/back/forward in sync with UI
  useEffect(() => {
    const onPop = () => {
      if (typeof window === "undefined") return;
      setSearchParams(new URLSearchParams(window.location.search));
      setPage(getPageFromLocation(window.location));
    };
    window.addEventListener("popstate", onPop);
    return () => window.removeEventListener("popstate", onPop);
  }, []);

  // Update URL without triggering App Router reload/suspense
  const updatePage = (newPage: string) => {
    if (typeof window === "undefined") return;
    // 1) instant UI update
    setPage(newPage);
    // 2) URL update under base prefix
    const sp = new URLSearchParams(window.location.search);
    sp.set("page", newPage);
    const url = withBase(`/?${sp.toString()}`);
    window.history.pushState(null, "", url);
    // 3) keep our local mirror in sync
    setSearchParams(new URLSearchParams(sp));
  };

  const [token, setToken] = useState<string | null>(null);
  const [createClicked, setCreateClicked] = useState<boolean>(false);
  const [authLoading, setAuthLoading] = useState(true);
  const [userID, setUserID] = useState<string | null>(null);

  const [accessToken, setAccessToken] = useState<string | null>(null);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);

  const toggleSidebar = () => setSidebarCollapsed((v) => !v);

  const addKey = (data: any) => {
    setKeys((prevData) => (prevData ? [...prevData, data] : [data]));
    setCreateClicked((v) => !v);
  };
  const redirectToLogin = authLoading === false && token === null && invitation_id === null;

  useEffect(() => {
    const token = getCookie("token");
    getUiConfig().then(() => {
      setToken(token);
      setAuthLoading(false);
    });
  }, []);

  useEffect(() => {
    if (redirectToLogin) {
      window.location.href = (proxyBaseUrl || "") + "/sso/key/generate";
    }
  }, [redirectToLogin]);

  useEffect(() => {
    if (!token) return;
    const decoded = jwtDecode(token) as { [key: string]: any };
    if (!decoded) return;

    setAccessToken(decoded.key);
    setDisabledPersonalKeyCreation(decoded.disabled_non_admin_personal_key_creation);

    if (decoded.user_role) {
      const formattedUserRole = formatUserRole(decoded.user_role);
      setUserRole(formattedUserRole);
      if (formattedUserRole === "Admin Viewer") setPage("usage");
    }
    if (decoded.user_email) setUserEmail(decoded.user_email);
    if (decoded.login_method) setShowSSOBanner(decoded.login_method === "username_password");
    if (decoded.premium_user) setPremiumUser(decoded.premium_user);
    if (decoded.auth_header_name) setGlobalLitellmHeaderName(decoded.auth_header_name);
    if (decoded.user_id) setUserID(decoded.user_id);
  }, [token]);

  useEffect(() => {
    if (accessToken && userID && userRole) {
      fetchUserModels(userID, userRole, accessToken, setUserModels);
      fetchTeams(accessToken, userID, userRole, null, setTeams);
    }
    if (accessToken) {
      fetchOrganizations(accessToken, setOrganizations);
    }
  }, [accessToken, userID, userRole]);

  if (authLoading || redirectToLogin) {
    return <LoadingScreen />;
  }

  return (
    <QueryClientProvider client={queryClient}>
      <ThemeProvider accessToken={accessToken}>
        {invitation_id ? (
          <UserDashboard
            userID={userID}
            userRole={userRole}
            userEmail={userEmail}
            premiumUser={premiumUser}
            teams={teams}
            keys={keys}
            setUserRole={setUserRole}
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
            />
            <div className="flex flex-1 overflow-auto">
              <div className="mt-2">
                <Sidebar2
                  defaultSelectedKey={page}
                  setPage={updatePage}
                  accessToken={accessToken}
                  userRole={userRole}
                />
              </div>

              {page === "api-keys" ? (
                <UserDashboard
                  userID={userID}
                  userRole={userRole}
                  userEmail={userEmail}
                  premiumUser={premiumUser}
                  teams={teams}
                  keys={keys}
                  setUserRole={setUserRole}
                  setUserEmail={setUserEmail}
                  setTeams={setTeams}
                  setKeys={setKeys}
                  organizations={organizations}
                  addKey={addKey}
                  createClicked={createClicked}
                />
              ) : page === "models" ? (
                <ModelDashboard
                  userID={userID}
                  userRole={userRole}
                  token={token}
                  keys={keys}
                  accessToken={accessToken}
                  modelData={{ data: [] }}
                  setModelData={() => {}}
                  premiumUser={premiumUser}
                  teams={teams}
                />
              ) : page === "llm-playground" ? (
                <ChatUI
                  userID={userID}
                  userRole={userRole}
                  token={token}
                  accessToken={accessToken}
                  disabledPersonalKeyCreation={disabledPersonalKeyCreation}
                />
              ) : page === "users" ? (
                <ViewUserDashboard
                  userID={userID}
                  userRole={userRole}
                  token={token}
                  keys={keys}
                  teams={teams}
                  accessToken={accessToken}
                  setKeys={setKeys}
                />
              ) : page === "teams" ? (
                <Teams
                  teams={teams}
                  setTeams={setTeams}
                  searchParams={searchParams}
                  accessToken={accessToken}
                  userID={userID}
                  userRole={userRole}
                  organizations={organizations}
                  premiumUser={premiumUser}
                />
              ) : page === "organizations" ? (
                <Organizations
                  organizations={organizations}
                  setOrganizations={setOrganizations}
                  userModels={userModels}
                  accessToken={accessToken}
                  userRole={userRole}
                  premiumUser={premiumUser}
                />
              ) : page === "admin-panel" ? (
                <AdminPanel
                  setTeams={setTeams}
                  searchParams={searchParams}
                  accessToken={accessToken}
                  userID={userID}
                  showSSOBanner={showSSOBanner}
                  premiumUser={premiumUser}
                  proxySettings={proxySettings}
                />
              ) : page === "api_ref" ? (
                <APIRef proxySettings={proxySettings} />
              ) : page === "settings" ? (
                <Settings userID={userID} userRole={userRole} accessToken={accessToken} premiumUser={premiumUser} />
              ) : page === "budgets" ? (
                <BudgetPanel accessToken={accessToken} />
              ) : page === "guardrails" ? (
                <GuardrailsPanel accessToken={accessToken} userRole={userRole} />
              ) : page === "prompts" ? (
                <PromptsPanel accessToken={accessToken} userRole={userRole} />
              ) : page === "transform-request" ? (
                <TransformRequestPanel accessToken={accessToken} />
              ) : page === "general-settings" ? (
                <GeneralSettings userID={userID} userRole={userRole} accessToken={accessToken} modelData={{}} />
              ) : page === "ui-theme" ? (
                <UIThemeSettings userID={userID} userRole={userRole} accessToken={accessToken} />
              ) : page === "model-hub-table" ? (
                <ModelHubTable
                  accessToken={accessToken}
                  publicPage={false}
                  premiumUser={premiumUser}
                  userRole={userRole}
                />
              ) : page === "caching" ? (
                <CacheDashboard
                  userID={userID}
                  userRole={userRole}
                  token={token}
                  accessToken={accessToken}
                  premiumUser={premiumUser}
                />
              ) : page === "pass-through-settings" ? (
                <PassThroughSettings userID={userID} userRole={userRole} accessToken={accessToken} modelData={{}} />
              ) : page === "logs" ? (
                <SpendLogsTable
                  userID={userID}
                  userRole={userRole}
                  token={token}
                  accessToken={accessToken}
                  allTeams={(teams as Team[]) ?? []}
                  premiumUser={premiumUser}
                />
              ) : page === "mcp-servers" ? (
                <MCPServers accessToken={accessToken} userRole={userRole} userID={userID} />
              ) : page === "tag-management" ? (
                <TagManagement accessToken={accessToken} userRole={userRole} userID={userID} />
              ) : page === "vector-stores" ? (
                <VectorStoreManagement accessToken={accessToken} userRole={userRole} userID={userID} />
              ) : page === "new_usage" ? (
                <NewUsagePage
                  userID={userID}
                  userRole={userRole}
                  accessToken={accessToken}
                  teams={(teams as Team[]) ?? []}
                  premiumUser={premiumUser}
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
          </div>
        )}
      </ThemeProvider>
    </QueryClientProvider>
  );
}
