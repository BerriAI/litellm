"use client";

import { getProxyBaseUrl } from "@/components/networking";
import { clearTokenCookies, getCookie } from "@/utils/cookieUtils";
import { checkTokenValidity, decodeToken } from "@/utils/jwtUtils";
import { useProxyConnection } from "@/contexts/ProxyConnectionContext";
import { useRouter } from "next/navigation";
import { useEffect, useMemo } from "react";
import { useUIConfig } from "./uiConfig/useUIConfig";

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

const useAuthorized = () => {
  const router = useRouter();
  const { data: uiConfig, isLoading: isUIConfigLoading } = useUIConfig();
  const { activeConnection, isRemoteProxy } = useProxyConnection();

  const token = typeof document !== "undefined" ? getCookie("token") : null;

  const decoded = useMemo(() => decodeToken(token), [token]);
  const isTokenValid = useMemo(() => checkTokenValidity(token), [token]);
  const isLoading = isUIConfigLoading && !isRemoteProxy;

  // For remote proxies, authorized if we have a stored API key
  const isRemoteAuthorized = isRemoteProxy && !!activeConnection?.apiKey;
  const isAuthorized = isRemoteAuthorized || (isTokenValid && !uiConfig?.admin_ui_disabled);

  // Single useEffect for all redirect logic
  useEffect(() => {
    if (isLoading) return;

    if (!isAuthorized && !isRemoteProxy) {
      if (token) {
        clearTokenCookies();
      }
      router.replace(`${getProxyBaseUrl()}/ui/login`);
    }
  }, [isLoading, isAuthorized, isRemoteProxy, token, router]);

  // For remote proxies, use the stored API key as the access token
  const effectiveAccessToken = isRemoteProxy
    ? activeConnection?.apiKey ?? null
    : decoded?.key ?? null;

  return {
    isLoading,
    isAuthorized,
    token: isAuthorized ? token : null,
    accessToken: effectiveAccessToken,
    userId: isRemoteProxy ? null : (decoded?.user_id ?? null),
    userEmail: isRemoteProxy ? null : (decoded?.user_email ?? null),
    userRole: isRemoteProxy ? "Admin" : formatUserRole(decoded?.user_role),
    premiumUser: decoded?.premium_user ?? null,
    disabledPersonalKeyCreation: decoded?.disabled_non_admin_personal_key_creation ?? null,
    showSSOBanner: isRemoteProxy ? false : decoded?.login_method === "username_password",
  };
};

export default useAuthorized;
