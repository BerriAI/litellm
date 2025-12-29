"use client";

import { getProxyBaseUrl } from "@/components/networking";
import { clearTokenCookies, getCookie } from "@/utils/cookieUtils";
import { jwtDecode } from "jwt-decode";
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

  const token = typeof document !== "undefined" ? getCookie("token") : null;

  // Redirect after mount if missing/invalid token
  useEffect(() => {
    if (isUIConfigLoading) {
      return;
    }
    if (!token || uiConfig?.admin_ui_disabled) {
      router.replace(`${getProxyBaseUrl()}/ui/login`);
    }
  }, [token, router, isUIConfigLoading, uiConfig]);

  // Decode safely
  const decoded = useMemo(() => {
    if (!token) return null;
    try {
      return jwtDecode(token) as Record<string, any>;
    } catch {
      // Bad token in cookie — clear and bounce
      clearTokenCookies();
      router.replace(`${getProxyBaseUrl()}/ui/login`);
      return null;
    }
  }, [token, router]);

  return {
    token: token,
    accessToken: decoded?.key ?? null,
    userId: decoded?.user_id ?? null,
    userEmail: decoded?.user_email ?? null,
    userRole: formatUserRole(decoded?.user_role ?? null),
    premiumUser: decoded?.premium_user ?? null,
    disabledPersonalKeyCreation: decoded?.disabled_non_admin_personal_key_creation ?? null,
    showSSOBanner: decoded?.login_method === "username_password",
  };
};

export default useAuthorized;
