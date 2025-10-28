"use client";

import { useEffect, useMemo } from "react";
import { useRouter } from "next/navigation";
import { jwtDecode } from "jwt-decode";
import { clearTokenCookies, getCookie } from "@/utils/cookieUtils";

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

  const token = typeof document !== "undefined" ? getCookie("token") : null;

  // Redirect after mount if missing/invalid token
  useEffect(() => {
    if (!token) {
      router.replace("/sso/key/generate");
    }
  }, [token, router]);

  // Decode safely
  const decoded = useMemo(() => {
    if (!token) return null;
    try {
      return jwtDecode(token) as Record<string, any>;
    } catch {
      // Bad token in cookie — clear and bounce
      clearTokenCookies();
      router.replace("/sso/key/generate");
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
    showSSOBanner: decoded?.login_method === "username_password" ?? false,
  };
};

export default useAuthorized;
