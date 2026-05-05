"use client";

import { getProxyBaseUrl } from "@/components/networking";
import { clearTokenCookies, getCookie } from "@/utils/cookieUtils";
import { checkTokenValidity, decodeToken } from "@/utils/jwtUtils";
import { buildLoginUrlWithReturn, storeReturnUrl } from "@/utils/returnUrlUtils";
import { useRouter } from "next/navigation";
import { useCallback, useEffect, useMemo } from "react";
import { formatUserRole } from "@/utils/roles";
import { useUIConfig } from "./uiConfig/useUIConfig";

const useAuthorized = () => {
  const router = useRouter();
  const { data: uiConfig, isLoading: isUIConfigLoading } = useUIConfig();

  const token = typeof document !== "undefined" ? getCookie("token") : null;

  const decoded = useMemo(() => decodeToken(token), [token]);
  const isTokenValid = useMemo(() => checkTokenValidity(token), [token]);
  const isLoading = isUIConfigLoading;
  const isAuthorized = isTokenValid && !uiConfig?.admin_ui_disabled;

  // Helper function to redirect to login while preserving the current URL
  const redirectToLogin = useCallback(() => {
    storeReturnUrl();
    const baseLoginUrl = `${getProxyBaseUrl()}/ui/login`;
    const loginUrlWithReturn = buildLoginUrlWithReturn(baseLoginUrl);
    router.replace(loginUrlWithReturn);
  }, [router]);

  // Single useEffect for all redirect logic
  useEffect(() => {
    if (isLoading) return;

    if (!isAuthorized) {
      if (token) {
        clearTokenCookies();
      }
      redirectToLogin();
    }
  }, [isLoading, isAuthorized, token, redirectToLogin]);

  return {
    isLoading,
    isAuthorized,
    token: isAuthorized ? token : null,
    accessToken: decoded?.key ?? null,
    userId: decoded?.user_id ?? null,
    userEmail: decoded?.user_email ?? null,
    userRole: formatUserRole(decoded?.user_role),
    premiumUser: decoded?.premium_user ?? null,
    disabledPersonalKeyCreation: decoded?.disabled_non_admin_personal_key_creation ?? null,
    showSSOBanner: decoded?.login_method === "username_password",
  };
};

export default useAuthorized;
