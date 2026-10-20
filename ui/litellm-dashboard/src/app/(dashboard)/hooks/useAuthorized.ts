"use client";

import { clearTokenCookies, getCookie } from "@/utils/cookieUtils";
import { checkTokenValidity, decodeToken } from "@/utils/jwtUtils";
import { useEffect, useMemo } from "react";
import { effectiveSessionRole, formatUserRole, isViewOnlySessionRole } from "@/utils/roles";
import { useCurrentUser } from "./currentUser/useCurrentUser";
import { redirectToLogin } from "./redirectToLogin";
import { useUIConfig } from "./uiConfig/useUIConfig";

const useAuthorized = () => {
  const { data: uiConfig, isLoading: isUIConfigLoading } = useUIConfig();

  const token = typeof document !== "undefined" ? getCookie("token") : null;

  const decoded = useMemo(() => decodeToken(token), [token]);
  const isTokenValid = useMemo(() => checkTokenValidity(token), [token]);
  const {
    data: currentUser,
    isLoading: isCurrentUserLoading,
    isError: isCurrentUserError,
  } = useCurrentUser(isTokenValid ? decoded?.key ?? null : null, decoded?.auth_header_name, !isUIConfigLoading);
  const serverRole = currentUser?.user_role ?? "";
  const isLoading = isUIConfigLoading || isCurrentUserLoading;
  const isAuthorized = isTokenValid && !uiConfig?.admin_ui_disabled && !isCurrentUserError;

  // Single useEffect for all redirect logic
  useEffect(() => {
    if (isLoading) return;

    if (!isAuthorized) {
      if (token) {
        clearTokenCookies();
      }
      redirectToLogin();
    }
  }, [isLoading, isAuthorized, token]);

  return {
    isLoading,
    isAuthorized,
    token: isAuthorized ? token : null,
    accessToken: decoded?.key ?? null,
    userId: currentUser?.user_id ?? null,
    userEmail: currentUser?.user_email ?? null,
    userRole: effectiveSessionRole(serverRole),
    userRoleLabel: formatUserRole(serverRole),
    isViewOnly: isViewOnlySessionRole(serverRole),
    premiumUser: decoded?.premium_user ?? null,
    disabledPersonalKeyCreation: decoded?.disabled_non_admin_personal_key_creation ?? null,
    loginMethod: decoded?.login_method ?? null,
    showSSOBanner: decoded?.login_method === "username_password",
    passwordResetRequired: decoded?.password_reset_required === true,
  };
};

export default useAuthorized;
