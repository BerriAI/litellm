"use client";

import { clearTokenCookies, getCookie } from "@/utils/cookieUtils";
import { checkTokenValidity, decodeToken } from "@/utils/jwtUtils";
import { useEffect, useMemo } from "react";
import { effectiveSessionRole, formatUserRole, isViewOnlySessionRole, userRoleDisplayLabel } from "@/utils/roles";
import { useCurrentUser } from "./currentUser/useCurrentUser";
import { redirectToLogin } from "./redirectToLogin";
import { useUIConfig } from "./uiConfig/useUIConfig";
import { useControlPlane } from "@/contexts/ControlPlaneContext";

const useAuthorized = () => {
  const controlPlane = useControlPlane();
  const { data: uiConfig, isLoading: isUIConfigLoading } = useUIConfig();

  const token = !controlPlane && typeof document !== "undefined" ? getCookie("token") : null;

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
    if (controlPlane || isLoading) return;

    if (!isAuthorized) {
      if (token) {
        clearTokenCookies();
      }
      redirectToLogin();
    }
  }, [controlPlane, isLoading, isAuthorized, token]);

  if (controlPlane) {
    const { principal, token: accessToken, canWrite } = controlPlane;
    return {
      isLoading: false,
      isAuthorized: true,
      token: null,
      accessToken,
      userId: principal.subject,
      userEmail: principal.email ?? null,
      userAlias: null,
      userRole: canWrite ? "Admin" : "Admin Viewer",
      userRoleLabel: canWrite ? "Admin" : "Admin Viewer",
      isViewOnly: !canWrite,
      premiumUser: true,
      disabledPersonalKeyCreation: !canWrite,
      loginMethod: null,
      showSSOBanner: false,
      passwordResetRequired: false,
    };
  }

  return {
    isLoading,
    isAuthorized,
    token: isAuthorized ? token : null,
    accessToken: decoded?.key ?? null,
    userId: currentUser?.user_id ?? null,
    userEmail: currentUser?.user_email ?? null,
    userAlias: currentUser?.user_alias ?? null,
    userRole: effectiveSessionRole(serverRole),
    userRoleLabel: userRoleDisplayLabel(formatUserRole(serverRole)),
    isViewOnly: isViewOnlySessionRole(serverRole),
    premiumUser: decoded?.premium_user ?? null,
    disabledPersonalKeyCreation: decoded?.disabled_non_admin_personal_key_creation ?? null,
    loginMethod: decoded?.login_method ?? null,
    showSSOBanner: decoded?.login_method === "username_password",
    passwordResetRequired: decoded?.password_reset_required === true,
  };
};

export default useAuthorized;
