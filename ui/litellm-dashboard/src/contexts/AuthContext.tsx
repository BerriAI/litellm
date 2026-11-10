"use client";

import React, { createContext, useContext, useEffect, useMemo, useState } from "react";
import { jwtDecode } from "jwt-decode";
import { clearTokenCookies, getAuthToken, getCookie, useAuthToken } from "@/utils/cookieUtils";
import { effectiveSessionRole } from "@/utils/roles";
import { getUiConfig, setGlobalLitellmHeaderName } from "@/components/networking";

type SessionClaims = {
  key?: string;
  user_id?: string;
  user_role?: string;
  user_email?: string;
  premium_user?: boolean;
  disabled_non_admin_personal_key_creation?: boolean;
  login_method?: string;
  password_reset_required?: boolean;
  auth_header_name?: string;
};

type AuthContextValue = {
  authLoading: boolean;
  token: string | null;
  userID: string | null;
  userRole: string;
  userEmail: string | null;
  accessToken: string | null;
  premiumUser: boolean;
  disabledPersonalKeyCreation: boolean;
  showSSOBanner: boolean;
  passwordResetRequired: boolean;
};

const AuthContext = createContext<AuthContextValue | null>(null);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [authLoading, setAuthLoading] = useState(true);
  const token = useAuthToken();
  const decoded = useMemo(() => (token ? jwtDecode<SessionClaims>(token) : null), [token]);

  useEffect(() => {
    let cancelled = false;

    (async () => {
      try {
        await getUiConfig();
      } finally {
        if (!cancelled) setAuthLoading(false);
      }
    })().catch(() => {});

    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (getCookie("token") && !getAuthToken()) {
      clearTokenCookies();
    }
  }, [token]);

  useEffect(() => {
    setGlobalLitellmHeaderName(decoded?.auth_header_name ?? "Authorization");
  }, [decoded?.auth_header_name]);

  const value: AuthContextValue = {
    authLoading,
    token,
    userID: decoded?.user_id ?? null,
    userRole: decoded?.user_role ? effectiveSessionRole(decoded.user_role) : "",
    userEmail: decoded?.user_email ?? null,
    accessToken: decoded?.key ?? null,
    premiumUser: decoded?.premium_user ?? false,
    disabledPersonalKeyCreation: decoded?.disabled_non_admin_personal_key_creation ?? false,
    showSSOBanner: decoded?.login_method ? decoded.login_method === "username_password" : true,
    passwordResetRequired: decoded?.password_reset_required === true,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext);
  if (!ctx) {
    throw new Error("useAuth must be used within an AuthProvider");
  }
  return ctx;
}
