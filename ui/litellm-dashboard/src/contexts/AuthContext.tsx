"use client";

import React, { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";
import { jwtDecode } from "jwt-decode";
import { clearTokenCookies, getCookie } from "@/utils/cookieUtils";
import { isJwtExpired } from "@/utils/jwtUtils";
import { formatUserRole } from "@/utils/roles";
import { getUiConfig, setGlobalLitellmHeaderName } from "@/components/networking";

type AuthContextValue = {
  authLoading: boolean;
  token: string | null;
  userID: string | null;
  userRole: string;
  userEmail: string | null;
  accessToken: string | null;
  premiumUser: boolean;
  disabledPersonalKeyCreation: boolean;

  clearAuth: () => void;
  setUserRole: React.Dispatch<React.SetStateAction<string>>;
  setUserEmail: React.Dispatch<React.SetStateAction<string | null>>;
};

const AuthContext = createContext<AuthContextValue | null>(null);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [authLoading, setAuthLoading] = useState(true);
  const [token, setToken] = useState<string | null>(null);
  const [userRole, setUserRole] = useState(formatUserRole(""));
  const [userEmail, setUserEmail] = useState<string | null>(null);

  const decoded = useMemo<{ [k: string]: any } | null>(() => {
    if (!token || isJwtExpired(token)) return null;
    try {
      return jwtDecode(token);
    } catch {
      return null;
    }
  }, [token]);

  const clearAuth = useCallback(() => {
    clearTokenCookies();
    setToken(null);
  }, []);

  // Load runtime UI config (populates proxyBaseUrl etc.) before clearing
  // authLoading, so any consumer that builds proxy-rooted URLs from authLoading=false
  // (e.g. the unauthenticated login redirect) sees the resolved value rather than
  // the module-init default. Then read the cookie and validate JWT expiry.
  useEffect(() => {
    let cancelled = false;

    (async () => {
      try {
        await getUiConfig();
      } catch {
        // proceed regardless; auth state must still be resolved
      }

      if (cancelled) return;

      const raw = getCookie("token");
      const valid = raw && !isJwtExpired(raw) ? raw : null;

      // Clear expired/invalid token so downstream code doesn't keep trying to use it.
      if (raw && !valid) {
        clearTokenCookies();
      }

      setToken(valid);
      setAuthLoading(false);
    })();

    return () => {
      cancelled = true;
    };
  }, []);

  // Side effects of a token change: drop tokens that fail to decode, sync the
  // mutable user fields, and apply any custom auth header. The reset branch runs
  // after any commit that nulls the token, so user fields cannot survive clearAuth
  // even if a stale sync was queued in the same commit.
  useEffect(() => {
    if (token && !decoded) {
      clearAuth();
      return;
    }
    if (!decoded) {
      setUserRole(formatUserRole(""));
      setUserEmail(null);
      return;
    }

    if (decoded.user_role) {
      setUserRole(formatUserRole(decoded.user_role));
    }
    if (decoded.user_email) {
      setUserEmail(decoded.user_email);
    }
    if (decoded.auth_header_name) {
      setGlobalLitellmHeaderName(decoded.auth_header_name);
    }
  }, [token, decoded, clearAuth]);

  const value: AuthContextValue = {
    authLoading,
    token,
    userID: decoded?.user_id ?? null,
    userRole,
    userEmail,
    accessToken: decoded?.key ?? null,
    premiumUser: !!decoded?.premium_user,
    disabledPersonalKeyCreation: !!decoded?.disabled_non_admin_personal_key_creation,
    clearAuth,
    setUserRole,
    setUserEmail,
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
