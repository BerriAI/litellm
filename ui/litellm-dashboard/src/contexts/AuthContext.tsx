"use client";

import React, { createContext, useContext, useEffect, useState } from "react";
import { jwtDecode } from "jwt-decode";
import { clearTokenCookies, getCookie } from "@/utils/cookieUtils";
import { isJwtExpired } from "@/utils/jwtUtils";
import { formatUserRole } from "@/utils/roles";
import { getUiConfig, setGlobalLitellmHeaderName } from "@/components/networking";

function deleteCookie(name: string, path = "/") {
  document.cookie = `${name}=; Max-Age=0; Path=${path}`;
  if (name === "token") {
    clearTokenCookies();
  }
}

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

  setToken: React.Dispatch<React.SetStateAction<string | null>>;
  setUserID: React.Dispatch<React.SetStateAction<string | null>>;
  setUserRole: React.Dispatch<React.SetStateAction<string>>;
  setUserEmail: React.Dispatch<React.SetStateAction<string | null>>;
  setAccessToken: React.Dispatch<React.SetStateAction<string | null>>;
  setPremiumUser: React.Dispatch<React.SetStateAction<boolean>>;
  setShowSSOBanner: React.Dispatch<React.SetStateAction<boolean>>;
};

const AuthContext = createContext<AuthContextValue | null>(null);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [authLoading, setAuthLoading] = useState(true);
  const [token, setToken] = useState<string | null>(null);
  const [userID, setUserID] = useState<string | null>(null);
  const [userRole, setUserRole] = useState("");
  const [userEmail, setUserEmail] = useState<string | null>(null);
  const [accessToken, setAccessToken] = useState<string | null>(null);
  const [premiumUser, setPremiumUser] = useState(false);
  const [disabledPersonalKeyCreation, setDisabledPersonalKeyCreation] = useState(false);
  const [showSSOBanner, setShowSSOBanner] = useState(true);

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
        deleteCookie("token", "/");
      }

      setToken(valid);
      setAuthLoading(false);
    })();

    return () => {
      cancelled = true;
    };
  }, []);

  // Decode JWT and populate derived auth state whenever the token changes.
  useEffect(() => {
    if (!token) {
      return;
    }

    if (isJwtExpired(token)) {
      deleteCookie("token", "/");
      setToken(null);
      return;
    }

    let decoded: { [k: string]: any } | null = null;
    try {
      decoded = jwtDecode(token);
    } catch {
      deleteCookie("token", "/");
      setToken(null);
      return;
    }

    if (!decoded) return;

    setAccessToken(decoded.key);
    setDisabledPersonalKeyCreation(decoded.disabled_non_admin_personal_key_creation);

    if (decoded.user_role) {
      setUserRole(formatUserRole(decoded.user_role));
    }
    if (decoded.user_email) {
      setUserEmail(decoded.user_email);
    }
    if (decoded.login_method) {
      setShowSSOBanner(decoded.login_method === "username_password");
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
  }, [token]);

  const value: AuthContextValue = {
    authLoading,
    token,
    userID,
    userRole,
    userEmail,
    accessToken,
    premiumUser,
    disabledPersonalKeyCreation,
    showSSOBanner,
    setToken,
    setUserID,
    setUserRole,
    setUserEmail,
    setAccessToken,
    setPremiumUser,
    setShowSSOBanner,
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
