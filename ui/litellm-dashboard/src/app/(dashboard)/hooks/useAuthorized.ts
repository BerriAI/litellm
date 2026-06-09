"use client";

import { getProxyBaseUrl } from "@/components/networking";
import { clearTokenCookies } from "@/utils/cookieUtils";
import { buildLoginUrlWithReturn, storeReturnUrl } from "@/utils/returnUrlUtils";
import { useRouter } from "next/navigation";
import { useCallback, useEffect } from "react";
import { useAuth } from "@/contexts/AuthContext";
import { useUIConfig } from "./uiConfig/useUIConfig";

// Decoded-JWT fields keep the pre-consolidation `any` typing: the old hook returned
// `any` here and ~25 call sites pass these where `string` is expected. Tightening to
// the context's `string | null` is a follow-up that has to fix those call sites.
type LegacyDecodedField = any;

/**
 * Policy hook for pages that require an authenticated user. Auth state itself
 * lives in AuthContext (single decode at the root); this hook layers on the
 * admin_ui_disabled check and the redirect-to-login side effect.
 */
const useAuthorized = () => {
  const router = useRouter();
  const { data: uiConfig, isLoading: isUIConfigLoading } = useUIConfig();
  const {
    authLoading,
    token,
    userID,
    userRole,
    userEmail,
    accessToken,
    premiumUser,
    disabledPersonalKeyCreation,
    showSSOBanner,
  } = useAuth();

  const isLoading = authLoading || isUIConfigLoading;
  const isAuthorized = token !== null && !uiConfig?.admin_ui_disabled;

  const redirectToLogin = useCallback(() => {
    storeReturnUrl();
    const baseLoginUrl = `${getProxyBaseUrl()}/ui/login`;
    router.replace(buildLoginUrlWithReturn(baseLoginUrl));
  }, [router]);

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
    accessToken: accessToken as LegacyDecodedField,
    userId: userID as LegacyDecodedField,
    userEmail: userEmail as LegacyDecodedField,
    userRole,
    premiumUser: premiumUser as LegacyDecodedField,
    disabledPersonalKeyCreation: disabledPersonalKeyCreation as LegacyDecodedField,
    showSSOBanner,
  };
};

export default useAuthorized;
