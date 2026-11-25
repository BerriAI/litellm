"use client";

import HomePage from "@/app/(dashboard)/home/_components/HomePage";
import LoadingScreen from "@/components/common_components/LoadingScreen";
import { getRuntimeBackendBaseUrl, getUiPageUrl, proxyBaseUrl } from "@/components/networking";
import { useAuth } from "@/contexts/AuthContext";
import {
  buildLoginUrlWithReturn,
  consumeReturnUrl,
  getLoginUrl,
  normalizeUrlForCompare,
  storeReturnUrl,
} from "@/utils/returnUrlUtils";
import { navigateToUiPage } from "@/utils/uiNavigation";
import { useUISettings } from "@/app/(dashboard)/hooks/uiSettings/useUISettings";
import { legacyPageRedirectHref } from "@/app/(dashboard)/legacyPageRoutes";
import { flattenNavItems } from "@/components/CommandPalette/utils";
import { useVisibleMenuGroups } from "@/components/CommandPalette/useVisibleMenuGroups";
import { internalUserRoles } from "@/utils/roles";
import { uiHref } from "@/utils/uiHref";
import { useRouter, useSearchParams } from "next/navigation";
import { Suspense, useEffect, useRef } from "react";

function CreateKeyPageContent() {
  const { authLoading, token, userRole } = useAuth();

  const router = useRouter();
  const searchParams = useSearchParams()!;
  const { data: uiSettings, isLoading: isUISettingsLoading } = useUISettings();
  const visibleNavItems = flattenNavItems(useVisibleMenuGroups());
  const isInternalUser = internalUserRoles.includes(userRole);
  const isHomeVisible = visibleNavItems.some(({ route }) => route === "home");
  const firstVisibleRoute = visibleNavItems.find(({ route }) => route !== "home")?.route;
  const hasConfiguredPageVisibility = uiSettings?.values?.enabled_ui_pages_internal_users != null;
  const shouldApplyPageVisibility = isInternalUser && hasConfiguredPageVisibility;
  const shouldRedirectHome = shouldApplyPageVisibility && !isUISettingsLoading && !isHomeVisible;
  const homeRedirectRoute = shouldRedirectHome ? firstVisibleRoute : undefined;

  // Track if we've already attempted a return URL redirect to prevent race conditions
  const hasAttemptedReturnRedirectRef = useRef(false);

  const redirectToLogin = authLoading === false && token === null;

  useEffect(() => {
    if (redirectToLogin) {
      // Store the current URL so we can redirect back after login
      storeReturnUrl();
      // Build login URL with return URL parameter
      const baseLoginUrl = getUiPageUrl(getLoginUrl(proxyBaseUrl || ""));
      const dest = buildLoginUrlWithReturn(baseLoginUrl);
      // Replace instead of assigning to avoid back-button loops
      window.location.replace(dest);
    }
  }, [redirectToLogin]);

  const legacyRedirectHref = legacyPageRedirectHref(searchParams);
  useEffect(() => {
    if (!authLoading && legacyRedirectHref !== null) {
      router.replace(legacyRedirectHref);
    }
  }, [authLoading, legacyRedirectHref, router]);

  useEffect(() => {
    if (!authLoading && homeRedirectRoute) {
      router.replace(uiHref(homeRedirectRoute));
    }
  }, [authLoading, homeRedirectRoute, router]);

  // Check for a stored return URL after successful authentication
  // This handles the case where user comes back from SSO and we need to redirect to the original URL
  useEffect(() => {
    // Skip if still loading, no token, or we've already attempted a redirect
    if (authLoading || !token || hasAttemptedReturnRedirectRef.current) {
      return;
    }

    // Mark that we've attempted the redirect to prevent race conditions
    // This prevents duplicate redirects if token changes (e.g., refresh)
    hasAttemptedReturnRedirectRef.current = true;

    // Check for a stored return URL
    const returnUrl = consumeReturnUrl(getRuntimeBackendBaseUrl());
    if (returnUrl) {
      const currentUrl = window.location.href;
      const normalizedReturnUrl = normalizeUrlForCompare(getUiPageUrl(returnUrl));
      const normalizedCurrentUrl = normalizeUrlForCompare(currentUrl);
      // Only redirect if the return URL is different from the current URL
      // This prevents infinite redirect loops
      if (normalizedReturnUrl !== normalizedCurrentUrl) {
        navigateToUiPage(returnUrl, true);
      }
    }
  }, [authLoading, token]);

  useEffect(() => {
    if (!token) {
      hasAttemptedReturnRedirectRef.current = false;
    }
  }, [token]);

  const isRedirectingToPage = legacyRedirectHref !== null || Boolean(homeRedirectRoute);
  const isRedirecting = redirectToLogin || isRedirectingToPage;

  if (authLoading || isRedirecting || (isInternalUser && isUISettingsLoading)) {
    return <LoadingScreen />;
  }

  return <HomePage />;
}

export default function CreateKeyPage() {
  return (
    <Suspense fallback={<LoadingScreen />}>
      <CreateKeyPageContent />
    </Suspense>
  );
}
