"use client";

import ApiKeysDashboard from "@/app/(dashboard)/api-keys/ApiKeysDashboard";
import LoadingScreen from "@/components/common_components/LoadingScreen";
import { proxyBaseUrl } from "@/components/networking";
import { useKeys } from "@/app/(dashboard)/hooks/keys/useKeys";
import { isAdminRole } from "@/utils/roles";
import { useAuth } from "@/contexts/AuthContext";
import {
  buildLoginUrlWithReturn,
  consumeReturnUrl,
  getLoginUrl,
  isValidReturnUrl,
  normalizeUrlForCompare,
  storeReturnUrl,
} from "@/utils/returnUrlUtils";
import { MIGRATED_PAGES, migratedHref } from "@/utils/migratedPages";
import { useRouter, useSearchParams } from "next/navigation";
import { Suspense, useEffect, useRef } from "react";

function CreateKeyPageContent() {
  const { authLoading, token, userRole, userID } = useAuth();

  const router = useRouter();
  const searchParams = useSearchParams()!;

  const explicitPage = searchParams.get("page");

  // Track if we've already attempted a return URL redirect to prevent race conditions
  const hasAttemptedReturnRedirectRef = useRef(false);
  const didReturnRedirectRef = useRef(false);

  const redirectToLogin = authLoading === false && token === null;

  useEffect(() => {
    if (redirectToLogin) {
      // Store the current URL so we can redirect back after login
      storeReturnUrl();
      // Build login URL with return URL parameter
      const baseLoginUrl = getLoginUrl(proxyBaseUrl || "");
      const dest = buildLoginUrlWithReturn(baseLoginUrl);
      // Replace instead of assigning to avoid back-button loops
      window.location.replace(dest);
    }
  }, [redirectToLogin]);

  // Redirect legacy ?page= deep links (old bookmarks) to their path-based routes.
  const isLegacyRedirect = explicitPage !== null && explicitPage in MIGRATED_PAGES;
  useEffect(() => {
    if (!authLoading && isLegacyRedirect) {
      router.replace(migratedHref(MIGRATED_PAGES[explicitPage]));
    }
  }, [authLoading, isLegacyRedirect, explicitPage, router]);

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
    const returnUrl = consumeReturnUrl();
    if (returnUrl && isValidReturnUrl(returnUrl)) {
      // Inline origin check: only redirect to same-origin URLs to prevent open redirect.
      const safeUrl = new URL(returnUrl, window.location.origin);
      if (safeUrl.origin !== window.location.origin) {
        return;
      }
      const currentUrl = window.location.href;
      const normalizedReturnUrl = normalizeUrlForCompare(returnUrl);
      const normalizedCurrentUrl = normalizeUrlForCompare(currentUrl);
      // Only redirect if the return URL is different from the current URL
      // This prevents infinite redirect loops
      if (normalizedReturnUrl !== normalizedCurrentUrl) {
        didReturnRedirectRef.current = true;
        window.location.replace(safeUrl.href);
      }
    }
  }, [authLoading, token]);

  useEffect(() => {
    if (!token) {
      hasAttemptedReturnRedirectRef.current = false;
      didReturnRedirectRef.current = false;
    }
  }, [token]);

  const isPostLoginLanding = searchParams.get("login") === "success";
  const isSignedIn = !authLoading && Boolean(token);
  const shouldCheckForKeys = isPostLoginLanding && isSignedIn && !isAdminRole(userRole);
  const { data: keysData, isLoading: keysLoading } = useKeys(1, 1, { userID }, shouldCheckForKeys);
  const isKeylessLanding = shouldCheckForKeys && !keysLoading && keysData?.keys?.length === 0;
  const isResolvingKeylessLanding = (shouldCheckForKeys && keysLoading) || isKeylessLanding;

  useEffect(() => {
    if (isKeylessLanding && !didReturnRedirectRef.current) {
      router.replace(migratedHref("connect"));
    }
  }, [isKeylessLanding, router]);

  const isRedirecting = redirectToLogin || isLegacyRedirect || isResolvingKeylessLanding;

  if (authLoading || isRedirecting) {
    return <LoadingScreen />;
  }

  return <ApiKeysDashboard />;
}

export default function CreateKeyPage() {
  return (
    <Suspense fallback={<LoadingScreen />}>
      <CreateKeyPageContent />
    </Suspense>
  );
}
