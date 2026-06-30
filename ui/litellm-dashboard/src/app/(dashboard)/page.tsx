"use client";

import ApiKeysDashboard from "@/app/(dashboard)/api-keys/ApiKeysDashboard";
import { teamListCall as v2TeamListCall } from "@/app/(dashboard)/hooks/teams/useTeams";
import LoadingScreen from "@/components/common_components/LoadingScreen";
import { Team } from "@/components/key_team_helpers/key_list";
import { proxyBaseUrl } from "@/components/networking";
import UserDashboard from "@/components/user_dashboard";
import { useAuth } from "@/contexts/AuthContext";
import {
  buildLoginUrlWithReturn,
  consumeReturnUrl,
  isValidReturnUrl,
  normalizeUrlForCompare,
  storeReturnUrl,
} from "@/utils/returnUrlUtils";
import { MIGRATED_PAGES, migratedHref } from "@/utils/migratedPages";
import { useRouter, useSearchParams } from "next/navigation";
import { Suspense, useEffect, useRef, useState } from "react";

function CreateKeyPageContent() {
  const { authLoading, token, userID, userRole, userEmail, accessToken, premiumUser, setUserRole, setUserEmail } =
    useAuth();

  const [teams, setTeams] = useState<Team[] | null>(null);
  const [keys, setKeys] = useState<null | any[]>([]);

  const router = useRouter();
  const searchParams = useSearchParams()!;
  const [createClicked, setCreateClicked] = useState<boolean>(false);

  const invitation_id = searchParams.get("invitation_id");

  const explicitPage = searchParams.get("page");
  const page = explicitPage || "api-keys";

  // Track if we've already attempted a return URL redirect to prevent race conditions
  const hasAttemptedReturnRedirectRef = useRef(false);

  const addKey = (data: any) => {
    setKeys((prevData) => (prevData ? [...prevData, data] : [data]));
    setCreateClicked(() => !createClicked);
  };
  const redirectToLogin = authLoading === false && token === null && invitation_id === null;

  useEffect(() => {
    if (redirectToLogin) {
      // Store the current URL so we can redirect back after login
      storeReturnUrl();
      // Build login URL with return URL parameter
      const baseLoginUrl = (proxyBaseUrl || "") + "/ui/login";
      const dest = buildLoginUrlWithReturn(baseLoginUrl);
      // Replace instead of assigning to avoid back-button loops
      window.location.replace(dest);
    }
  }, [redirectToLogin]);

  // Redirect legacy query-param pages to their new path-based routes. Only when the page is
  // explicitly requested via ?page=, so the bare landing renders inline and the post-login
  // return-URL handling below stays intact.
  const isLegacyRedirect = explicitPage !== null && explicitPage in MIGRATED_PAGES;
  useEffect(() => {
    if (!authLoading && isLegacyRedirect) {
      router.replace(migratedHref(MIGRATED_PAGES[page]));
    }
  }, [authLoading, isLegacyRedirect, page, router]);

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
        window.location.replace(safeUrl.href);
      }
    }
  }, [authLoading, token]);

  useEffect(() => {
    if (!token) {
      hasAttemptedReturnRedirectRef.current = false;
    }
  }, [token]);

  useEffect(() => {
    if (accessToken && userID && userRole) {
      v2TeamListCall(accessToken, 1, 100, {
        userID: userRole !== "Admin" && userRole !== "Admin Viewer" ? userID : null,
      })
        .then((response) => setTeams(response.teams ?? []))
        .catch(console.error);
    }
  }, [accessToken, userID, userRole]);

  if (authLoading || redirectToLogin || isLegacyRedirect) {
    return <LoadingScreen />;
  }

  return (
    <>
      {invitation_id ? (
        <UserDashboard
          userID={userID}
          userRole={userRole}
          premiumUser={premiumUser}
          teams={teams}
          keys={keys}
          setUserRole={setUserRole}
          userEmail={userEmail}
          setUserEmail={setUserEmail}
          setTeams={setTeams}
          setKeys={setKeys}
          addKey={addKey}
          createClicked={createClicked}
        />
      ) : (
        <ApiKeysDashboard />
      )}
    </>
  );
}

export default function CreateKeyPage() {
  return (
    <Suspense fallback={<LoadingScreen />}>
      <CreateKeyPageContent />
    </Suspense>
  );
}
