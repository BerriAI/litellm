"use client";

import { teamListCall as v2TeamListCall } from "@/app/(dashboard)/hooks/teams/useTeams";
import LoadingScreen from "@/components/common_components/LoadingScreen";
import { Team } from "@/components/key_team_helpers/key_list";
import { Organization, proxyBaseUrl } from "@/components/networking";
import { CreateKeyPrefillData } from "@/components/organisms/create_key_button";
import { fetchOrganizations } from "@/components/organizations";
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
import { Suspense, useEffect, useMemo, useRef, useState } from "react";

function CreateKeyPageContent() {
  const { authLoading, token, userID, userRole, userEmail, accessToken, premiumUser, setUserRole, setUserEmail } =
    useAuth();

  const [teams, setTeams] = useState<Team[] | null>(null);
  const [keys, setKeys] = useState<null | any[]>([]);
  const [organizations, setOrganizations] = useState<Organization[]>([]);

  const router = useRouter();
  const searchParams = useSearchParams()!;
  const [createClicked, setCreateClicked] = useState<boolean>(false);

  const invitation_id = searchParams.get("invitation_id");

  // Parse URL query parameters for pre-filling the create key form
  // Includes validation to prevent injection and DoS attacks
  const autoOpenCreate = searchParams.get("create") === "true";
  const prefillData: CreateKeyPrefillData | undefined = useMemo(() => {
    if (!autoOpenCreate) return undefined;

    const ownedBy = searchParams.get("owned_by");
    const teamId = searchParams.get("team_id");
    const keyAlias = searchParams.get("key_alias");
    const modelsParam = searchParams.get("models");
    const keyType = searchParams.get("key_type");

    // Only return prefill data if at least one field is provided
    if (!ownedBy && !teamId && !keyAlias && !modelsParam && !keyType) {
      return undefined;
    }

    // Validate owned_by against allowed values
    const validOwnedByValues = ["you", "service_account", "another_user"];
    const validatedOwnedBy =
      ownedBy && validOwnedByValues.includes(ownedBy) ? (ownedBy as CreateKeyPrefillData["owned_by"]) : undefined;

    // Validate key_type against allowed values
    const validKeyTypes = ["default", "llm_api", "management"];
    const validatedKeyType =
      keyType && validKeyTypes.includes(keyType) ? (keyType as CreateKeyPrefillData["key_type"]) : undefined;

    // Sanitize key_alias (limit length, trim whitespace)
    const sanitizedKeyAlias = keyAlias
      ? keyAlias.trim().slice(0, 256) // Reasonable max length
      : undefined;

    // Sanitize models (limit array size and individual model name length)
    const sanitizedModels = modelsParam
      ? modelsParam
          .split(",")
          .slice(0, 100) // Limit number of models to prevent DoS
          .map((m) => m.trim().slice(0, 256)) // Limit individual model name length
          .filter((m) => m.length > 0) // Remove empty strings
      : undefined;

    return {
      owned_by: validatedOwnedBy,
      team_id: teamId?.trim() || undefined,
      key_alias: sanitizedKeyAlias,
      models: sanitizedModels && sanitizedModels.length > 0 ? sanitizedModels : undefined,
      key_type: validatedKeyType,
    };
  }, [searchParams, autoOpenCreate]);

  const page = searchParams.get("page") || "api-keys";

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

  // Redirect legacy query-param pages to their new path-based routes
  const isLegacyRedirect = page in MIGRATED_PAGES;
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
    if (accessToken) {
      fetchOrganizations(accessToken, setOrganizations);
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
          organizations={organizations}
          addKey={addKey}
          createClicked={createClicked}
        />
      ) : (
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
          organizations={organizations}
          addKey={addKey}
          createClicked={createClicked}
          autoOpenCreate={autoOpenCreate}
          prefillData={prefillData}
        />
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
