"use client";

import ModelsAndEndpointsView from "@/app/(dashboard)/models-and-endpoints/ModelsAndEndpointsView";
import AdminPanel from "@/components/AdminPanel";
import AgentsPanel from "@/components/agents";
import ClaudeCodePluginsPanel from "@/components/claude_code_plugins";
import { teamListCall as v2TeamListCall } from "@/app/(dashboard)/hooks/teams/useTeams";
import { useUISettings } from "@/app/(dashboard)/hooks/uiSettings/useUISettings";
import useProxySettings from "@/app/(dashboard)/hooks/proxySettings/useProxySettings";
import LoadingScreen from "@/components/common_components/LoadingScreen";
import GeneralSettings from "@/components/general_settings";
import GuardrailsPanel from "@/components/guardrails";
import PoliciesPanel from "@/components/policies";
import { Team } from "@/components/key_team_helpers/key_list";
import ModelHubTable from "@/components/AIHub/ModelHubTable";
import { Organization, proxyBaseUrl, getInProductNudgesCall } from "@/components/networking";
import NewUsagePage from "@/components/UsagePage/components/UsagePageView";
import OldTeams from "@/components/OldTeams";
import { fetchUserModels, CreateKeyPrefillData } from "@/components/organisms/create_key_button";
import Organizations, { fetchOrganizations } from "@/components/organizations";
import PassThroughSettings from "@/components/pass_through_settings";
import PromptsPanel from "@/components/prompts";
import PublicModelHub from "@/components/public_model_hub";
import Settings from "@/components/settings";
import { SurveyPrompt, SurveyModal, ClaudeCodePrompt, ClaudeCodeModal } from "@/components/survey";
import Usage from "@/components/usage";
import UserDashboard from "@/components/user_dashboard";
import ToolPoliciesView from "@/components/ToolPoliciesView";
import ViewUserDashboard from "@/components/view_users";
import { useAuth } from "@/contexts/AuthContext";
import {
  buildLoginUrlWithReturn,
  consumeReturnUrl,
  isValidReturnUrl,
  normalizeUrlForCompare,
  storeReturnUrl,
} from "@/utils/returnUrlUtils";
import { isAdminRole } from "@/utils/roles";
import { MIGRATED_PAGES, migratedHref } from "@/utils/migratedPages";
import { useRouter, useSearchParams } from "next/navigation";
import { Suspense, useEffect, useMemo, useRef, useState } from "react";

function CreateKeyPageContent() {
  const { authLoading, token, userID, userRole, userEmail, accessToken, premiumUser, setUserRole, setUserEmail } =
    useAuth();

  const [teams, setTeams] = useState<Team[] | null>(null);
  const [keys, setKeys] = useState<null | any[]>([]);
  const [organizations, setOrganizations] = useState<Organization[]>([]);
  const [userModels, setUserModels] = useState<string[]>([]);
  const proxySettings = useProxySettings(accessToken);

  const router = useRouter();
  const searchParams = useSearchParams()!;
  const [modelData, setModelData] = useState<any>({ data: [] });
  const [createClicked, setCreateClicked] = useState<boolean>(false);

  const { data: uiSettingsData, isLoading: uiSettingsLoading } = useUISettings();
  const nudgesDisabled = uiSettingsLoading || Boolean(uiSettingsData?.values?.disable_ui_nudges);

  // Survey state - always show by default
  const [showSurveyPrompt, setShowSurveyPrompt] = useState(true);
  const [showSurveyModal, setShowSurveyModal] = useState(false);

  // Claude Code feedback state
  const [isClaudeCode, setIsClaudeCode] = useState(false);
  const [showClaudeCodePrompt, setShowClaudeCodePrompt] = useState(false);
  const [showClaudeCodeModal, setShowClaudeCodeModal] = useState(false);

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
      fetchUserModels(userID, userRole, accessToken, setUserModels);
    }
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

  // Fetch in-product nudges configuration from backend
  useEffect(() => {
    if (nudgesDisabled) {
      return;
    }
    if (accessToken && token) {
      (async () => {
        try {
          const nudgesConfig = await getInProductNudgesCall(accessToken);
          const isUsingClaudeCode = nudgesConfig?.is_claude_code_enabled || false;
          setIsClaudeCode(isUsingClaudeCode);

          // Show Claude Code prompt on login if enabled
          if (isUsingClaudeCode) {
            setShowClaudeCodePrompt(true);
            // Don't show the regular survey prompt if showing Claude Code prompt
            setShowSurveyPrompt(false);
          }
        } catch (error) {
          console.error("Failed to fetch in-product nudges:", error);
          // Silently fail and don't show Claude Code nudge
        }
      })();
    }
  }, [accessToken, token, nudgesDisabled]);

  // Auto-dismiss survey prompt after 15 seconds
  useEffect(() => {
    if (showSurveyPrompt && !showSurveyModal) {
      const timer = setTimeout(() => {
        setShowSurveyPrompt(false);
      }, 15000);
      return () => clearTimeout(timer);
    }
  }, [showSurveyPrompt, showSurveyModal]);

  // Auto-dismiss Claude Code prompt after 15 seconds
  useEffect(() => {
    if (showClaudeCodePrompt && !showClaudeCodeModal) {
      const timer = setTimeout(() => {
        setShowClaudeCodePrompt(false);
      }, 15000);
      return () => clearTimeout(timer);
    }
  }, [showClaudeCodePrompt, showClaudeCodeModal]);

  const handleOpenSurvey = () => {
    setShowSurveyPrompt(false);
    setShowSurveyModal(true);
  };

  const handleDismissSurveyPrompt = () => {
    setShowSurveyPrompt(false);
  };

  const handleSurveyComplete = () => {
    setShowSurveyModal(false);
  };

  const handleSurveyModalClose = () => {
    // If they close the modal without completing, show the prompt again
    setShowSurveyModal(false);
    setShowSurveyPrompt(true);
  };

  const handleOpenClaudeCode = () => {
    setShowClaudeCodePrompt(false);
    setShowClaudeCodeModal(true);
  };

  const handleDismissClaudeCodePrompt = () => {
    setShowClaudeCodePrompt(false);
  };

  const handleClaudeCodeComplete = () => {
    setShowClaudeCodeModal(false);
  };

  const handleClaudeCodeModalClose = () => {
    // If they close the modal without completing, show the prompt again
    setShowClaudeCodeModal(false);
    setShowClaudeCodePrompt(true);
  };

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
        <>
          {page == "api-keys" ? (
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
          ) : page == "models" ? (
            <ModelsAndEndpointsView
              token={token}
              keys={keys}
              modelData={modelData}
              setModelData={setModelData}
              premiumUser={premiumUser}
              teams={teams}
            />
          ) : page == "users" ? (
            <ViewUserDashboard
              userID={userID}
              userRole={userRole}
              token={token}
              keys={keys}
              teams={teams}
              accessToken={accessToken}
              setKeys={setKeys}
            />
          ) : page == "teams" ? (
            <OldTeams
              teams={teams}
              setTeams={setTeams}
              accessToken={accessToken}
              userID={userID}
              userRole={userRole}
              organizations={organizations}
              premiumUser={premiumUser}
              searchParams={searchParams}
            />
          ) : page == "organizations" ? (
            <Organizations
              organizations={organizations}
              setOrganizations={setOrganizations}
              userModels={userModels}
              accessToken={accessToken}
              userRole={userRole}
              premiumUser={premiumUser}
            />
          ) : page == "admin-panel" ? (
            <AdminPanel proxySettings={proxySettings} />
          ) : page == "logging-and-alerts" ? (
            <Settings userID={userID} userRole={userRole} accessToken={accessToken} premiumUser={premiumUser} />
          ) : page == "guardrails" ? (
            <GuardrailsPanel accessToken={accessToken} userRole={userRole} />
          ) : page == "policies" ? (
            <PoliciesPanel accessToken={accessToken} userRole={userRole} />
          ) : page == "agents" ? (
            <AgentsPanel accessToken={accessToken} userRole={userRole} teams={teams} />
          ) : page == "prompts" ? (
            <PromptsPanel accessToken={accessToken} userRole={userRole} />
          ) : page == "router-settings" ? (
            <GeneralSettings userID={userID} userRole={userRole} accessToken={accessToken} modelData={modelData} />
          ) : page == "model-hub-table" ? (
            isAdminRole(userRole) ? (
              <ModelHubTable
                accessToken={accessToken}
                publicPage={false}
                premiumUser={premiumUser}
                userRole={userRole}
              />
            ) : (
              <PublicModelHub accessToken={accessToken} isEmbedded={true} />
            )
          ) : page == "pass-through-settings" ? (
            <PassThroughSettings
              userID={userID}
              userRole={userRole}
              accessToken={accessToken}
              modelData={modelData}
              premiumUser={premiumUser}
            />
          ) : page == "skills" || page == "claude-code-plugins" ? (
            <ClaudeCodePluginsPanel accessToken={accessToken} userRole={userRole} />
          ) : page == "tool-policies" ? (
            <ToolPoliciesView accessToken={accessToken} userRole={userRole} />
          ) : page == "new_usage" ? (
            <NewUsagePage teams={(teams as Team[]) ?? []} organizations={(organizations as Organization[]) ?? []} />
          ) : (
            <Usage
              userID={userID}
              userRole={userRole}
              token={token}
              accessToken={accessToken}
              keys={keys}
              premiumUser={premiumUser}
            />
          )}

          {/* Survey Components */}
          <SurveyPrompt
            isVisible={showSurveyPrompt && !nudgesDisabled}
            onOpen={handleOpenSurvey}
            onDismiss={handleDismissSurveyPrompt}
          />
          <SurveyModal isOpen={showSurveyModal} onClose={handleSurveyModalClose} onComplete={handleSurveyComplete} />

          {/* Claude Code Components */}
          <ClaudeCodePrompt
            isVisible={showClaudeCodePrompt && !nudgesDisabled}
            onOpen={handleOpenClaudeCode}
            onDismiss={handleDismissClaudeCodePrompt}
          />
          <ClaudeCodeModal
            isOpen={showClaudeCodeModal}
            onClose={handleClaudeCodeModalClose}
            onComplete={handleClaudeCodeComplete}
          />
        </>
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
