"use client";

import { getPageDisplayName } from "@/components/page_utils";
import { isPageAccessibleForUser, type PageAccessParams } from "@/utils/page_access";
import PageAccessDenied from "./PageAccessDenied";

export interface PageAccessSettings {
  enabledPagesInternalUsers: string[] | null;
  enableProjectsUI: boolean;
  disableAgentsForInternalUsers: boolean;
  allowAgentsForTeamAdmins: boolean;
  disableVectorStoresForInternalUsers: boolean;
  allowVectorStoresForTeamAdmins: boolean;
  isLoading: boolean;
}

interface PageAccessGuardProps {
  page: string;
  userRole: string;
  teams: unknown;
  organizations: unknown;
  userId: string | null;
  pageAccessSettings: PageAccessSettings;
  onNavigateToDefault: () => void;
  children: React.ReactNode;
}

/**
 * Guards page content: when the user does not have access (e.g. proxy admin
 * hid the page from internal users), shows PageAccessDenied instead of
 * rendering children. This prevents the page component and its react-query
 * fetches from mounting.
 */
export default function PageAccessGuard({
  page,
  userRole,
  teams,
  organizations,
  userId,
  pageAccessSettings,
  onNavigateToDefault,
  children,
}: PageAccessGuardProps) {
  const { isLoading, ...settings } = pageAccessSettings;

  const accessible = isPageAccessibleForUser(page, {
    userRole,
    enabledPagesInternalUsers: settings.enabledPagesInternalUsers,
    enableProjectsUI: settings.enableProjectsUI,
    disableAgentsForInternalUsers: settings.disableAgentsForInternalUsers,
    allowAgentsForTeamAdmins: settings.allowAgentsForTeamAdmins,
    disableVectorStoresForInternalUsers: settings.disableVectorStoresForInternalUsers,
    allowVectorStoresForTeamAdmins: settings.allowVectorStoresForTeamAdmins,
    teams: teams as PageAccessParams["teams"],
    userId,
    organizations: organizations as PageAccessParams["organizations"],
  });

  if (isLoading) {
    return null;
  }

  if (!accessible) {
    return (
      <PageAccessDenied
        pageName={getPageDisplayName(page)}
        onNavigateToDefault={onNavigateToDefault}
      />
    );
  }

  return <>{children}</>;
}
