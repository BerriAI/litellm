"use client";

import ViewUserDashboard from "@/components/view_users";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import useTeams from "@/app/(dashboard)/hooks/useTeams";
import { useOrganizations } from "@/app/(dashboard)/hooks/organizations/useOrganizations";
import { isProxyAdminRole } from "@/utils/roles";
import { useState, useMemo } from "react";
import { Organization } from "@/components/networking";

const UsersPage = () => {
  const { accessToken, userRole, userId, token } = useAuthorized();
  const [keys, setKeys] = useState<null | any[]>([]);

  const { teams } = useTeams();
  const { data: organizations, isLoading: isOrgsLoading } = useOrganizations();

  // Three states:
  // - undefined: org data still loading (non-proxy-admin) — query should wait
  // - null: proxy admin or no org filtering needed — query runs unfiltered
  // - Array<{organization_id, organization_alias}>: org admin orgs — query runs filtered
  const orgAdminOrgIds = useMemo((): Array<{organization_id: string, organization_alias: string}> | null | undefined => {
    if (!userId || !userRole) return null;
    // Proxy admins see all users — no org filtering
    if (isProxyAdminRole(userRole)) return null;

    // Still loading org data — signal "not ready yet"
    if (isOrgsLoading || !organizations) return undefined;

    const adminOrgs = organizations
      .filter((org: Organization) =>
        org.members?.some((member) => member.user_id === userId && member.user_role === "org_admin")
      )
      .map((org: Organization) => ({ organization_id: org.organization_id, organization_alias: org.organization_alias }));

    return adminOrgs.length > 0 ? adminOrgs : null;
  }, [userId, organizations, userRole, isOrgsLoading]);

  return (
    <ViewUserDashboard
      accessToken={accessToken}
      token={token}
      keys={keys}
      userRole={userRole}
      userID={userId}
      teams={teams as any}
      setKeys={setKeys}
      orgAdminOrgIds={orgAdminOrgIds}
    />
  );
};

export default UsersPage;
