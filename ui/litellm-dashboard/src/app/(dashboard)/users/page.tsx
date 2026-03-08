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
  const { data: organizations } = useOrganizations();

  // Compute org IDs where the user is an org_admin, but only if they're NOT a proxy admin
  const orgAdminOrgIds = useMemo(() => {
    if (!userId || !organizations || !userRole) return null;
    // Proxy admins see all users — no org filtering
    if (isProxyAdminRole(userRole)) return null;

    const adminOrgIds = organizations
      .filter((org: Organization) =>
        org.members?.some((member) => member.user_id === userId && member.user_role === "org_admin")
      )
      .map((org: Organization) => org.organization_id);

    return adminOrgIds.length > 0 ? adminOrgIds : null;
  }, [userId, organizations, userRole]);

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
