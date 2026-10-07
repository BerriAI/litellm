import { useUISettings } from "@/app/(dashboard)/hooks/uiSettings/useUISettings";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { parseTeamAdminEditableFields } from "@/components/team/teamAdminEditAccess";
import { all_admin_roles, internalUserRoles, isAdminRole, isProxyAdminRole } from "@/utils/roles";

const TEAM_ADMIN_PROJECTS_PERMISSION = "projects";

export const projectReaderRoles: readonly string[] = [...all_admin_roles, "Org Admin", ...internalUserRoles];

export const canReadProjects = (userRole: string | null): boolean => projectReaderRoles.includes(userRole ?? "");

export interface ProjectsPageViewer {
  readonly userRole: string;
  readonly isOrgAdmin: boolean;
  readonly isTeamAdmin: boolean;
}

export const canViewProjectsPage = ({ userRole, isOrgAdmin, isTeamAdmin }: ProjectsPageViewer): boolean =>
  canReadProjects(userRole) && (isAdminRole(userRole) || isOrgAdmin || isTeamAdmin);

export interface ProjectManager {
  readonly userRole: string;
  readonly isViewOnly: boolean;
  readonly isTeamAdmin: boolean;
  readonly teamAdminEditableFields: readonly string[];
}

export const canManageProjects = ({
  userRole,
  isViewOnly,
  isTeamAdmin,
  teamAdminEditableFields,
}: ProjectManager): boolean =>
  !isViewOnly &&
  (isProxyAdminRole(userRole) || (isTeamAdmin && teamAdminEditableFields.includes(TEAM_ADMIN_PROJECTS_PERMISSION)));

export const useCanManageProjects = (isTeamAdmin: boolean): boolean => {
  const { userRole, isViewOnly } = useAuthorized();
  const { data: uiSettings } = useUISettings();
  return canManageProjects({
    userRole,
    isViewOnly,
    isTeamAdmin,
    teamAdminEditableFields: parseTeamAdminEditableFields(uiSettings?.values),
  });
};
