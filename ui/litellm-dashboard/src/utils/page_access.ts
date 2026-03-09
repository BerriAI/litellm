/**
 * Page access utilities - determines which pages are accessible to a user
 * based on role and proxy admin UI settings (enabled_ui_pages_internal_users, etc.).
 * Must stay in sync with leftnav filterItemsByRole logic.
 */

import { menuGroups } from "@/components/leftnav";

interface MenuItemLike {
  key: string;
  page: string;
  roles?: string[];
  children?: MenuItemLike[];
}
import {
  all_admin_roles,
  internalUserRoles,
  isAdminRole,
  isUserTeamAdminForAnyTeam,
} from "./roles";
import type { Team } from "@/components/networking";

export interface PageAccessParams {
  userRole: string;
  enabledPagesInternalUsers: string[] | null | undefined;
  enableProjectsUI?: boolean;
  disableAgentsForInternalUsers?: boolean;
  allowAgentsForTeamAdmins?: boolean;
  disableVectorStoresForInternalUsers?: boolean;
  allowVectorStoresForTeamAdmins?: boolean;
  /** Teams with members_with_roles; structurally compatible with Team[] from networking */
  teams?: { members_with_roles?: unknown }[] | null;
  userId?: string | null;
  /** Orgs with members; structurally compatible with Organization[] from networking */
  organizations?: { members?: unknown }[] | null;
}

/**
 * Returns the set of page strings that are accessible to the user.
 * Matches the filter logic used in leftnav for sidebar visibility.
 */
export function getAccessiblePages(params: PageAccessParams): Set<string> {
  const {
    userRole,
    enabledPagesInternalUsers,
    enableProjectsUI = false,
    disableAgentsForInternalUsers = false,
    allowAgentsForTeamAdmins = false,
    disableVectorStoresForInternalUsers = false,
    allowVectorStoresForTeamAdmins = false,
    teams = null,
    userId = null,
    organizations = null,
  } = params;

  const isAdmin = isAdminRole(userRole);
  const isOrgAdmin =
    userId &&
    organizations?.some((org) =>
      (org.members as Array<{ user_id?: string | null; user_role: string }> | undefined)?.some(
        (m) => m.user_id === userId && m.user_role === "org_admin"
      )
    );
  const isTeamAdmin = teams && userId ? isUserTeamAdminForAnyTeam(teams as Team[], userId) : false;

  const filterItemsByRole = (items: MenuItemLike[]): MenuItemLike[] => {
    return items
      .map((item) => ({
        ...item,
        children: item.children ? filterItemsByRole(item.children) : undefined,
      }))
      .filter((item) => {
        if (item.key === "organizations") {
          const hasRoleAccess = !item.roles || item.roles.includes(userRole) || isOrgAdmin;
          if (!hasRoleAccess) return false;
          if (!isAdmin && enabledPagesInternalUsers != null) {
            return enabledPagesInternalUsers.includes(item.page);
          }
          return true;
        }

        if (item.key === "projects" && !enableProjectsUI) return false;

        if (
          !isAdmin &&
          item.key === "agents" &&
          disableAgentsForInternalUsers &&
          !(allowAgentsForTeamAdmins && isTeamAdmin)
        )
          return false;
        if (
          !isAdmin &&
          item.key === "vector-stores" &&
          disableVectorStoresForInternalUsers &&
          !(allowVectorStoresForTeamAdmins && isTeamAdmin)
        )
          return false;

        if (item.roles && !item.roles.includes(userRole)) return false;

        if (!isAdmin && enabledPagesInternalUsers != null) {
          if (item.children?.length) {
            const hasVisibleChildren = item.children.some((child) =>
              enabledPagesInternalUsers.includes(child.page)
            );
            if (hasVisibleChildren) return true;
          }
          return enabledPagesInternalUsers.includes(item.page);
        }

        return true;
      });
  };

  const pages = new Set<string>();
  for (const group of menuGroups) {
    if (group.roles && !group.roles.includes(userRole)) continue;
    const filtered = filterItemsByRole(group.items);
    for (const item of filtered) {
      if (item.page && item.page !== "tools" && item.page !== "experimental" && item.page !== "settings") {
        pages.add(item.page);
      }
      if (item.children) {
        for (const child of item.children) {
          if (child.page) pages.add(child.page);
        }
      }
    }
  }
  return pages;
}

/**
 * Returns true if the given page is accessible to the user.
 */
export function isPageAccessibleForUser(page: string, params: PageAccessParams): boolean {
  const accessible = getAccessiblePages(params);
  return accessible.has(page);
}
