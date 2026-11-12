import type { ParseKeys } from "i18next";

/**
 * Team info tab configuration and permission logic.
 * Extracted for testability - permission rules can be unit tested in isolation.
 */

export const TEAM_INFO_TAB_KEYS = {
  OVERVIEW: "overview",
  ACTIVITY: "activity",
  MY_USER: "my-user",
  VIRTUAL_KEYS: "virtual-keys",
  MEMBERS: "members",
  MEMBER_PERMISSIONS: "member-permissions",
  SETTINGS: "settings",
} as const;

export const TEAM_INFO_TAB_LABELS: Record<string, ParseKeys> = {
  [TEAM_INFO_TAB_KEYS.ACTIVITY]: "shared.resourceActivity.title",
  [TEAM_INFO_TAB_KEYS.OVERVIEW]: "teams.teamInfo.tabs.overview",
  [TEAM_INFO_TAB_KEYS.MY_USER]: "teams.teamInfo.tabs.myUser",
  [TEAM_INFO_TAB_KEYS.VIRTUAL_KEYS]: "teams.teamInfo.tabs.virtualKeys",
  [TEAM_INFO_TAB_KEYS.MEMBERS]: "teams.teamInfo.tabs.members",
  [TEAM_INFO_TAB_KEYS.MEMBER_PERMISSIONS]: "teams.teamInfo.tabs.memberPermissions",
  [TEAM_INFO_TAB_KEYS.SETTINGS]: "teams.teamInfo.tabs.settings",
};

/**
 * Returns the list of tab keys that should be visible based on permissions.
 * - Overview, My User, Virtual Keys: always visible
 * - Members, Member Permissions, Settings: only when canEditTeam is true
 */
export function getTeamInfoVisibleTabs(canEditTeam: boolean): readonly string[] {
  const baseTabs = [
    TEAM_INFO_TAB_KEYS.OVERVIEW,
    TEAM_INFO_TAB_KEYS.ACTIVITY,
    TEAM_INFO_TAB_KEYS.MY_USER,
    TEAM_INFO_TAB_KEYS.VIRTUAL_KEYS,
  ];
  if (canEditTeam) {
    return [
      ...baseTabs,
      TEAM_INFO_TAB_KEYS.MEMBERS,
      TEAM_INFO_TAB_KEYS.MEMBER_PERMISSIONS,
      TEAM_INFO_TAB_KEYS.SETTINGS,
    ];
  }
  return baseTabs;
}

/**
 * Returns the default active tab key based on permissions and edit intent.
 * - When editTeam is true and user can edit: open Settings tab
 * - Otherwise: open Overview tab
 */
export function getTeamInfoDefaultTab(editTeam: boolean, canEditTeam: boolean): string {
  if (editTeam && canEditTeam) {
    return TEAM_INFO_TAB_KEYS.SETTINGS;
  }
  return TEAM_INFO_TAB_KEYS.OVERVIEW;
}

/**
 * Checks if a specific tab should be visible based on permissions.
 */
export function isTeamInfoTabVisible(tabKey: string, canEditTeam: boolean): boolean {
  const visibleTabs = getTeamInfoVisibleTabs(canEditTeam);
  return visibleTabs.includes(tabKey);
}
