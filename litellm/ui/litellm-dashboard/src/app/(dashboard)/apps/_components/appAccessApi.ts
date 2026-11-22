import {
  apiClient,
  organizationInfoCall,
  teamInfoCall,
  teamUpdateCall,
  userGetInfoV2,
  userUpdateUserCall,
} from "@/components/networking";

export const GRANTEE_KINDS = ["team", "user", "organization"] as const;
export type GranteeKind = (typeof GRANTEE_KINDS)[number];

interface StoredAccess {
  object_permission?: { apps?: string[] | null } | null;
}

export const grantedAppIds = (stored: StoredAccess | null | undefined): string[] =>
  stored?.object_permission?.apps ?? [];

export const appAccessChanged = (stored: readonly string[], selected: readonly string[]): boolean =>
  [...stored].sort().join("\n") !== [...selected].sort().join("\n");

export const loadGranteeApps = async (kind: GranteeKind, id: string, accessToken: string): Promise<string[]> => {
  if (kind === "team") return grantedAppIds((await teamInfoCall(accessToken, id, { includeMembers: false })).team_info);
  if (kind === "user") return grantedAppIds(await userGetInfoV2(accessToken, id));
  return grantedAppIds(await organizationInfoCall(accessToken, id));
};

export const saveGranteeApps = async (
  kind: GranteeKind,
  id: string,
  apps: string[],
  accessToken: string,
): Promise<void> => {
  const object_permission = { apps };
  if (kind === "team") await teamUpdateCall(accessToken, { team_id: id, object_permission });
  else if (kind === "user") await userUpdateUserCall(accessToken, { user_id: id, object_permission }, null);
  else await apiClient.patch(`/organization/update`, { accessToken, body: { organization_id: id, object_permission } });
};
