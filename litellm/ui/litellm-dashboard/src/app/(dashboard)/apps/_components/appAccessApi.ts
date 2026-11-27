import {
  apiClient,
  organizationInfoCall,
  teamInfoCall,
  teamUpdateCall,
  userGetInfoV2,
  userUpdateUserCall,
} from "@/components/networking";
import { grantedAppIds, granteeAppsWrite } from "./internalAccessMapper";

export { grantedAppIds } from "./internalAccessMapper";

export const GRANTEE_KINDS = ["team", "user", "organization"] as const;
export type GranteeKind = (typeof GRANTEE_KINDS)[number];

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
  const write = granteeAppsWrite(kind, id, apps);
  if (write.kind === "team") await teamUpdateCall(accessToken, write.body);
  else if (write.kind === "user") await userUpdateUserCall(accessToken, write.body, null);
  else await apiClient.patch(`/organization/update`, { accessToken, body: write.body });
};
