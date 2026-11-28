import type { MemberPermissionsFields } from "./schema";

export const emptyMemberPermissions: MemberPermissionsFields = { team_member_permissions: [] };

export const togglePermission = (selected: string[], permission: string, checked: boolean): string[] =>
  checked ? [...selected, permission] : selected.filter((p) => p !== permission);
