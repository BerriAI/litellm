import type { components } from "@/lib/http/schema";

export type TeamMemberPermission = components["schemas"]["TeamMemberPermissions"];

const PERMISSION_IS_SELECTABLE: Record<TeamMemberPermission, boolean> = {
  "/key/generate": true,
  "/key/update": true,
  "/key/delete": true,
  "/key/regenerate": true,
  "/key/service-account/generate": true,
  "/key/access_group_assignment": true,
  "/key/list": true,
  "/team/daily/activity": true,
  "/spend/logs": true,
  "/key/info": false,
  "/key/health": false,
};

export const SELECTABLE_PERMISSIONS: readonly TeamMemberPermission[] = (
  Object.keys(PERMISSION_IS_SELECTABLE) as TeamMemberPermission[]
).filter((route) => PERMISSION_IS_SELECTABLE[route]);

export const isSelectablePermission = (value: string): value is TeamMemberPermission =>
  (SELECTABLE_PERMISSIONS as readonly string[]).includes(value);
