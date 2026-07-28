import type { components } from "@/lib/http/schema";

export type KeyManagementRoute = components["schemas"]["KeyManagementRoutes"];

const PERMISSION_IS_SELECTABLE: Record<KeyManagementRoute, boolean> = {
  "/key/generate": true,
  "/key/update": true,
  "/key/delete": true,
  "/key/regenerate": true,
  "/key/{key_id}/regenerate": true,
  "/key/service-account/generate": true,
  "/key/block": true,
  "/key/unblock": true,
  "/key/bulk_update": true,
  "/team/key/bulk_update": true,
  "/key/{key_id}/reset_spend": true,
  "/key/access_group_assignment": true,
  "/key/info": true,
  "/key/list": true,
  "/key/aliases": true,
  "/team/daily/activity": true,
  "/spend/logs": true,
  "/spend/logs/v2": true,
  "/key/health": false,
};

export const SELECTABLE_PERMISSIONS: readonly KeyManagementRoute[] = (
  Object.keys(PERMISSION_IS_SELECTABLE) as KeyManagementRoute[]
).filter((route) => PERMISSION_IS_SELECTABLE[route]);

export const isSelectablePermission = (value: string): value is KeyManagementRoute =>
  (SELECTABLE_PERMISSIONS as readonly string[]).includes(value);
