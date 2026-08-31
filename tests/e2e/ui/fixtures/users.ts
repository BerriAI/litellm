import {
  ADMIN_STORAGE_PATH,
  ADMIN_VIEWER_STORAGE_PATH,
  INTERNAL_USER_STORAGE_PATH,
  INTERNAL_VIEWER_STORAGE_PATH,
  TEAM_ADMIN_STORAGE_PATH,
} from "../constants";

export enum Role {
  ProxyAdmin = "proxy_admin",
  ProxyAdminViewer = "proxy_admin_viewer",
  InternalUser = "internal_user",
  InternalUserViewer = "internal_user_viewer",
  TeamAdmin = "team_admin",
}

export type SeedApiRole = "proxy_admin_viewer" | "internal_user" | "internal_user_viewer";

export const users: Record<Role, { email: string; password: string; seedApiRole?: SeedApiRole }> = {
  [Role.ProxyAdmin]: {
    email: "admin",
    password: process.env.LITELLM_MASTER_KEY || "sk-1234",
  },
  [Role.ProxyAdminViewer]: {
    email: "adminviewer@test.local",
    password: "test",
    seedApiRole: "proxy_admin_viewer",
  },
  [Role.InternalUser]: {
    email: "internal@test.local",
    password: "test",
    seedApiRole: "internal_user",
  },
  [Role.InternalUserViewer]: {
    email: "viewer@test.local",
    password: "test",
    seedApiRole: "internal_user_viewer",
  },
  [Role.TeamAdmin]: {
    email: "teamadmin@test.local",
    password: "test",
    seedApiRole: "internal_user",
  },
};

// Re-exported from constants so the paths have one definition; they must honor
// ARTIFACT_DIR, since the suite runs from a read-only cwd in the e2e image.
export const STORAGE_PATHS: Record<Role, string> = {
  [Role.ProxyAdmin]: ADMIN_STORAGE_PATH,
  [Role.ProxyAdminViewer]: ADMIN_VIEWER_STORAGE_PATH,
  [Role.InternalUser]: INTERNAL_USER_STORAGE_PATH,
  [Role.InternalUserViewer]: INTERNAL_VIEWER_STORAGE_PATH,
  [Role.TeamAdmin]: TEAM_ADMIN_STORAGE_PATH,
};
