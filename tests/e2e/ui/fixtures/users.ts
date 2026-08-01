import {
  ADMIN_STORAGE_PATH,
  ADMIN_VIEWER_STORAGE_PATH,
  E2E_ADMIN_VIEWER_EMAIL,
  E2E_INTERNAL_USER_EMAIL,
  E2E_INTERNAL_VIEWER_EMAIL,
  E2E_TEAM_ADMIN_EMAIL,
  E2E_USER_PASSWORD,
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

export const users: Record<Role, { email: string; password: string }> = {
  [Role.ProxyAdmin]: {
    email: "admin",
    password: process.env.LITELLM_MASTER_KEY || "sk-1234",
  },
  [Role.ProxyAdminViewer]: {
    email: E2E_ADMIN_VIEWER_EMAIL,
    password: E2E_USER_PASSWORD,
  },
  [Role.InternalUser]: {
    email: E2E_INTERNAL_USER_EMAIL,
    password: E2E_USER_PASSWORD,
  },
  [Role.InternalUserViewer]: {
    email: E2E_INTERNAL_VIEWER_EMAIL,
    password: E2E_USER_PASSWORD,
  },
  [Role.TeamAdmin]: {
    email: E2E_TEAM_ADMIN_EMAIL,
    password: E2E_USER_PASSWORD,
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
