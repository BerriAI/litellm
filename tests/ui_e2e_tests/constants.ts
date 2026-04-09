export const ADMIN_STORAGE_PATH = "admin.storageState.json";

// Page enum — maps to ?page= query parameter values in the UI
export enum Page {
  ApiKeys = "api-keys",
  Teams = "teams",
  AdminSettings = "settings",
}

// Test user credentials — all users have password "test" (hashed in seed.sql)
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
    email: "adminviewer@test.local",
    password: "test",
  },
  [Role.InternalUser]: {
    email: "internal@test.local",
    password: "test",
  },
  [Role.InternalUserViewer]: {
    email: "viewer@test.local",
    password: "test",
  },
  [Role.TeamAdmin]: {
    email: "teamadmin@test.local",
    password: "test",
  },
};
