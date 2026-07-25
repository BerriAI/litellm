import { STATE_DIR } from "../constants";

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

export const STORAGE_PATHS: Record<Role, string> = {
  [Role.ProxyAdmin]: `${STATE_DIR}/admin.storageState.json`,
  [Role.ProxyAdminViewer]: `${STATE_DIR}/adminViewer.storageState.json`,
  [Role.InternalUser]: `${STATE_DIR}/internalUser.storageState.json`,
  [Role.InternalUserViewer]: `${STATE_DIR}/internalViewer.storageState.json`,
  [Role.TeamAdmin]: `${STATE_DIR}/teamAdmin.storageState.json`,
};
