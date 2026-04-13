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
  [Role.ProxyAdmin]: "admin.storageState.json",
  [Role.ProxyAdminViewer]: "adminViewer.storageState.json",
  [Role.InternalUser]: "internalUser.storageState.json",
  [Role.InternalUserViewer]: "internalViewer.storageState.json",
  [Role.TeamAdmin]: "teamAdmin.storageState.json",
};
