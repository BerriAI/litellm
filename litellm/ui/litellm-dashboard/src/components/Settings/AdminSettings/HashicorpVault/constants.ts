import type { ParseKeys } from "i18next";

export const SENSITIVE_FIELDS = new Set(["vault_token", "approle_secret_id", "client_key"]);

export const FIELD_LABELS: Record<string, ParseKeys> = {
  vault_addr: "adminPanel.hashicorpVault.fields.vaultAddr",
  vault_namespace: "adminPanel.hashicorpVault.fields.vaultNamespace",
  vault_login_namespace: "adminPanel.hashicorpVault.fields.vaultLoginNamespace",
  vault_secret_namespace: "adminPanel.hashicorpVault.fields.vaultSecretNamespace",
  vault_mount_name: "adminPanel.hashicorpVault.fields.vaultMountName",
  vault_path_prefix: "adminPanel.hashicorpVault.fields.vaultPathPrefix",
  vault_token: "adminPanel.hashicorpVault.fields.vaultToken",
  approle_role_id: "adminPanel.hashicorpVault.fields.approleRoleId",
  approle_secret_id: "adminPanel.hashicorpVault.fields.approleSecretId",
  approle_mount_path: "adminPanel.hashicorpVault.fields.approleMountPath",
  client_cert: "adminPanel.hashicorpVault.fields.clientCert",
  client_key: "adminPanel.hashicorpVault.fields.clientKey",
  vault_cert_role: "adminPanel.hashicorpVault.fields.vaultCertRole",
};
