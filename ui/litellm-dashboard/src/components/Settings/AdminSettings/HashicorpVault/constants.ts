export const SENSITIVE_FIELDS = new Set([
  "vault_token",
  "approle_role_id",
  "approle_secret_id",
  "client_key",
]);

export const FIELD_LABELS: Record<string, string> = {
  vault_addr: "Vault Address",
  vault_namespace: "Namespace",
  vault_mount_name: "KV Mount Name",
  vault_path_prefix: "Path Prefix",
  vault_token: "Token",
  approle_role_id: "Role ID",
  approle_secret_id: "Secret ID",
  approle_mount_path: "Mount Path",
  client_cert: "Client Certificate",
  client_key: "Client Key",
  vault_cert_role: "Certificate Role",
};
