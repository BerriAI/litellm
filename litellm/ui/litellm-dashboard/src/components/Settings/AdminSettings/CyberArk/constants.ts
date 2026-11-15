import type { ParseKeys } from "i18next";

export const SENSITIVE_FIELDS = new Set(["cyberark_api_key", "client_key"]);

export const FIELD_LABELS: Record<string, ParseKeys> = {
  cyberark_api_base: "adminPanel.cyberArk.fields.apiBase",
  cyberark_account: "adminPanel.cyberArk.fields.account",
  cyberark_username: "adminPanel.cyberArk.fields.username",
  cyberark_api_key: "adminPanel.cyberArk.fields.apiKey",
  client_cert: "adminPanel.cyberArk.fields.clientCert",
  client_key: "adminPanel.cyberArk.fields.clientKey",
  ssl_verify: "adminPanel.cyberArk.fields.sslVerify",
  refresh_interval: "adminPanel.cyberArk.fields.refreshInterval",
};
