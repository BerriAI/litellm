export const SENSITIVE_FIELDS = new Set(["cyberark_api_key", "client_key"]);

export const FIELD_LABELS: Record<string, string> = {
  cyberark_api_base: "Conjur Server URL",
  cyberark_account: "Account",
  cyberark_username: "Username",
  cyberark_api_key: "API Key",
  client_cert: "Client Certificate",
  client_key: "Client Key",
  ssl_verify: "SSL Verification",
  refresh_interval: "Token Refresh Interval (seconds)",
};
