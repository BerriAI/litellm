// PROTOTYPE: localStorage-backed *global* per-user variables store, plus the
// mocked "credential store backend" admin config (HashiCorp Vault, AWS Secrets
// Manager, local DB, etc.).
//
// In the real implementation:
//   - Variables would be looked up by virtual API key → internal user → user
//     variables in the configured credential store.
//   - Saving would write an opaque encrypted JSON blob to the backend, keyed
//     by user id. The admin/user *never* sees the plaintext after save.
//
// For the prototype we just persist values in localStorage and simulate the
// "write-only" property by tracking which variables have known plaintext in
// the current session (a `Set<string>` of variable names that were touched).

export interface UserVariableEntry {
  name: string;
  // Empty string in storage means "set but unreadable" — the value lives in
  // the credential store and was never returned to the client. We treat any
  // present-but-empty entry as a "stored secret" for display purposes.
  value: string;
  // ISO timestamp of last save, surfaced in the UI as "Last updated …".
  updated_at: string;
}

export type CredentialStoreProvider =
  | "litellm_db"
  | "hashicorp_vault"
  | "aws_secrets_manager"
  | "gcp_secret_manager"
  | "azure_key_vault";

export interface CredentialStoreConfig {
  provider: CredentialStoreProvider;
  // Provider-specific extras shown read-only on the Variables tab. The values
  // here would come from config.yaml or env vars in the real implementation.
  hashicorp?: {
    address: string;
    mount_path: string;
    namespace?: string;
    auth_method: "token" | "approle" | "kubernetes";
  };
  aws?: { region: string; prefix: string };
  gcp?: { project_id: string };
  azure?: { vault_url: string };
}

const VARS_KEY = (userId: string) => `mock-mcp-user-vars::${userId}`;
const REVEALED_KEY = (userId: string) => `mock-mcp-user-vars-revealed::${userId}`;
const STORE_KEY = "mock-mcp-credential-store::v1";
const CHANGE_EVENT = "mock-mcp-user-vars-changed";

// ---------------------------------------------------------------------------
// per-user variables
// ---------------------------------------------------------------------------

export function listUserVariables(userId: string): UserVariableEntry[] {
  if (!userId || typeof window === "undefined") return [];
  try {
    const raw = window.localStorage.getItem(VARS_KEY(userId));
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

export function saveUserVariables(
  userId: string,
  entries: UserVariableEntry[],
): void {
  if (!userId || typeof window === "undefined") return;
  window.localStorage.setItem(VARS_KEY(userId), JSON.stringify(entries));
  window.dispatchEvent(new CustomEvent(CHANGE_EVENT));
}

// Track which variable names have a *known* plaintext value in the current
// session. Persisted in localStorage so a tab refresh on the Variables page
// itself doesn't lose your in-progress edits, but in a real impl this would
// live in memory only since we genuinely don't have the value back.
export function markVariableRevealed(userId: string, name: string): void {
  const set = new Set(getRevealedSet(userId));
  set.add(name);
  if (typeof window === "undefined") return;
  window.localStorage.setItem(
    REVEALED_KEY(userId),
    JSON.stringify(Array.from(set)),
  );
}

export function getRevealedSet(userId: string): string[] {
  if (!userId || typeof window === "undefined") return [];
  try {
    const raw = window.localStorage.getItem(REVEALED_KEY(userId));
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

export function clearRevealed(userId: string): void {
  if (!userId || typeof window === "undefined") return;
  window.localStorage.removeItem(REVEALED_KEY(userId));
}

// Simulates the page reload effect — wipes "known plaintext" markers so all
// existing variables go back to the unknown-length dot display.
export function simulateReload(userId: string): void {
  clearRevealed(userId);
  window.dispatchEvent(new CustomEvent(CHANGE_EVENT));
}

export function subscribeUserVariablesChanged(handler: () => void): () => void {
  if (typeof window === "undefined") return () => undefined;
  window.addEventListener(CHANGE_EVENT, handler);
  return () => {
    window.removeEventListener(CHANGE_EVENT, handler);
  };
}

// ---------------------------------------------------------------------------
// credential store backend (admin config)
// ---------------------------------------------------------------------------

const DEFAULT_STORE: CredentialStoreConfig = {
  provider: "litellm_db",
};

export function getCredentialStoreConfig(): CredentialStoreConfig {
  if (typeof window === "undefined") return DEFAULT_STORE;
  try {
    const raw = window.localStorage.getItem(STORE_KEY);
    if (!raw) return DEFAULT_STORE;
    const parsed = JSON.parse(raw);
    return parsed && typeof parsed === "object" ? parsed : DEFAULT_STORE;
  } catch {
    return DEFAULT_STORE;
  }
}

export function saveCredentialStoreConfig(cfg: CredentialStoreConfig): void {
  if (typeof window === "undefined") return;
  window.localStorage.setItem(STORE_KEY, JSON.stringify(cfg));
  window.dispatchEvent(new CustomEvent(CHANGE_EVENT));
}

export const CREDENTIAL_STORE_LABELS: Record<CredentialStoreProvider, string> = {
  litellm_db: "LiteLLM DB (default)",
  hashicorp_vault: "HashiCorp Vault",
  aws_secrets_manager: "AWS Secrets Manager",
  gcp_secret_manager: "GCP Secret Manager",
  azure_key_vault: "Azure Key Vault",
};
