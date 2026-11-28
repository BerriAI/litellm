export type BulkKeyAction = "budget" | "rpm" | "tpm" | "expiry" | "block" | "unblock" | "delete";
export type KeyUpdate = { max_budget: number } | { rpm_limit: number } | { tpm_limit: number } | { duration: string };

export function buildBulkKeyUpdate(action: BulkKeyAction, value: string): KeyUpdate | null {
  if (action === "expiry") return /^[1-9]\d*[smhd]$/.test(value.trim()) ? { duration: value.trim() } : null;
  if (!value.trim()) return null;
  const number = Number(value);
  if (!Number.isFinite(number) || number < 0) return null;
  if (action === "budget") return { max_budget: number };
  if (!Number.isSafeInteger(number)) return null;
  if (action === "rpm") return { rpm_limit: number };
  if (action === "tpm") return { tpm_limit: number };
  return null;
}
