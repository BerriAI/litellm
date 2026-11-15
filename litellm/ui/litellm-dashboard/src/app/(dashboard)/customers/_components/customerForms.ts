import type { components } from "@/lib/http/schema";

export type NewCustomerBody = components["schemas"]["NewCustomerRequest"];
export type UpdateCustomerBody = components["schemas"]["UpdateCustomerRequest"];

export const digitsOnly = (s: string) => s.replace(/[^0-9]/g, "");
export const decimalOnly = (s: string) => s.replace(/[^0-9.]/g, "");

const numberOrNull = (s: string): number | null => (s.trim() === "" ? null : Number(s));

export interface CreateCustomerFields {
  userId: string;
  alias: string;
  budget: string;
  tpm: string;
  rpm: string;
}

export const buildCreateCustomerBody = (f: CreateCustomerFields): NewCustomerBody | null => {
  const userId = f.userId.trim();
  if (!userId) return null;
  const maxBudget = numberOrNull(f.budget);
  return {
    user_id: userId,
    alias: f.alias.trim() || null,
    blocked: false,
    max_budget: maxBudget,
    budget_duration: maxBudget === null ? null : "1mo",
    tpm_limit: numberOrNull(f.tpm),
    rpm_limit: numberOrNull(f.rpm),
  };
};

export interface EditCustomerFields {
  userId: string;
  alias: string;
  budget: string;
  blocked: boolean;
}

export const buildUpdateCustomerBody = (f: EditCustomerFields): UpdateCustomerBody => ({
  user_id: f.userId,
  alias: f.alias.trim() || null,
  blocked: f.blocked,
  max_budget: numberOrNull(f.budget),
});
