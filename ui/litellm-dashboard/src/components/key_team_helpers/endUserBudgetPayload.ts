const metadataString = (metadata: unknown, key: string): string => {
  if (metadata === null || typeof metadata !== "object" || Array.isArray(metadata)) return "";
  const value = (metadata as Record<string, unknown>)[key];
  return typeof value === "string" ? value : "";
};

export const storedEndUserBudgetId = (metadata: unknown): string => metadataString(metadata, "end_user_budget_id");

export const keyOffersEndUserBudget = (metadata: unknown): boolean =>
  metadataString(metadata, "service_account_id") !== "" || storedEndUserBudgetId(metadata) !== "";

export const endUserBudgetIdUpdate = (selected: string | null, stored: string): string | undefined => {
  const next = selected ?? "";
  return next === stored ? undefined : next;
};
