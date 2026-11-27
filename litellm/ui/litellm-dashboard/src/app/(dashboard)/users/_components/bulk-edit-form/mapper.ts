interface BulkUserEditValues {
  user_role?: string | null;
  max_budget?: number | string | null;
  models?: string[] | null;
  budget_duration?: string | null;
  metadata?: unknown;
}

export const mapBulkUserEdit = ({ user_role, max_budget, models, budget_duration, metadata }: BulkUserEditValues) => ({
  ...(user_role ? { user_role } : {}),
  ...(max_budget !== null && max_budget !== undefined ? { max_budget } : {}),
  ...(models && models.length > 0 ? { models } : {}),
  ...(budget_duration ? { budget_duration } : {}),
  ...(metadata && Object.keys(metadata).length > 0 ? { metadata } : {}),
});
