export function withNullsForRemovedKeys(
  previous: Record<string, unknown> | null | undefined,
  next: Record<string, unknown>,
  skip?: ReadonlySet<string>,
): Record<string, unknown> {
  if (!previous) {
    return { ...next };
  }

  const result: Record<string, unknown> = { ...next };
  for (const key of Object.keys(previous)) {
    if (skip?.has(key)) {
      continue;
    }
    if (!(key in result)) {
      result[key] = null;
    }
  }
  return result;
}

export const PROTECTED_MODEL_INFO_KEYS = new Set([
  "id",
  "db_model",
  "team_id",
  "team_public_model_name",
  "created_at",
  "created_by",
  "updated_at",
  "updated_by",
]);
