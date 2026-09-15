export const normalizeCustomEntityName = (raw: string): string | null => {
  const normalized = raw.trim().toUpperCase().replace(/[\s-]+/g, "_");
  return normalized !== "" && /^[A-Z0-9_]+$/.test(normalized) ? normalized : null;
};

export const mergeCustomEntities = (supported: readonly string[], selected: readonly string[]): string[] => [
  ...new Set([...supported, ...selected]),
];
