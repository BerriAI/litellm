export interface CustomerLogRow {
  scope: "org" | "team" | "member" | "legacy";
  id?: string;
  name: string;
}

function identityString(value: unknown): string | undefined {
  return typeof value === "string" && value.trim() ? value : undefined;
}

export function customerLogIdentity(metadata?: Record<string, unknown> | null, endUser?: string) {
  const rows: CustomerLogRow[] = (["org", "team", "member"] as const).flatMap((scope) => {
    const id = identityString(metadata?.[`user_api_key_customer_${scope}_id`]);
    const alias = identityString(metadata?.[`user_api_key_customer_${scope}_alias`]);
    const name = alias ?? id;
    return name ? [{ scope, id, name }] : [];
  });
  const legacy = identityString(endUser);
  const fallbackRows: CustomerLogRow[] = legacy ? [{ scope: "legacy", name: legacy }] : [];
  const visibleRows = rows.length ? rows : fallbackRows;
  const summary = visibleRows.filter((row) => row.scope === "org" || row.scope === "member");
  return {
    rows: visibleRows,
    summary: summary.length ? summary : visibleRows.slice(0, 1),
    recognized: Boolean(identityString(metadata?.user_api_key_customer_org_id)),
  };
}
