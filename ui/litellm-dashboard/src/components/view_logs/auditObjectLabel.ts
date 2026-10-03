import type { AuditLogEntry } from "./AuditLogsTableColumns";

const NAME_FIELD_BY_TABLE: Record<string, string> = {
  LiteLLM_ProxyModelTable: "model_name",
  LiteLLM_VerificationToken: "key_alias",
  LiteLLM_TeamTable: "team_alias",
  LiteLLM_UserTable: "user_email",
  LiteLLM_OrganizationTable: "organization_alias",
};

const readName = (value: unknown, field: string): string | null => {
  if (typeof value !== "object" || value === null) return null;
  const name = (value as Record<string, unknown>)[field];
  return typeof name === "string" && name.trim() !== "" ? name : null;
};

export const getAuditObjectName = (log: Pick<AuditLogEntry, "table_name" | "before_value" | "updated_values">) => {
  const field = NAME_FIELD_BY_TABLE[log.table_name];
  if (field == null) return null;
  return readName(log.updated_values, field) ?? readName(log.before_value, field);
};
