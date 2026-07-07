import type { components } from "@/lib/http/schema";

export type OrganizationUpdateV2Body = components["schemas"]["OrganizationUpdateRequestV2"];

export interface OrgSettingsFormValues {
  organization_alias?: string;
  models?: string[];
  tpm_limit?: number | string | null;
  rpm_limit?: number | string | null;
  max_budget?: number | string | null;
  budget_duration?: string | null;
  metadata?: string;
  vector_stores?: string[];
  mcp_servers_and_groups?: { servers?: string[]; accessGroups?: string[] };
}

export class OrgMetadataParseError extends Error {
  constructor(message = "Metadata must be a valid JSON object") {
    super(message);
    this.name = "OrgMetadataParseError";
  }
}

export const toNumberOrNull = (value: number | string | null | undefined): number | null => {
  if (value === null || value === undefined || value === "") {
    return null;
  }
  const parsed = typeof value === "number" ? value : Number(value);
  return Number.isFinite(parsed) ? parsed : null;
};

export const parseMetadata = (raw: string | undefined): Record<string, unknown> | null => {
  if (raw === undefined || raw.trim() === "") {
    return null;
  }
  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch {
    throw new OrgMetadataParseError();
  }
  if (parsed === null) {
    return null;
  }
  if (typeof parsed !== "object" || Array.isArray(parsed)) {
    throw new OrgMetadataParseError();
  }
  return parsed as Record<string, unknown>;
};

export const buildOrganizationUpdateV2Payload = (
  values: OrgSettingsFormValues,
  isTouched: (name: string) => boolean,
): OrganizationUpdateV2Body => ({
  ...(isTouched("organization_alias") ? { organization_alias: values.organization_alias } : {}),
  ...(isTouched("models") ? { models: values.models ?? [] } : {}),
  ...(isTouched("tpm_limit") ? { tpm_limit: toNumberOrNull(values.tpm_limit) } : {}),
  ...(isTouched("rpm_limit") ? { rpm_limit: toNumberOrNull(values.rpm_limit) } : {}),
  ...(isTouched("max_budget") ? { max_budget: toNumberOrNull(values.max_budget) } : {}),
  ...(isTouched("budget_duration") ? { budget_duration: values.budget_duration || null } : {}),
  ...(isTouched("metadata") ? { metadata: parseMetadata(values.metadata) } : {}),
  ...(isTouched("vector_stores") || isTouched("mcp_servers_and_groups")
    ? {
        object_permission: {
          vector_stores: values.vector_stores ?? [],
          mcp_servers: values.mcp_servers_and_groups?.servers ?? [],
          mcp_access_groups: values.mcp_servers_and_groups?.accessGroups ?? [],
        },
      }
    : {}),
});
