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
}

export interface OrgSettingsBaseline {
  organization_alias: string | null;
  models: string[] | null;
  tpm_limit: number | null;
  rpm_limit: number | null;
  max_budget: number | null;
  budget_duration: string | null;
  metadata: Record<string, unknown> | null;
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

const deepEqual = (a: unknown, b: unknown): boolean => {
  if (a === b) {
    return true;
  }
  if (a === null || b === null || typeof a !== "object" || typeof b !== "object") {
    return false;
  }
  if (Array.isArray(a) || Array.isArray(b)) {
    if (!Array.isArray(a) || !Array.isArray(b) || a.length !== b.length) {
      return false;
    }
    return a.every((item, index) => deepEqual(item, b[index]));
  }
  const aKeys = Object.keys(a as Record<string, unknown>);
  const bKeys = Object.keys(b as Record<string, unknown>);
  if (aKeys.length !== bKeys.length) {
    return false;
  }
  return aKeys.every((key) => deepEqual((a as Record<string, unknown>)[key], (b as Record<string, unknown>)[key]));
};

const NUMERIC_FIELDS = ["tpm_limit", "rpm_limit", "max_budget"] as const;

export const buildOrganizationUpdateV2Payload = ({
  values,
  baseline,
}: {
  values: OrgSettingsFormValues;
  baseline: OrgSettingsBaseline;
}): OrganizationUpdateV2Body => {
  const body: OrganizationUpdateV2Body = {};

  if (values.organization_alias !== undefined && values.organization_alias !== baseline.organization_alias) {
    body.organization_alias = values.organization_alias;
  }

  if (values.models !== undefined && !deepEqual(values.models, baseline.models ?? [])) {
    body.models = values.models;
  }

  for (const field of NUMERIC_FIELDS) {
    if (values[field] === undefined) {
      continue;
    }
    const next = toNumberOrNull(values[field]);
    if (next !== baseline[field]) {
      body[field] = next;
    }
  }

  if (values.budget_duration !== undefined) {
    const nextBudgetDuration = values.budget_duration || null;
    if (nextBudgetDuration !== baseline.budget_duration) {
      body.budget_duration = nextBudgetDuration;
    }
  }

  if (values.metadata !== undefined) {
    const nextMetadata = parseMetadata(values.metadata);
    if (!deepEqual(nextMetadata, baseline.metadata)) {
      body.metadata = nextMetadata;
    }
  }

  return body;
};

export interface OrgBaselineSource {
  organization_alias?: string | null;
  models?: string[] | null;
  metadata?: Record<string, unknown> | null;
  litellm_budget_table?: {
    tpm_limit?: number | null;
    rpm_limit?: number | null;
    max_budget?: number | null;
    budget_duration?: string | null;
  } | null;
}

export const buildOrgSettingsBaseline = (org: OrgBaselineSource): OrgSettingsBaseline => ({
  organization_alias: org.organization_alias ?? null,
  models: org.models ?? null,
  tpm_limit: org.litellm_budget_table?.tpm_limit ?? null,
  rpm_limit: org.litellm_budget_table?.rpm_limit ?? null,
  max_budget: org.litellm_budget_table?.max_budget ?? null,
  budget_duration: org.litellm_budget_table?.budget_duration ?? null,
  metadata: org.metadata ?? null,
});
