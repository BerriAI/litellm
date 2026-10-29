import { z } from "zod";
import { numberOrNull } from "@/lib/forms/numberOrNull";

import {
  type MetadataPair,
  metadataObjectToPairs,
  metadataPairsToObject,
} from "../common_components/MetadataKeyValueFields";

export const TEAM_ADMIN_EDITABLE_TEAM_FIELDS_SETTING = "team_admin_editable_team_fields";

export const TEAM_ADMIN_RAISE_MAX_BUDGET_PERMISSION = "raise_max_budget";

export const TEAM_ADMIN_CREATE_MEMBER_KEYS_PERMISSION = "create_member_keys";

export const TEAM_ADMIN_METADATA_FIELD = "metadata";

export const TEAM_ADMIN_EDITING_DISABLED_TITLE = "Team admins cannot edit team settings on this proxy";
export const TEAM_ADMIN_EDITING_DISABLED_DESCRIPTION =
  "Ask a proxy admin to enable fields under Settings > UI > Team admin editable fields.";

const teamAdminEditAccessShape = {
  kind: z.literal("team_admin"),
  editable_fields: z.array(z.string()),
  may_raise_max_budget: z.boolean().catch(false),
  proxy_controlled_metadata_keys: z.array(z.string()).catch([]),
};

const callerEditAccessSchema = z.discriminatedUnion("kind", [
  z.object({ kind: z.literal("unrestricted") }),
  z.object(teamAdminEditAccessShape),
  z.object({ kind: z.literal("team_admin_disabled") }),
  z.object({ kind: z.literal("none") }),
]);

export type CallerEditAccess = z.infer<typeof callerEditAccessSchema>;

export type TeamEditAccess =
  | { readonly kind: "unrestricted" }
  | {
      readonly kind: "team_admin";
      readonly editableFields: ReadonlySet<string>;
      readonly mayRaiseMaxBudget: boolean;
      readonly proxyControlledMetadataKeys: ReadonlySet<string>;
    }
  | { readonly kind: "team_admin_disabled" }
  | { readonly kind: "none" };

const fieldListSchema = z.array(z.string()).catch([]);

export const parseTeamAdminEditableFields = (uiSettingsValues: unknown): readonly string[] => {
  const values = z.record(z.string(), z.unknown()).catch({}).parse(uiSettingsValues);
  return fieldListSchema.parse(values[TEAM_ADMIN_EDITABLE_TEAM_FIELDS_SETTING]);
};

export const parseSupportedTeamAdminEditableFields = (uiSettingsFieldSchema: unknown): readonly string[] => {
  const property = z
    .object({ properties: z.object({ [TEAM_ADMIN_EDITABLE_TEAM_FIELDS_SETTING]: z.object({ items: z.unknown() }) }) })
    .safeParse(uiSettingsFieldSchema);
  if (!property.success) return [];
  const items = z
    .object({ enum: z.unknown() })
    .safeParse(property.data.properties[TEAM_ADMIN_EDITABLE_TEAM_FIELDS_SETTING].items);
  return items.success ? fieldListSchema.parse(items.data.enum) : [];
};

export const TEAM_ADMIN_SETTINGS_FIELDS = [
  "tpm_limit",
  "rpm_limit",
  "max_budget",
  "soft_budget",
  "team_member_budget",
  "team_member_budget_duration",
  "team_member_tpm_limit",
  "team_member_rpm_limit",
] as const;

export type TeamAdminSettingsField = (typeof TEAM_ADMIN_SETTINGS_FIELDS)[number];

type TeamAdminDurationSettingsField = "team_member_budget_duration";

type TeamAdminNumericSettingsField = Exclude<TeamAdminSettingsField, TeamAdminDurationSettingsField>;

export const isTeamAdminDurationSettingsField = (
  field: TeamAdminSettingsField,
): field is TeamAdminDurationSettingsField => field === "team_member_budget_duration";

const TEAM_ADMIN_FIELD_LABELS: ReadonlyMap<string, string> = new Map([
  ["tpm_limit", "Tokens per minute Limit (TPM)"],
  ["rpm_limit", "Requests per minute Limit (RPM)"],
  ["max_budget", "Max Budget (USD)"],
  [TEAM_ADMIN_METADATA_FIELD, "Team Metadata"],
  ["soft_budget", "Soft Budget (USD)"],
  ["team_member_budget", "Member Default Budget (USD)"],
  ["team_member_budget_duration", "Member Default Budget Duration"],
  ["team_member_tpm_limit", "Member Default TPM Limit"],
  ["team_member_rpm_limit", "Member Default RPM Limit"],
  [TEAM_ADMIN_RAISE_MAX_BUDGET_PERMISSION, "Raise the team's max budget"],
  ["projects", "Create and update projects"],
  ["member_key_budgets", "Update budgets on team members' keys"],
  [TEAM_ADMIN_CREATE_MEMBER_KEYS_PERMISSION, "Create keys for team members"],
]);

export const teamAdminFieldLabel = (field: string): string => TEAM_ADMIN_FIELD_LABELS.get(field) ?? field;

export type TeamAdminSettingsValues = { readonly [F in TeamAdminNumericSettingsField]?: string | number | null } & {
  readonly [F in TeamAdminDurationSettingsField]?: string | null;
};

export type TeamAdminSettingsChanges = { readonly [F in TeamAdminNumericSettingsField]?: number | null } & {
  readonly [F in TeamAdminDurationSettingsField]?: string | null;
} & { readonly metadata?: Record<string, unknown> };

const durationOrNull = (value: string | number | null | undefined): string | null =>
  typeof value === "string" && value !== "" ? value : null;

const changedValue = (
  field: TeamAdminSettingsField,
  value: string | number | null | undefined,
  initialValue: string | number | null | undefined,
): readonly [TeamAdminSettingsField, string | number | null] | null => {
  if (!isTeamAdminDurationSettingsField(field)) {
    const number = numberOrNull(value);
    return number === numberOrNull(initialValue) ? null : [field, number];
  }
  if (value === undefined) return null;
  const duration = durationOrNull(value);
  return initialValue !== undefined && duration === durationOrNull(initialValue) ? null : [field, duration];
};

export const teamAdminSettingsChanges = (
  values: TeamAdminSettingsValues,
  initialValues: TeamAdminSettingsValues,
  editableFields: ReadonlySet<string>,
): TeamAdminSettingsChanges =>
  Object.fromEntries(
    TEAM_ADMIN_SETTINGS_FIELDS.flatMap((field) => {
      const change = editableFields.has(field) ? changedValue(field, values[field], initialValues[field]) : null;
      return change === null ? [] : [change];
    }),
  );

export const parseTeamEditAccess = (callerEditAccess: unknown): TeamEditAccess => {
  const parsed = callerEditAccessSchema.safeParse(callerEditAccess);
  if (!parsed.success) return { kind: "none" };
  if (parsed.data.kind === "team_admin") {
    return {
      kind: "team_admin",
      editableFields: new Set(parsed.data.editable_fields),
      mayRaiseMaxBudget: parsed.data.may_raise_max_budget,
      proxyControlledMetadataKeys: new Set(parsed.data.proxy_controlled_metadata_keys),
    };
  }
  return parsed.data;
};

export interface TeamAdminMetadataContext {
  readonly storedMetadata: Readonly<Record<string, unknown>>;
  readonly proxyControlledKeys: ReadonlySet<string>;
  readonly declaredKeys: ReadonlySet<string>;
}

const canonicalJson = (value: unknown): string =>
  JSON.stringify(value, (_key, nested: unknown) =>
    nested !== null && typeof nested === "object" && !Array.isArray(nested)
      ? Object.fromEntries(Object.entries(nested).sort(([a], [b]) => a.localeCompare(b)))
      : nested,
  );

const pairValue = (pair: Partial<MetadataPair> | undefined): unknown =>
  metadataPairsToObject([{ key: "value", value: pair?.value ?? "" }]).value;

const isBlankString = (value: unknown): boolean => typeof value === "string" && value.trim() === "";

const isBlank = (value: unknown): boolean => value === null || value === undefined || isBlankString(value);

const isUnsetDeclaredPair = (pair: Partial<MetadataPair> | undefined, context: TeamAdminMetadataContext): boolean => {
  const key = pair?.key;
  if (key === undefined || !context.declaredKeys.has(key)) return false;
  return !(key in context.storedMetadata) && isBlank(pairValue(pair));
};

export const teamAdminDeclaredFields = <Field extends { readonly key: string }>(
  schemaFields: readonly Field[],
  proxyControlledKeys: ReadonlySet<string>,
): Field[] => schemaFields.filter((field) => !proxyControlledKeys.has(field.key));

export const teamAdminMetadataPairs = (
  metadata: Readonly<Record<string, unknown>> | null | undefined,
  proxyControlledKeys: ReadonlySet<string>,
  declaredKeys: readonly string[],
): MetadataPair[] => {
  const pairs = metadataObjectToPairs(metadata, proxyControlledKeys);
  const declaredRank = (key: string): number => {
    const rank = declaredKeys.indexOf(key);
    return rank === -1 ? declaredKeys.length : rank;
  };
  return pairs.toSorted((a, b) => declaredRank(a.key) - declaredRank(b.key));
};

export const teamAdminMetadataChange = (
  pairs: readonly (Partial<MetadataPair> | undefined)[] | undefined,
  context: TeamAdminMetadataContext,
): { readonly metadata?: Record<string, unknown> } => {
  const edited = metadataPairsToObject((pairs ?? []).filter((pair) => !isUnsetDeclaredPair(pair, context)));
  const proxyControlled = Object.entries(context.storedMetadata).filter(([key]) =>
    context.proxyControlledKeys.has(key),
  );
  const metadata = { ...Object.fromEntries(proxyControlled), ...edited };
  return canonicalJson(metadata) === canonicalJson(context.storedMetadata) ? {} : { metadata };
};

export interface TeamAdminMetadataPairIssue {
  readonly path: "key" | "value";
  readonly message: string;
}

export const teamAdminMetadataPairIssue = (
  pair: Partial<MetadataPair> | undefined,
  context: TeamAdminMetadataContext,
): TeamAdminMetadataPairIssue | null => {
  if (pair?.key === undefined) return null;
  if (context.proxyControlledKeys.has(pair.key)) return { path: "key", message: "Only proxy admins can set this key" };
  if (!context.declaredKeys.has(pair.key) || !(pair.key in context.storedMetadata)) return null;
  const value = pairValue(pair);
  const clearsStoredValue = isBlank(value) && canonicalJson(value) !== canonicalJson(context.storedMetadata[pair.key]);
  return clearsStoredValue ? { path: "value", message: "This field cannot be left empty" } : null;
};
