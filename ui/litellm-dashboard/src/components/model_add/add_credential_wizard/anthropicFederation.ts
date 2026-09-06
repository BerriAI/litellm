export const ANTHROPIC_FEDERATION_FIELDS = [
  {
    key: "anthropic_organization_id",
    label: "Organization ID",
    required: true,
    hint: "The UUID under Settings > Organization in the Claude Console.",
  },
  {
    key: "anthropic_federation_rule_id",
    label: "Federation Rule ID",
    required: true,
    hint: "The fdrl_... id on the rule's detail page under Settings > Workload identity.",
  },
  {
    key: "anthropic_service_account_id",
    label: "Service Account ID",
    required: false,
    hint:
      "The svac_... id of the service account the rule targets, on the same page. Anthropic lists it as required, " +
      "though the exchange works without it when the rule targets a single service account, so fill it in whenever " +
      "you have it.",
  },
  {
    key: "anthropic_workspace_id",
    label: "Workspace ID",
    required: false,
    hint:
      "Required only when the rule is enabled in more than one workspace (for example All workspaces); Anthropic " +
      "then rejects the exchange with a 401 logged as workspace_id_required. Use the wrkspc_... id under " +
      "Settings > Workspaces, or the literal default. Leave blank when the rule is enabled in a single workspace.",
  },
] as const;

export type AnthropicFederationKey = (typeof ANTHROPIC_FEDERATION_FIELDS)[number]["key"];

export const ANTHROPIC_FEDERATION_KEYS: readonly AnthropicFederationKey[] = ANTHROPIC_FEDERATION_FIELDS.map(
  (field) => field.key,
);

const isFederationKey = (key: string): key is AnthropicFederationKey =>
  (ANTHROPIC_FEDERATION_KEYS as readonly string[]).includes(key);

export type AnthropicFederationIds = Readonly<Record<AnthropicFederationKey, string>>;

export interface FederationIdsUpdate {
  readonly credential_values: Readonly<Partial<Record<AnthropicFederationKey, string>>>;
  readonly credential_values_to_delete: readonly AnthropicFederationKey[];
}

const trimmed = (value: unknown): string => (typeof value === "string" ? value.trim() : "");

export const readFederationIds = (values: Readonly<Record<string, unknown>>): AnthropicFederationIds =>
  Object.fromEntries(ANTHROPIC_FEDERATION_FIELDS.map((field) => [field.key, trimmed(values[field.key])])) as Record<
    AnthropicFederationKey,
    string
  >;

export const missingFederationFields = (ids: AnthropicFederationIds): readonly string[] =>
  ANTHROPIC_FEDERATION_FIELDS.filter((field) => field.required && ids[field.key] === "").map((field) => field.label);

/**
 * The PATCH that brings a saved credential in line with the ids entered on the Register issuer
 * step, or null when every id already matches what was saved. Ids the operator cleared since the
 * save are deleted rather than merged over, the same rule the Authentication step applies.
 */
export const federationIdsUpdate = (
  saved: Readonly<Record<string, unknown>>,
  ids: AnthropicFederationIds,
): FederationIdsUpdate | null => {
  const changed = ANTHROPIC_FEDERATION_FIELDS.map((field) => field.key).filter(
    (key) => ids[key] !== trimmed(saved[key]),
  );
  if (changed.length === 0) {
    return null;
  }
  return {
    credential_values: Object.fromEntries(changed.filter((key) => ids[key] !== "").map((key) => [key, ids[key]])),
    credential_values_to_delete: changed.filter((key) => ids[key] === ""),
  };
};

/**
 * The ids a re-save of the Authentication step must leave untouched on a LiteLLM-signed
 * credential: that step no longer mounts them, so they would otherwise read as deletions.
 */
export const savedFederationIds = (saved: Readonly<Record<string, unknown>>): Readonly<Record<string, unknown>> =>
  Object.fromEntries(Object.entries(saved).filter(([key]) => isFederationKey(key)));

export const withFederationIds = (
  saved: Readonly<Record<string, unknown>>,
  ids: AnthropicFederationIds,
): Readonly<Record<string, unknown>> =>
  Object.fromEntries([
    ...Object.entries(saved).filter(([key]) => !(key in ids)),
    ...Object.entries(ids).filter(([, value]) => value !== ""),
  ]);
