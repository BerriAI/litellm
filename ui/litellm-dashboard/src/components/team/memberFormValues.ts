export type MemberFieldValue = string | number | null | undefined | string[];

export type MemberFormValues = Record<string, MemberFieldValue>;

export type MemberFieldType = "input" | "select" | "numerical" | "multi-select" | "budget-duration" | "utc-datetime";

export interface MemberAdditionalField {
  name: string;
  label: string | React.ReactNode;
  type: MemberFieldType;
  options?: Array<{ label: string; value: string }>;
  step?: number;
  min?: number;
  placeholder?: string;
}

export interface MemberFieldsConfig {
  roleOptions: Array<{ label: string; value: string }>;
  defaultRole?: string;
  showEmail?: boolean;
  showUserId?: boolean;
  additionalFields?: Array<MemberAdditionalField>;
}

const NULLABLE_NUMERIC_FIELDS: ReadonlySet<string> = new Set([
  "max_budget_in_team",
  "tpm_limit",
  "rpm_limit",
  "temp_budget_increase",
]);

export const TEMP_BUDGET_PAIR_MESSAGE = "Set both a temporary budget increase and its expiry, or neither";

const isUnset = (value: MemberFieldValue): boolean => value === null || value === undefined || value === "";

export const tempBudgetPairError = (values: MemberFormValues): "temp_budget_increase" | "temp_budget_expiry" | null => {
  const increaseUnset = isUnset(values.temp_budget_increase);
  const expiryUnset = isUnset(values.temp_budget_expiry);
  if (increaseUnset === expiryUnset) return null;
  return increaseUnset ? "temp_budget_increase" : "temp_budget_expiry";
};

export const memberFieldNames = (config: MemberFieldsConfig): string[] => [
  ...(config.showEmail ? ["user_email"] : []),
  ...(config.showUserId ? ["user_id"] : []),
  "role",
  ...(config.additionalFields ?? []).map((field) => field.name),
];

const pickFieldNames = (config: MemberFieldsConfig, source: MemberFormValues): MemberFormValues =>
  Object.fromEntries(memberFieldNames(config).map((name) => [name, source[name]]));

export const buildMemberFormValues = (
  mode: "add" | "edit",
  initialData: MemberFormValues | null | undefined,
  config: MemberFieldsConfig,
): MemberFormValues => {
  if (mode === "edit" && initialData) {
    const seeded: MemberFormValues = {
      ...initialData,
      role: (initialData.role as string) || config.defaultRole,
      max_budget_in_team: initialData.max_budget_in_team ?? null,
      tpm_limit: initialData.tpm_limit ?? null,
      rpm_limit: initialData.rpm_limit ?? null,
      budget_duration: initialData.budget_duration || null,
      allowed_models: initialData.allowed_models || [],
      temp_budget_increase: initialData.temp_budget_increase ?? null,
      temp_budget_expiry: initialData.temp_budget_expiry || null,
    };

    return pickFieldNames(config, seeded);
  }

  return pickFieldNames(config, { role: config.defaultRole || config.roleOptions[0]?.value });
};

const emptyValueForType = (type: MemberFieldType | undefined): MemberFieldValue => {
  switch (type) {
    case "multi-select":
      return [];
    case "numerical":
    case "budget-duration":
    case "utc-datetime":
      return null;
    default:
      return "";
  }
};

export const emptyMemberFormValues = (config: MemberFieldsConfig): MemberFormValues => {
  const typeByName = new Map((config.additionalFields ?? []).map((field) => [field.name, field.type]));
  return Object.fromEntries(memberFieldNames(config).map((name) => [name, emptyValueForType(typeByName.get(name))]));
};

export const buildMemberFormData = (values: MemberFormValues): MemberFormValues =>
  Object.fromEntries(
    Object.entries(values).map(([key, value]) => {
      if (typeof value !== "string") return [key, value];

      const trimmed = value.trim();
      if (trimmed === "" && NULLABLE_NUMERIC_FIELDS.has(key)) return [key, null];

      return [key, trimmed];
    }),
  );
