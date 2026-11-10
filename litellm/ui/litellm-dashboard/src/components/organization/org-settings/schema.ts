import type { TFunction } from "i18next";
import { z } from "zod";

const isBlank = (value: string): boolean => value.trim() === "";

const isJsonObject = (value: string): boolean => {
  try {
    const parsed: unknown = JSON.parse(value);
    return typeof parsed === "object" && parsed !== null && !Array.isArray(parsed);
  } catch {
    return false;
  }
};

const wholeNumberOrEmpty = (t: TFunction) =>
  z
    .string()
    .refine(
      (value) => isBlank(value) || /^\d+$/.test(value.trim()),
      t("organizations.settingsForm.errors.wholeNumber"),
    );

const amountOrEmpty = (t: TFunction) =>
  z
    .string()
    .refine(
      (value) => isBlank(value) || (Number.isFinite(Number(value)) && Number(value) >= 0),
      t("organizations.settingsForm.errors.nonNegativeNumber"),
    );

const buildOrgSettingsShape = (t: TFunction) => ({
  organization_alias: z.string().min(1, t("organizations.settingsForm.errors.nameRequired")),
  models: z.array(z.string()),
  max_budget: amountOrEmpty(t),
  budget_duration: z.string(),
  tpm_limit: wholeNumberOrEmpty(t),
  rpm_limit: wholeNumberOrEmpty(t),
  vector_stores: z.array(z.string()),
  mcp: z.object({
    servers: z.array(z.string()),
    accessGroups: z.array(z.string()),
    toolsets: z.array(z.string()),
  }),
  metadata: z
    .string()
    .refine((value) => isBlank(value) || isJsonObject(value), t("organizations.settingsForm.errors.invalidMetadata")),
});

export const buildOrgSettingsSchema = (t: TFunction) => z.object(buildOrgSettingsShape(t));

export type OrgSettingsFormValues = z.output<ReturnType<typeof buildOrgSettingsSchema>>;
