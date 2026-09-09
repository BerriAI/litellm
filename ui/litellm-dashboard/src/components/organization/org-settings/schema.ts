import { z } from "zod/v4";

const isBlank = (value: string): boolean => value.trim() === "";

const isJsonObject = (value: string): boolean => {
  try {
    const parsed: unknown = JSON.parse(value);
    return typeof parsed === "object" && parsed !== null && !Array.isArray(parsed);
  } catch {
    return false;
  }
};

const wholeNumberOrEmpty = z
  .string()
  .refine((value) => isBlank(value) || /^\d+$/.test(value.trim()), "Must be a non-negative whole number");

const amountOrEmpty = z
  .string()
  .refine(
    (value) => isBlank(value) || (Number.isFinite(Number(value)) && Number(value) >= 0),
    "Must be a non-negative number",
  );

const orgSettingsShape = {
  organization_alias: z.string().min(1, "Please input an organization name"),
  models: z.array(z.string()),
  max_budget: amountOrEmpty,
  budget_duration: z.string(),
  tpm_limit: wholeNumberOrEmpty,
  rpm_limit: wholeNumberOrEmpty,
  vector_stores: z.array(z.string()),
  mcp: z.object({
    servers: z.array(z.string()),
    accessGroups: z.array(z.string()),
    toolsets: z.array(z.string()),
  }),
  metadata: z.string().refine((value) => isBlank(value) || isJsonObject(value), "Metadata must be a valid JSON object"),
};

export const orgSettingsSchema = z.object(orgSettingsShape);

export type OrgSettingsFormValues = z.output<typeof orgSettingsSchema>;
