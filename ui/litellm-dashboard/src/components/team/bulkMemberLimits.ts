import { z } from "zod";
import type { Member } from "@/components/networking";
import type { components } from "@/lib/http/schema";

export type TeamMemberBudgetPatch = components["schemas"]["TeamMemberBudgetPatch"];
export type MemberLimitsPatch = Omit<TeamMemberBudgetPatch, "user_id" | "user_email">;

const blankToNull = (value: string): string | null => (value === "" ? null : value);

const amount = z
  .string()
  .trim()
  .refine((value) => value === "" || (Number.isFinite(Number(value)) && Number(value) >= 0), {
    message: "Enter an amount of 0 or more, or leave blank to clear",
  })
  .transform((value) => (value === "" ? null : Number(value)));

const count = z
  .string()
  .trim()
  .refine((value) => value === "" || (/^\d+$/.test(value) && Number.isSafeInteger(Number(value))), {
    message: "Enter a whole number of 0 or more, or leave blank to clear",
  })
  .transform((value) => (value === "" ? null : Number(value)));

const toggled = <Raw extends z.ZodType, Parsed extends z.ZodType>(raw: Raw, parsed: Parsed) =>
  z.discriminatedUnion("change", [
    z.object({ change: z.literal(false), value: raw }),
    z.object({ change: z.literal(true), value: parsed }),
  ]);

const models = z.array(z.string());

const limitFields = {
  max_budget_in_team: toggled(z.string(), amount),
  budget_duration: toggled(z.string(), z.string().transform(blankToNull)),
  tpm_limit: toggled(z.string(), count),
  rpm_limit: toggled(z.string(), count),
  allowed_models: toggled(
    models,
    models.transform((selected) => (selected.length === 0 ? null : selected)),
  ),
};

export const bulkMemberLimitsSchema = z.object(limitFields).transform(
  (fields): MemberLimitsPatch => ({
    ...(fields.max_budget_in_team.change ? { max_budget_in_team: fields.max_budget_in_team.value } : {}),
    ...(fields.budget_duration.change ? { budget_duration: fields.budget_duration.value } : {}),
    ...(fields.tpm_limit.change ? { tpm_limit: fields.tpm_limit.value } : {}),
    ...(fields.rpm_limit.change ? { rpm_limit: fields.rpm_limit.value } : {}),
    ...(fields.allowed_models.change ? { allowed_models: fields.allowed_models.value } : {}),
  }),
);

export type BulkMemberLimitsFormValues = z.input<typeof bulkMemberLimitsSchema>;
export type BulkMemberLimitField = keyof BulkMemberLimitsFormValues;

export const EMPTY_BULK_MEMBER_LIMITS: BulkMemberLimitsFormValues = {
  max_budget_in_team: { change: false, value: "" },
  budget_duration: { change: false, value: "" },
  tpm_limit: { change: false, value: "" },
  rpm_limit: { change: false, value: "" },
  allowed_models: { change: false, value: [] },
};

const memberRef = (member: Member): Pick<TeamMemberBudgetPatch, "user_id" | "user_email"> =>
  member.user_id ? { user_id: member.user_id } : { user_email: member.user_email ?? null };

export const buildBulkMemberPatches = (
  members: readonly Member[],
  limits: MemberLimitsPatch,
): TeamMemberBudgetPatch[] => members.map((member) => ({ ...memberRef(member), ...limits }));
